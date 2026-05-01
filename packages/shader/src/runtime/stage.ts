// Effect — the user-facing runtime object that holds a list of stages
// (each = IR template + closure-getter map). The build plugin emits one
// stage per `vertex(...)`/`fragment(...)`/`compute(...)` call; multiple
// effects compose via `effect(...effects)` which flattens their stages.
//
// At compile time (`compile({ target })`), the runtime:
//
//   1. Samples each stage's closure-getters into concrete values.
//   2. Runs `resolveHoles(template, sampledValues)` per stage — turning
//      every `ReadInput("Closure")` into an inlined IR constant (the
//      FShade specialization-by-capture pattern).
//   3. Concatenates stage `ValueDef`s into one Module and hands that to
//      `compileModule`, which runs the optimiser passes (including
//      `composeStages` for v+v / f+f fuse) and emits.
//
// Only constant holes are supported here. The IR template carries
// `ReadInput("Closure", name, type)` placeholders; the values flow in
// via the getters, so two compiles of the same Effect with different
// captured values produce different specialised shaders (a deliberate
// behavior — runtime uniform binding is a separate, later opt-in).

import {
  combineHashes,
  hashModule,
  hashValue,
  prettyPrint,
  type Module,
  type ValueDef,
} from "../ir/index.js";
import { resolveHoles, type HoleValue } from "../passes/index.js";
import {
  compileModule,
  type CompileOptions,
  type CompiledEffect,
} from "./compile.js";

export type HoleGetter = () => HoleValue;
export type HoleGetters = Readonly<Record<string, HoleGetter>>;

/**
 * Aval-binding getters: opaque to the runtime. Each getter returns
 * an `aval<T>` (or any value the future rendering backend
 * understands as a uniform binding source). The runtime never reads
 * the value — it only retains the getter so the backend can
 * subscribe to changes and write the GPU buffer slot.
 */
export type AvalGetter = () => unknown;
export type AvalGetters = Readonly<Record<string, AvalGetter>>;

/** One IR template + its closure-getter map. Internal building block. */
export interface Stage {
  readonly template: Module;
  readonly holes: HoleGetters;
  readonly avalHoles: AvalGetters;
  /** Build-time stable hash of the template (= shape + types). */
  readonly id: string;
}

/**
 * User-facing object: a list of stages plus a render-time compile.
 *
 * `id` is a build-time stable hash of the underlying templates'
 * shapes. Two effects produced from the same source ASTs and the
 * same captured-value *types* (closure values themselves don't move
 * the id) end up with the same id — so backends can key a GLSL/WGSL
 * cache off it.
 *
 * `effect(...effects)` composes the input ids deterministically so
 * `effect(v, f)` always has the same id when `v` and `f` do.
 */
export interface Effect {
  readonly stages: readonly Stage[];
  readonly id: string;
  compile(options: CompileOptions): CompiledEffect;
  /**
   * Human-readable dump of every stage's IR template. Used during
   * pass development and to debug "why is the emitter producing X."
   * Closure holes appear as `Closure.<name>`; aval / sampler bindings
   * appear as `Uniform.<name>`. Hole values aren't substituted (this
   * is the pre-`compile` template).
   */
  dumpIR(): string;
}

/**
 * Build an Effect from a single stage. Plugin-emitted code calls
 * `__wombat_stage(template, holes, id, avalHoles)`:
 *   - `holes`     — closure-captured values to specialize as constants.
 *   - `id`        — build-time stable hash of `template`. If omitted
 *                   the runtime hashes; that path is for tests.
 *   - `avalHoles` — aval-bound uniforms preserved on the Effect for
 *                   the future rendering backend.
 */
export function stage(
  template: Module,
  holes: HoleGetters = {},
  id?: string,
  avalHoles: AvalGetters = {},
): Effect {
  const stageId = id ?? hashModule(template);
  return makeEffect([{ template, holes, avalHoles, id: stageId }], stageId);
}

/**
 * Compose any number of effects into one. Stage lists are flattened
 * in argument order; the optimiser's `composeStages` pass fuses v+v
 * / f+f pairs at compile time, leaving v→f as the standard pipeline.
 * The composed effect's id is `combineHashes(...ids)` so the same
 * inputs in the same order always give the same id.
 */
export function effect(...effects: readonly Effect[]): Effect {
  const allStages: Stage[] = [];
  for (const e of effects) allStages.push(...e.stages);
  const composedId = combineHashes(...effects.map((e) => e.id));
  return makeEffect(allStages, composedId);
}

function makeEffect(stages: readonly Stage[], id: string): Effect {
  // Per-effect cache: key = effect.id + resolved-hole-values hash +
  // target. Closure values are sampled fresh each compile, so two
  // compile() calls with the same captured values hit the cache;
  // the moment any captured value changes the key changes too and
  // a fresh emit is performed.
  const cache = new Map<string, CompiledEffect>();
  return {
    stages, id,
    dumpIR() {
      return stages.map((s, i) => `// stage ${i} (id=${s.id})\n${prettyPrint(s.template)}`).join("\n\n");
    },
    compile(options) {
      const sampledPerStage: Record<string, HoleValue>[] = [];
      for (const s of stages) {
        const sampled: Record<string, HoleValue> = {};
        for (const [name, getter] of Object.entries(s.holes)) {
          sampled[name] = getter();
        }
        sampledPerStage.push(sampled);
      }
      const cacheKey = `${id}:${hashValue(sampledPerStage)}:${options.target}` +
        (options.skipOptimisations ? ":raw" : "");
      const cached = cache.get(cacheKey);
      if (cached) return cached;

      const allValues: ValueDef[] = [];
      for (let i = 0; i < stages.length; i++) {
        const s = stages[i]!;
        const filled = resolveHoles(s.template, sampledPerStage[i]!);
        allValues.push(...filled.values);
      }
      const merged: Module = { types: [], values: allValues };
      const baseCompiled = compileModule(merged, options);
      // Attach the merged aval-getter map so the rendering backend
      // can subscribe per-uniform. Same name across multiple stages
      // resolves to one entry (last-write-wins; same-named avals
      // across stages should be the same handle anyway).
      const avalBindings: Record<string, AvalGetter> = {};
      for (const s of stages) {
        for (const [name, getter] of Object.entries(s.avalHoles)) {
          avalBindings[name] = getter;
        }
      }
      const compiled: CompiledEffect = { ...baseCompiled, avalBindings };
      cache.set(cacheKey, compiled);
      return compiled;
    },
  };
}

// ─────────────────────────────────────────────────────────────────────
// Source-level markers
//
// Zero-runtime stubs. The Vite plugin scans for calls to these and
// replaces each call site at build time with a `__wombat_stage(...)`
// literal — which is the same `stage()` function above and produces
// an `Effect`. If a call survives to runtime, the marker throws —
// that means the build plugin didn't run.
// ─────────────────────────────────────────────────────────────────────

const NOT_PROCESSED =
  "wombat.shader: this `vertex/fragment/compute` call was not processed by the build plugin. " +
  "Add `@aardworx/wombat.shader-vite` to your Vite config so the plugin scans your sources.";

/**
 * Marker — replaced at build time. Returns a single-stage Effect.
 * `O` defaults to `unknown` so callers can specify just the input
 * type (`vertex<{ a_position: V2f }>(...)`) and let the build
 * plugin infer the output shape from the lambda body. Specifying
 * both is also fine (`vertex<I, O>(...)`); leaving both off is too
 * (full inference from the lambda).
 */
export function vertex<I, O = unknown>(_fn: (input: I) => O): Effect {
  throw new Error(NOT_PROCESSED);
}
/** See `vertex` — same partial-inference rule for `O`. */
export function fragment<I, O = unknown>(_fn: (input: I) => O): Effect {
  throw new Error(NOT_PROCESSED);
}
/**
 * Marker — replaced at build time. Returns a `ComputeShader`,
 * which is its own peer to `Effect`: a compute pipeline always has
 * exactly one stage, so it doesn't need the multi-stage composition
 * machinery `Effect` carries. Consumed by the rendering layer's
 * imperative compute API (input bindings + dispatch).
 */
export function compute<B>(_fn: (builtins: B) => void): ComputeShader {
  throw new Error(NOT_PROCESSED);
}

/**
 * User-facing object for a compute pipeline. One stage, plus
 * `compile(options)` returning a `CompiledEffect` (whose
 * `stages` has length 1 with `stage === "compute"`). The id is the
 * stage's IR-shape hash; stable across builds.
 *
 * The runtime uses the same closure-getter / aval-getter machinery
 * as `Effect`, just specialised to a single stage. The build
 * plugin emits `__wombat_compute(template, holes, id, avalHoles)`
 * which produces this object.
 */
export interface ComputeShader {
  readonly stage: Stage;
  readonly id: string;
  compile(options: CompileOptions): CompiledEffect;
  /** Pretty-printed IR template for debugging. */
  dumpIR(): string;
}

/**
 * Build a `ComputeShader` from a single compute stage. Plugin-emitted
 * code calls `__wombat_compute(template, holes, id, avalHoles)` —
 * mirrors `stage(...)` for the graphics path.
 */
export function computeShader(
  template: Module,
  holes: HoleGetters = {},
  id?: string,
  avalHoles: AvalGetters = {},
): ComputeShader {
  const stageId = id ?? hashModule(template);
  const stage: Stage = { template, holes, avalHoles, id: stageId };
  const cache = new Map<string, CompiledEffect>();
  return {
    stage,
    id: stageId,
    dumpIR() {
      return `// compute stage (id=${stageId})\n${prettyPrint(template)}`;
    },
    compile(options) {
      const sampled: Record<string, HoleValue> = {};
      for (const [name, getter] of Object.entries(holes)) sampled[name] = getter();
      const cacheKey = `${stageId}:${hashValue(sampled)}:${options.target}` +
        (options.skipOptimisations ? ":raw" : "");
      const cached = cache.get(cacheKey);
      if (cached) return cached;

      const filled = resolveHoles(template, sampled);
      const merged: Module = { types: [], values: filled.values };
      const baseCompiled = compileModule(merged, options);
      const avalBindings: Record<string, AvalGetter> = {};
      for (const [name, getter] of Object.entries(avalHoles)) avalBindings[name] = getter;
      const compiled: CompiledEffect = { ...baseCompiled, avalBindings };
      cache.set(cacheKey, compiled);
      return compiled;
    },
  };
}
