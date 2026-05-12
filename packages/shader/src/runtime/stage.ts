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
// via the getters at compile time and get inlined as IR constants
// (FShade specialization-by-capture). The compile cache is keyed on the
// effect's `id` (content hash) + compile options only — NOT on the
// sampled hole values — so the holes for a given `id` are assumed
// invariant across compiles (true for module-level-const captures,
// which is everything in practice). An effect that captures a value
// varying per construction for a fixed `id` would get a stale cached
// shader; pass such a value as a uniform instead.

import {
  combineHashes,
  hashModule,
  prettyPrint,
  type Expr,
  type Module,
  type Stage as IRStage,
  type Type,
  type ValueDef,
} from "../ir/index.js";
import {
  relinkVars,
  renameEntries,
  renameFunctions,
  renameInputsInStage,
  renameOutputsInStage,
  renameTypes,
  renameVaryings,
  resolveHoles,
  substituteInputsInStage,
  type HoleValue,
  type RenameMapping,
} from "../passes/index.js";
import {
  compileModule,
  type CompileOptions,
  type CompiledEffect,
} from "./compile.js";
import type { FragmentOutputLayout } from "../passes/index.js";

/**
 * Stable string key for a `FragmentOutputLayout`. Sorting by name
 * produces a deterministic representation independent of map insertion
 * order, so two layouts with the same (name → location) pairs hit
 * the same cache slot.
 */
function layoutKey(layout: FragmentOutputLayout | undefined): string {
  if (layout === undefined) return "";
  const entries = [...layout.locations.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  return ":fbo[" + entries.map(([n, l]) => `${n}=${l}`).join("|") + "]";
}

/** Cache-key fragment for the `instanceUniforms` pass. `compileModule`
 *  rewrites the named uniforms to per-instance attribute reads when this
 *  is set, so two compiles of the same effect id with different
 *  `instanceAttributes` must NOT share a cache slot. */
function instanceAttrsKey(attrs: ReadonlySet<string> | undefined): string {
  if (attrs === undefined || attrs.size === 0) return "";
  return ":inst[" + [...attrs].sort().join(",") + "]";
}

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

// ─── IR-rewrite spec types ──────────────────────────────────────────────
//
// Used by `Effect.substitute` / `Effect.rename`. Each per-stage spec
// names which scope (input/uniform/builtin/output) to rewrite plus
// the actual mapping. `EffectRename` additionally has module-wide
// keys (types/entries/functions) which apply across every stage.

type ExprMappingSource =
  | ReadonlyMap<string, Expr>
  | ((name: string) => Expr | undefined);

export interface StageSubstitution {
  readonly inputs?:   ExprMappingSource;
  readonly uniforms?: ExprMappingSource;
  readonly builtins?: ExprMappingSource;
}

export interface EffectSubstitution {
  readonly vertex?:   StageSubstitution;
  readonly fragment?: StageSubstitution;
  readonly compute?:  StageSubstitution;
}

export interface StageRename {
  readonly inputs?:   RenameMapping<string, Type>;
  readonly outputs?:  RenameMapping<string, Type>;
  readonly uniforms?: RenameMapping<string, Type>;
  readonly builtins?: RenameMapping<string, Type>;
  readonly types?:    ReadonlyMap<string, string>;
  readonly entries?:  ReadonlyMap<string, string>;
}

export interface EffectRename {
  readonly vertex?:   StageRename;
  readonly fragment?: StageRename;
  readonly compute?:  StageRename;
  /** Applied across every stage's template. */
  readonly types?:     ReadonlyMap<string, string>;
  readonly entries?:   ReadonlyMap<string, string>;
  readonly functions?: ReadonlyMap<string, string>;
  /**
   * Cross-stage varyings — paired VS output + FS input rename. Applied
   * BEFORE the per-stage `vertex.outputs` / `fragment.inputs` renames,
   * so a per-stage entry can override the paired default for one side
   * (rarely needed, but available). If any name in `varyings` also
   * appears as a target in `vertex.outputs` or `fragment.inputs`, that's
   * a configuration error and we throw.
   *
   * Accepts either a name-keyed Map or a `(name, type) => newName |
   * undefined` function — useful when distinct stages of the same
   * effect share varying names but need disambiguation by type.
   */
  readonly varyings?:  RenameMapping<string, Type>;
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
  /**
   * Rewrite every `ReadInput(scope, name)` in the targeted stage's
   * template Module via the supplied mapping(s). The Effect's `id`
   * is recomputed from the rewritten templates, so the per-effect
   * compile cache can't return a pre-rewrite result.
   */
  substitute(spec: EffectSubstitution): Effect;
  /**
   * Rename identifiers in each stage's template Module. Per-stage
   * keys (`vertex` / `fragment` / `compute`) map to a `StageRename`
   * that can rename inputs / outputs / uniforms / builtins / struct
   * types / entries within that stage. Module-wide keys (`types` /
   * `entries` / `functions`) apply across every stage's template.
   */
  rename(spec: EffectRename): Effect;
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
  // Templates emitted by the shader-vite plugin arrive via JSON, which
  // strips Var object identity — every reference to a single Var
  // becomes a fresh duplicate. Relink by name so downstream passes
  // (DCE, CSE, inline) match Vars by reference equality again.
  const linked = relinkVars(template);
  const stageId = id ?? hashModule(linked);
  return makeEffect([{ template: linked, holes, avalHoles, id: stageId }], stageId);
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

// Build a Module-level rewriter that applies one StageSubstitution to
// the given stage. Returns the (possibly identical) rewritten Module.
function applyStageSubstitution(
  template: Module,
  stage: IRStage,
  spec: StageSubstitution | undefined,
): Module {
  if (!spec) return template;
  const toFn = (m: ExprMappingSource | undefined): ((n: string) => Expr | undefined) | undefined => {
    if (!m) return undefined;
    if (typeof m === "function") return m;
    return (n: string) => m.get(n);
  };
  let out = template;
  const ins = toFn(spec.inputs);
  if (ins) out = substituteInputsInStage(out, stage, "Input", ins);
  const uni = toFn(spec.uniforms);
  if (uni) out = substituteInputsInStage(out, stage, "Uniform", uni);
  const bi = toFn(spec.builtins);
  if (bi) out = substituteInputsInStage(out, stage, "Builtin", bi);
  return out;
}

/**
 * Truthy if the supplied mapping has work to do — for Maps that means
 * non-empty, for functions we always pass through (we can't know
 * cheaply whether the function would return undefined for every name,
 * so let the rename pass short-circuit if the materialised map ends
 * up empty).
 */
function hasMappingWork<K, V>(m: RenameMapping<K, V> | undefined): boolean {
  if (m === undefined) return false;
  if (typeof m === "function") return true;
  return m.size > 0;
}

function applyStageRename(
  template: Module,
  stage: IRStage,
  spec: StageRename | undefined,
): Module {
  if (!spec) return template;
  let out = template;
  if (hasMappingWork(spec.inputs)) {
    out = renameInputsInStage(out, stage, "Input", spec.inputs!);
  }
  if (hasMappingWork(spec.outputs)) {
    out = renameOutputsInStage(out, stage, spec.outputs!);
  }
  if (hasMappingWork(spec.uniforms)) {
    out = renameInputsInStage(out, stage, "Uniform", spec.uniforms!);
  }
  if (hasMappingWork(spec.builtins)) {
    out = renameInputsInStage(out, stage, "Builtin", spec.builtins!);
  }
  if (spec.types && spec.types.size > 0) {
    out = renameTypes(out, spec.types);
  }
  if (spec.entries && spec.entries.size > 0) {
    out = renameEntries(out, spec.entries);
  }
  return out;
}

function inferStageOf(template: Module): IRStage | undefined {
  // A wombat Stage holds one Entry (per `vertex(...)` / `fragment(...)`
  // call). For Effect.substitute / Effect.rename we need the stage tag
  // to route the rewrite. If the template has no Entry (theoretical),
  // we conservatively return undefined and skip per-stage routing.
  for (const v of template.values) {
    if (v.kind === "Entry") return v.entry.stage;
  }
  return undefined;
}

function rebuildStage(s: Stage, newTemplate: Module): Stage {
  if (newTemplate === s.template) return s;
  const linked = relinkVars(newTemplate);
  return { template: linked, holes: s.holes, avalHoles: s.avalHoles, id: hashModule(linked) };
}

// Module-level compile cache shared across all Effect instances.
//
// Why module-level rather than per-instance: every Effect transform in
// this codebase derives its result's `id` deterministically from its
// inputs (e.g. `instanceEffect(inner, attrs)` → `combineHashes(inner.id,
// "INST", ...attrs)`; `effect(...)` → `combineHashes(...stage ids)`).
// Two semantically-equivalent transforms applied through different
// wrapper objects produce different *object identity* but the same
// `id`. With a per-instance cache, each fresh wrapper had its own empty
// cache and the first `compile()` call from a fresh wrapper redid the
// full IR pipeline (`compileModule` + all passes) — even though an
// identical compile had already run somewhere else. In the heap-demo-sg
// toggle, this drove 142 `compileModule` invocations per add-half-back
// click (one per textured RO instance). A module-level Map keyed on
// `id + target + skipOpt + layoutKey` collapses that to one miss ever
// per logical effect+options — and that key is cheap to build, so the
// hit path (the 20k-render-objects common case) is just a Map lookup.
//
// Sharing is safe: a fresh `stage(...)` call's `holes` / `avalHoles`
// closures are constructed once per template at JS module-eval time,
// then reused across every `makeEffect` that includes that stage —
// so two Effects with the same id reference the same stage objects
// and therefore the same avalHoles handles. Returning the cached
// `CompiledEffect` (which has the captured `avalBindings`) is the
// correct shared result.
//
// Caveat: anyone constructing an Effect with a non-deterministic id
// (random / identity-based) breaks the sharing contract. All current
// id-producing sites use `combineHashes` over input ids + transform
// parameters and are therefore safe.
const moduleCompileCache: Map<string, CompiledEffect> = new Map();

function makeEffect(stages: readonly Stage[], id: string): Effect {
  return {
    stages, id,
    dumpIR() {
      return stages.map((s, i) => `// stage ${i} (id=${s.id})\n${prettyPrint(s.template)}`).join("\n\n");
    },
    compile(options) {
      // Cache key: the effect's content hash (`id`) plus the compile
      // options. We deliberately do NOT hash the sampled hole values
      // into the key. Holes are closure-captured values from the
      // marker's lexical scope, baked into the WGSL as literals; in
      // every realistic effect they're module-level constants, so for
      // a given `id` they don't vary between compiles. Hashing the
      // (multi-KB) sampled-stage data on *every* `compile()` call was
      // real overhead on the hot path — at 20k render objects this is
      // hit ~40k times (heap eligibility check + heap-spec build).
      // Now the hot path is just a short string concat + a Map lookup;
      // hole sampling + the IR pipeline only run on a genuine miss.
      //
      // Constraint this assumes: anything that makes two effects with
      // the *same `id`* compile to *different* WGSL must be folded into
      // the `id` at the construction site — that already holds for ids
      // (all built via `combineHashes` over input ids + transform
      // params) and is required of holes too. An effect whose holes
      // vary per-instance for a fixed `id` (e.g. a `vertex(...)` marker
      // closing over a function parameter) breaks this — pass that
      // value as a uniform instead.
      //
      // The cache is module-level and never evicted; in practice it's
      // bounded by the app's distinct effect count (tens), so the
      // "leak" is a non-issue.
      const cacheKey = `${id}:${options.target}` +
        (options.skipOptimisations ? ":raw" : "") +
        layoutKey(options.fragmentOutputLayout) +
        instanceAttrsKey(options.instanceAttributes);
      const cached = moduleCompileCache.get(cacheKey);
      if (cached) return cached;

      // Miss — do the work: sample holes, fill templates, compile.
      const sampledPerStage: Record<string, HoleValue>[] = [];
      for (const s of stages) {
        const sampled: Record<string, HoleValue> = {};
        for (const [name, getter] of Object.entries(s.holes)) {
          sampled[name] = getter();
        }
        sampledPerStage.push(sampled);
      }

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
      moduleCompileCache.set(cacheKey, compiled);
      return compiled;
    },
    substitute(spec: EffectSubstitution): Effect {
      const newStages = stages.map((s) => {
        const stageTag = inferStageOf(s.template);
        if (stageTag === undefined) return s;
        const perStage = spec[stageTag];
        const next = applyStageSubstitution(s.template, stageTag, perStage);
        return rebuildStage(s, next);
      });
      const newId = combineHashes(...newStages.map((s) => s.id));
      return makeEffect(newStages, newId);
    },
    rename(spec: EffectRename): Effect {
      // Validate `varyings` against per-stage outputs/inputs ahead of
      // time. If both maps touch the same name (as source or target on
      // the relevant axis) the user almost certainly has a bug — apply
      // order would cause one to silently win or collide downstream;
      // throwing here puts the diagnostic right at the call site.
      //
      // The overlap check is purely structural and only runs when both
      // mappings are concrete Maps. With function-form mappings the
      // domain is effectively the IR's universe, so a true overlap
      // surfaces as a downstream collision-policy throw from the
      // underlying rename pass — sufficient diagnostic.
      if (spec.varyings && typeof spec.varyings !== "function" && spec.varyings.size > 0) {
        const varyingsMap = spec.varyings;
        const checkOverlap = (
          stageMap: RenameMapping<string, Type> | undefined,
          axis: "vertex.outputs" | "fragment.inputs",
        ): void => {
          if (!stageMap || typeof stageMap === "function" || stageMap.size === 0) return;
          const vSrc = new Set(varyingsMap.keys());
          const vDst = new Set(varyingsMap.values());
          for (const [from, to] of stageMap) {
            if (vSrc.has(from) || vDst.has(from) || vSrc.has(to) || vDst.has(to)) {
              throw new Error(
                `Effect.rename: varyings rename + ${axis} overlap on name "${
                  vSrc.has(from) || vDst.has(from) ? from : to
                }"`,
              );
            }
          }
        };
        checkOverlap(spec.vertex?.outputs, "vertex.outputs");
        checkOverlap(spec.fragment?.inputs, "fragment.inputs");
      }

      const newStages = stages.map((s) => {
        const stageTag = inferStageOf(s.template);
        let next = s.template;
        // Paired varyings first — a per-stage override (if any) can then
        // run on the already-renamed name. The collision check above
        // ensures the per-stage map doesn't touch a name varyings just
        // produced or consumed, so this order is safe.
        if (hasMappingWork(spec.varyings)) {
          next = renameVaryings(next, spec.varyings!);
        }
        if (stageTag !== undefined) {
          next = applyStageRename(next, stageTag, spec[stageTag]);
        }
        if (spec.types && spec.types.size > 0) next = renameTypes(next, spec.types);
        if (spec.entries && spec.entries.size > 0) next = renameEntries(next, spec.entries);
        if (spec.functions && spec.functions.size > 0) next = renameFunctions(next, spec.functions);
        return rebuildStage(s, next);
      });
      const newId = combineHashes(...newStages.map((s) => s.id));
      return makeEffect(newStages, newId);
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
export function vertex<I, O = unknown>(_fn: (input: I) => O): Effect;
export function vertex<I, B, O = unknown>(_fn: (input: I, builtins: B) => O): Effect;
export function vertex(_fn: unknown): Effect {
  throw new Error(NOT_PROCESSED);
}
/** See `vertex` — same partial-inference rule for `O`. */
export function fragment<I, O = unknown>(_fn: (input: I) => O): Effect;
export function fragment<I, B, O = unknown>(_fn: (input: I, builtins: B) => O): Effect;
export function fragment(_fn: unknown): Effect {
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
  // Same JSON-roundtrip Var-identity hazard as `stage()` — relink.
  template = relinkVars(template);
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
      // Key by the stage's content hash + options only — see the note
      // in `makeEffect`'s `compile` for why the sampled holes aren't
      // hashed into the key.
      const cacheKey = `${stageId}:${options.target}` +
        (options.skipOptimisations ? ":raw" : "") +
        layoutKey(options.fragmentOutputLayout);
      const cached = cache.get(cacheKey);
      if (cached) return cached;

      const sampled: Record<string, HoleValue> = {};
      for (const [name, getter] of Object.entries(holes)) sampled[name] = getter();
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
