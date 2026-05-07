// `compileEffect` — the canonical pipeline:
//
//     TS source
//   → @aardworx/wombat.shader-frontend  (TS Compiler API → IR Module)
//   → composeStages                     (fuse v+v / f+f)
//   → inlinePass + foldConstants        (simplify)
//   → cse + dce                          (dedupe + clean)
//   → pruneCrossStage + reduceUniforms  (drop unread outputs / bindings)
//   → liftReturns                       (`return { outColor }` → WriteOutput)
//   → legaliseTypes(target)             (target-specific lowering)
//   → emitGlsl / emitWgsl
//
// Returns the emitted source plus the runtime-relevant binding map.
//
// For pre-built modules (no TS source — caller hand-built the IR),
// `compileModule` skips the frontend step.

import type { Module, Stage, ValueDef } from "../ir/index.js";
import {
  composeStages,
  cse,
  dce,
  linkHelpers,
  pruneVertexInputs,
  foldConstants,
  inferStorageAccess,
  inlinePass,
  legaliseTypes,
  instanceUniforms,
  liftReturns,
  linkCrossStage,
  linkFragmentOutputs,
  pruneCrossStage,
  reduceUniforms,
  reverseMatrixOps,
  simplifyTranspose,
  type FragmentOutputLayout,
} from "../passes/index.js";
import { emitGlsl, type EmitResult as GlslEmitResult } from "../glsl/index.js";
import { emitWgsl, type EmitResult as WgslEmitResult } from "../wgsl/index.js";
import { buildSourceMap } from "../ir/index.js";
import { parseShader, type EntryRequest } from "../frontend/index.js";
import { buildInterface, type ProgramInterface, type StageSourceInfo } from "./interface.js";

export type Target = "glsl" | "wgsl";

export interface CompiledStage {
  readonly stage: Stage;
  readonly entryName: string;
  readonly source: string;
  readonly bindings: GlslEmitResult["bindings"] | WgslEmitResult["bindings"];
  readonly meta: GlslEmitResult["meta"] | WgslEmitResult["meta"];
  /**
   * v3 source map — one entry per emitted shader source line,
   * pointing back at the originating TypeScript source position.
   * `null` when no spans were available (hand-built IR Modules
   * with no AST origin).
   */
  readonly sourceMap: import("../ir/index.js").SourceMap | null;
}

export interface CompiledEffect {
  readonly target: Target;
  readonly stages: readonly CompiledStage[];
  /**
   * Full backend-ready signature: vertex attributes, fragment outputs,
   * uniform blocks with std140/WGSL-resolved field offsets, samplers,
   * textures, storage buffers, and per-stage source.
   * Modeled after FShade's GLSLProgramInterface.
   */
  readonly interface: ProgramInterface;
  /**
   * Aval-bound uniforms: name → opaque getter. The rendering backend
   * subscribes each getter (typically returning an `aval<T>` from
   * `@aardworx/adaptive`) to drive the corresponding uniform binding
   * at draw time. Empty when no aval-typed captures were present.
   */
  readonly avalBindings: Readonly<Record<string, () => unknown>>;
}

export interface CompileOptions {
  readonly target: Target;
  /** Skip the optimiser passes; useful for debugging. */
  readonly skipOptimisations?: boolean;
  /**
   * Extra module-level ValueDefs to merge with the parsed entries.
   * Use this to declare `Uniform` blocks, `Sampler` / `StorageBuffer`
   * bindings, or auxiliary `Function` decls that the shader source
   * references by name.
   */
  readonly extraValues?: readonly ValueDef[];
  /**
   * Top-level helper-function names to translate alongside the
   * entries. Forwarded to `parseShader`'s `helpers` option — each
   * named helper becomes a `Function` ValueDef in the IR Module
   * and entry-side calls to it lower to a properly-typed
   * `Call(FunctionRef)`.
   */
  readonly helpers?: readonly string[];
  /**
   * Skip the matrix-reversal pass that converts row-major
   * (wombat.base / Aardvark) matrix operations into the column-
   * major form GPUs expect. Default: `false` (reversal applied).
   *
   * Set to `true` if your runtime code already pre-transposes
   * matrices before upload, or you're using
   * `layout(row_major)` UBO qualifiers in GLSL — in those cases
   * the shader expects column-major math and reversal would be
   * wrong.
   */
  readonly skipMatrixReversal?: boolean;
  /** Filename associated with the source (for source maps). */
  readonly file?: string;
  /**
   * Optional target framebuffer output layout. When supplied, the
   * `linkFragmentOutputs` pass runs late in the pipeline:
   *   - each fragment output's `Location` decoration is replaced
   *     with `layout.locations.get(name)`;
   *   - outputs whose names aren't in the layout are dropped, with
   *     their `WriteOutput` statements DCE'd from the entry body;
   *   - builtin outputs (`frag_depth`) pass through unchanged.
   *
   * Effect inputs / uniforms / textures keep their effect-pinned
   * locations — only fragment outputs are re-pinned.
   */
  readonly fragmentOutputLayout?: FragmentOutputLayout;
  /**
   * Auto-instancing rewrite. When set, the named uniforms become
   * per-instance vertex attributes via the `instanceUniforms` IR
   * pass — see `passes/instanceUniforms.ts`. Runs after `liftReturns`
   * but before `composeStages` so subsequent linking / pruning sees
   * the rewritten reads + the synth varyings the rewrite emits.
   *
   * Special-cased trafo aliases (`ModelTrafo` / inverses /
   * `NormalMatrix`) trigger only when `"ModelTrafo"` is in the set;
   * the runtime is then expected to pre-multiply the scope's
   * accumulated `ModelTrafo` into each per-instance trafo and bind
   * the leaf's `ModelTrafo` uniform to identity.
   */
  readonly instanceAttributes?: ReadonlySet<string>;
}

export function compileShaderSource(
  source: string,
  entries: readonly EntryRequest[],
  options: CompileOptions,
): CompiledEffect {
  // Build a name → IR-type table from extraValues so the frontend
  // can resolve free identifiers (uniforms, samplers) to their actual
  // types instead of defaulting to f32.
  const externalTypes = new Map<string, import("../ir/index.js").Type>();
  for (const v of options.extraValues ?? []) {
    if (v.kind === "Uniform") {
      for (const u of v.uniforms) externalTypes.set(u.name, u.type);
    } else if (v.kind === "Sampler" || v.kind === "StorageBuffer") {
      externalTypes.set(v.name, v.kind === "StorageBuffer" ? v.layout : v.type);
    } else if (v.kind === "Constant") {
      externalTypes.set(v.name, v.varType);
    }
  }
  const parsed = parseShader({
    source, entries, externalTypes,
    ...(options.file !== undefined ? { file: options.file } : {}),
    ...(options.helpers !== undefined ? { helpers: options.helpers } : {}),
  });
  const merged: Module = options.extraValues
    ? { ...parsed, values: [...options.extraValues, ...parsed.values] }
    : parsed;
  return compileModule(merged, options);
}

export function compileModule(module: Module, options: CompileOptions): CompiledEffect {
  // liftReturns runs first so the rest of the pipeline sees explicit
  // WriteOutput statements. The carrier annotation that liftReturns
  // matches on is non-enumerable; spread-clones in later passes would
  // strip it.
  let m = liftReturns(module);
  // Auto-instancing rewrite — runs ON the merged module so
  // FS-reads-detection sees the FS entries the VS rewrite needs to
  // synthesise varyings for.
  if (options.instanceAttributes && options.instanceAttributes.size > 0) {
    const uniformTypes = new Map<string, import("../ir/index.js").Type>();
    for (const v of m.values) {
      if (v.kind === "Uniform") for (const u of v.uniforms) uniformTypes.set(u.name, u.type);
    }
    m = instanceUniforms(m, options.instanceAttributes, uniformTypes);
    // Algebraic Transpose-propagation runs BEFORE CSE so we can push
    // `Transpose` through `MatrixFromCols/Rows`, distribute over
    // `MulMatMat`, and cancel `T(T(X))` while the IR is still in its
    // pre-CSE inlined form. Once CSE binds intermediate matrices into
    // local lets, `Transpose(localVar)` becomes opaque to the
    // propagation rules and the transpose has to fall back to a real
    // `transpose(...)` call at emit. (See `passes/simplifyTranspose.ts`
    // for the rules; `passes/reverseMatrixOps.ts` does the row/col-vec
    // form swaps that absorb any leaf-level Transpose that survives.)
    m = simplifyTranspose(m);
  }
  if (!options.skipOptimisations) {
    m = composeStages(m);
    // FShade-style cross-stage linker: matches VS outputs <-> FS inputs
    // by semantic (with name fallback), renames FS inputs to the VS
    // output names, and synthesises auto pass-throughs from VS
    // attributes when a semantic-matched VS output is missing.
    // Runs BEFORE linkFragmentOutputs so FS-input name changes propagate
    // before any further DCE / pruning.
    m = linkCrossStage(m);
  }
  // Pin fragment outputs to the target framebuffer signature, if one
  // was provided. Runs after composition + cross-stage linking and
  // before pruneCrossStage / DCE so dead fragment outputs (and the
  // cross-stage inputs that fed only them) get cleaned up below.
  if (options.fragmentOutputLayout !== undefined) {
    m = linkFragmentOutputs(m, options.fragmentOutputLayout);
  }
  if (!options.skipOptimisations) {
    m = inlinePass(m);
    m = foldConstants(m);
    m = cse(m);
    m = dce(m);
    // Cross-helper liveness DCE — propagates "this wrapper output is
    // dead" / "this State field is unread" backward through the
    // helper-call chain and prunes helper bodies + the State struct.
    // Runs BEFORE pruneCrossStage so that dropping fused-VS outputs
    // (which lose their consumers in linkFragmentOutputs) can shrink
    // the State struct, freeing up dead VS-side reads in turn.
    m = linkHelpers(m);
    // pruneCrossStage runs AFTER linkFragmentOutputs so that dead
    // fragment outputs have already been removed (and their reads
    // DCE'd). The pass is iterative: dropping a VS output may shrink
    // VS-body reads, which can in turn allow dropping more outputs in
    // a downstream sibling — handled by the fixed-point loop inside
    // pruneCrossStage itself.
    m = pruneCrossStage(m);
    // Re-run cross-helper liveness after pruneCrossStage may have
    // dropped more outputs. Cheap when there's nothing left to do
    // (single fixed-point iteration).
    m = linkHelpers(m);
    // Drop declared vertex INPUTS the body no longer reads. Runs
    // AFTER the second `linkHelpers` so state-init writes
    // (`s.X = in.X`) for now-dead state fields are gone — without
    // that ordering the wrapper body still has those init writes
    // referencing inputs we want to drop, and the prune sees
    // their `ReadInput` calls as live.
    m = pruneVertexInputs(m);
    m = reduceUniforms(m);
  }
  // Storage-buffer inference must run after the optimiser passes (which
  // may have eliminated dead writes) and before legaliseTypes (the WGSL
  // backend needs the right element type for binding declarations).
  m = inferStorageAccess(m);
  // Row-major → column-major reversal for matrix ops. Runs late so
  // composeStages / inlinePass / etc. operate on the canonical
  // row-major form; the reversal is a final adjustment for the GPU's
  // memory-layout convention. `simplifyTranspose` already ran above
  // (before CSE); `reverseMatrixOps` absorbs any surviving leaf-level
  // Transpose into the row/col-vec operand-flip rules.
  if (!options.skipMatrixReversal) m = reverseMatrixOps(m);
  m = legaliseTypes(m, options.target);
  return emitAll(m, options.target);
}

function emitAll(module: Module, target: Target): CompiledEffect {
  const stages: CompiledStage[] = [];
  const stageInfos: StageSourceInfo[] = [];
  // Per-source-file content lookup. The frontend records `Span.file`
  // per IR node; the source-map builder needs the file's text to
  // turn byte offsets into (line, col) positions.
  const fileContents = collectSourceContents(module);

  for (const v of module.values) {
    if (v.kind !== "Entry") continue;
    if (target === "glsl") {
      const r = emitGlsl(module, v.entry.name);
      const sourceMap = anyMapped(r.lineSpans)
        ? buildSourceMap({ lineSpans: r.lineSpans, fileContents })
        : null;
      stages.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        bindings: r.bindings, meta: r.meta, sourceMap,
      });
      stageInfos.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        ...(r.meta.workgroupSize ? { workgroupSize: r.meta.workgroupSize } : {}),
      });
    } else {
      const r = emitWgsl(module, v.entry.name);
      const sourceMap = anyMapped(r.lineSpans)
        ? buildSourceMap({ lineSegments: r.lineSegments, fileContents })
        : null;
      stages.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        bindings: r.bindings, meta: r.meta, sourceMap,
      });
      stageInfos.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        ...(r.meta.workgroupSize ? { workgroupSize: r.meta.workgroupSize } : {}),
      });
    }
  }
  const iface = buildInterface({ target, module, stages: stageInfos });
  return { target, stages, interface: iface, avalBindings: {} };
}

function anyMapped(spans: ReadonlyArray<unknown>): boolean {
  for (const s of spans) if (s) return true;
  return false;
}

/**
 * Walk the module's IR for any `span` referencing a source file
 * whose contents we can recover. We don't currently embed sources
 * in the module — the file path is just an identifier from the
 * frontend's `parseShader` `file` option. For now we return an
 * empty map; callers that want `sourcesContent` populated can
 * pass file → contents through a future `compile` option.
 */
function collectSourceContents(_module: Module): Map<string, string> {
  return new Map();
}
