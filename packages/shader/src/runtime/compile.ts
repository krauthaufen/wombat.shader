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
  foldConstants,
  inferStorageAccess,
  inlinePass,
  legaliseTypes,
  liftReturns,
  pruneCrossStage,
  reduceUniforms,
  reverseMatrixOps,
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
  if (!options.skipOptimisations) {
    m = composeStages(m);
    m = inlinePass(m);
    m = foldConstants(m);
    m = cse(m);
    m = dce(m);
    m = pruneCrossStage(m);
    m = reduceUniforms(m);
  }
  // Storage-buffer inference must run after the optimiser passes (which
  // may have eliminated dead writes) and before legaliseTypes (the WGSL
  // backend needs the right element type for binding declarations).
  m = inferStorageAccess(m);
  // Row-major → column-major reversal for matrix ops. Runs late so
  // composeStages / inlinePass / etc. operate on the canonical
  // row-major form; the reversal is a final adjustment for the GPU's
  // memory-layout convention.
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
