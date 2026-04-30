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

import type { Module, Stage } from "@aardworx/wombat.shader-ir";
import {
  composeStages,
  cse,
  dce,
  foldConstants,
  inlinePass,
  legaliseTypes,
  liftReturns,
  pruneCrossStage,
  reduceUniforms,
} from "@aardworx/wombat.shader-passes";
import { emitGlsl, type EmitResult as GlslEmitResult } from "@aardworx/wombat.shader-glsl";
import { emitWgsl, type EmitResult as WgslEmitResult } from "@aardworx/wombat.shader-wgsl";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader-frontend";
import { buildInterface, type ProgramInterface, type StageInfo } from "./interface.js";

export type Target = "glsl" | "wgsl";

export interface CompiledStage {
  readonly stage: Stage;
  readonly entryName: string;
  readonly source: string;
  readonly bindings: GlslEmitResult["bindings"] | WgslEmitResult["bindings"];
  readonly meta: GlslEmitResult["meta"] | WgslEmitResult["meta"];
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
}

export interface CompileOptions {
  readonly target: Target;
  /** Skip the optimiser passes; useful for debugging. */
  readonly skipOptimisations?: boolean;
}

export function compileShaderSource(
  source: string,
  entries: readonly EntryRequest[],
  options: CompileOptions,
): CompiledEffect {
  const module = parseShader({ source, entries });
  return compileModule(module, options);
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
  m = legaliseTypes(m, options.target);
  return emitAll(m, options.target);
}

function emitAll(module: Module, target: Target): CompiledEffect {
  const stages: CompiledStage[] = [];
  const stageInfos: StageInfo[] = [];
  for (const v of module.values) {
    if (v.kind !== "Entry") continue;
    if (target === "glsl") {
      const r = emitGlsl(module, v.entry.name);
      stages.push({ stage: v.entry.stage, entryName: v.entry.name, source: r.source, bindings: r.bindings, meta: r.meta });
      stageInfos.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        ...(r.meta.workgroupSize ? { workgroupSize: r.meta.workgroupSize } : {}),
      });
    } else {
      const r = emitWgsl(module, v.entry.name);
      stages.push({ stage: v.entry.stage, entryName: v.entry.name, source: r.source, bindings: r.bindings, meta: r.meta });
      stageInfos.push({
        stage: v.entry.stage, entryName: v.entry.name, source: r.source,
        ...(r.meta.workgroupSize ? { workgroupSize: r.meta.workgroupSize } : {}),
      });
    }
  }
  const iface = buildInterface({ target, module, stages: stageInfos });
  return { target, stages, interface: iface };
}
