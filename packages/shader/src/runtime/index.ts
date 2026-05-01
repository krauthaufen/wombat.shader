// Public API for `@aardworx/wombat.shader-runtime`.

export {
  compileShaderSource,
  compileModule,
} from "./compile.js";

export type {
  CompileOptions,
  CompiledEffect,
  CompiledStage,
  Target,
} from "./compile.js";

export type {
  ProgramInterface,
  StageInfo,
  AttributeInfo,
  OutputInfo,
  LooseUniformInfo,
  UniformBlockInfo,
  UniformFieldInfo,
  SamplerInfo,
  TextureInfo,
  StorageBufferInfo,
} from "./interface.js";

export type { LayoutInfo, LayoutTarget, FieldLayout } from "./layout.js";
export { computeLayout } from "./layout.js";

export { stage, effect, vertex, fragment, compute, computeShader } from "./stage.js";
export type {
  Stage, Effect, ComputeShader,
  HoleGetter, HoleGetters,
} from "./stage.js";

// Per-output dependency analysis — used by downstream layers
// (e.g. wombat.dom's pick `chooseChain`) to ask "can this effect
// produce semantic X given that the geometry only exposes
// attributes Y?". Lives in passes but exposed at runtime so
// callers can avoid the `/passes` subpath.
export { effectDependencies, type OutputDep } from "../passes/effectDeps.js";

// Backend-specific subpaths:
//
//   import { linkEffect } from "@aardworx/wombat.shader-runtime/webgl2";
//   import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";
