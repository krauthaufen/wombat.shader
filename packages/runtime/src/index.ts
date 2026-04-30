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

// Backend-specific subpaths:
//
//   import { linkEffect } from "@aardworx/wombat.shader-runtime/webgl2";
//   import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";
