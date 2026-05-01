// Public API of `@aardworx/wombat.shader`.
//
// Re-exports the runtime layer — the user-facing API for compiling
// shader source through the frontend, the optimisation passes, and
// the WGSL/GLSL emitters.
//
// Subpath exports give advanced consumers access to the layered
// pieces without pulling everything into the main entry:
//
//   import { ... } from "@aardworx/wombat.shader";        // runtime
//   import { ... } from "@aardworx/wombat.shader/ir";     // IR types/visitors
//   import { ... } from "@aardworx/wombat.shader/wgsl";   // IR → WGSL
//   import { ... } from "@aardworx/wombat.shader/glsl";   // IR → GLSL ES 3.00
//   import { ... } from "@aardworx/wombat.shader/frontend"; // TS source → IR
//   import { ... } from "@aardworx/wombat.shader/passes";   // optimiser passes
//   import { ... } from "@aardworx/wombat.shader/webgpu";   // ShaderModule helpers
//   import { ... } from "@aardworx/wombat.shader/webgl2";   // WebGL2 program linking

export * from "./runtime/index.js";
