// Public API for `@aardworx/wombat.shader-types`.
//
// What this package provides:
//
//  - `.d.ts` declarations for vector / matrix / sampler types users
//    write in shader source. There is no runtime: calling these in
//    plain JavaScript throws (or worse, returns `undefined`). The
//    frontend recognises them by their `__wombatShaderBrand` symbol
//    and translates references to structural IR (`Vector(Float, n)`,
//    `Matrix(Float, R, C)`, `Sampler`, …).
//
//  - Functional constructors (`vec3`, `mat4`, …) that the frontend
//    treats as the canonical way to build vectors / matrices, mapping
//    to `NewVector` / `NewMatrix` / `MatrixFromCols` IR nodes.
//
//  - Built-in functions (`sin`, `mix`, `texture`, `dFdx`, …) that the
//    frontend translates to `CallIntrinsic` IR nodes.
//
//  - Stage-builtin parameter shapes (`VertexBuiltinIn`, …) that the
//    frontend reads to figure out which `@builtin(...)` decorations
//    to emit.

export * from "./vectors.js";
export * from "./matrices.js";
export * from "./samplers.js";
export * from "./intrinsics.js";
export * from "./builtins.js";
