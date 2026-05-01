// Sampler / Texture types. The IR splits these (WGSL needs them
// separate); for GLSL targets the frontend / legaliseTypes folds a
// matched `Sampler*` + `Texture*` declaration into a combined `sampler*`.

import type { V2f, V3f, V4f } from "./vectors.js";


// ─── Combined GLSL-style samplers (the most common shape users write) ─

export declare class Sampler2D {
  static readonly __aardworxShaderBrand:"Sampler2D";
}

export declare class SamplerCube {
  static readonly __aardworxShaderBrand:"SamplerCube";
}

export declare class Sampler3D {
  static readonly __aardworxShaderBrand:"Sampler3D";
}

export declare class Sampler2DArray {
  static readonly __aardworxShaderBrand:"Sampler2DArray";
}

export declare class SamplerCubeArray {
  static readonly __aardworxShaderBrand:"SamplerCubeArray";
}

// Integer / unsigned-integer samplers
export declare class ISampler2D { static readonly __aardworxShaderBrand:"ISampler2D"; }
export declare class USampler2D { static readonly __aardworxShaderBrand:"USampler2D"; }

// Comparison samplers (depth comparison)
export declare class Sampler2DShadow {
  static readonly __aardworxShaderBrand:"Sampler2DShadow";
}

// Multisample samplers — WebGPU only. WGSL emits as
// `texture_multisampled_2d<T>`; GLSL ES 3.00 has no surface for
// these and the GLSL emitter throws.
export declare class Sampler2DMS  { static readonly __aardworxShaderBrand:"Sampler2DMS"; }
export declare class ISampler2DMS { static readonly __aardworxShaderBrand:"ISampler2DMS"; }
export declare class USampler2DMS { static readonly __aardworxShaderBrand:"USampler2DMS"; }

// ─── Intrinsic functions on samplers ─────────────────────────────────

export declare function texture(s: Sampler2D, uv: V2f): V4f;
export declare function texture(s: Sampler3D, uvw: V3f): V4f;
export declare function texture(s: SamplerCube, dir: V3f): V4f;
export declare function texture(s: Sampler2DArray, uvLayer: V3f): V4f;

export declare function textureLod(s: Sampler2D, uv: V2f, lod: number): V4f;
export declare function textureLod(s: SamplerCube, dir: V3f, lod: number): V4f;

export declare function textureGrad(
  s: Sampler2D, uv: V2f, dPdx: V2f, dPdy: V2f,
): V4f;

export declare function texelFetch(s: Sampler2D, ij: import("./vectors.js").V2i, lod: number): V4f;
export declare function texelFetch(s: Sampler2DMS, ij: import("./vectors.js").V2i, sample: number): V4f;
export declare function texelFetch(s: ISampler2DMS, ij: import("./vectors.js").V2i, sample: number): import("./vectors.js").V4i;
export declare function texelFetch(s: USampler2DMS, ij: import("./vectors.js").V2i, sample: number): import("./vectors.js").V4ui;

// Comparison sampling (depth/shadow). WGSL: `textureSampleCompare`;
// GLSL ES 3.00: a regular `texture(...)` call on a `sampler*Shadow`,
// which returns a scalar in [0,1].
export declare function textureSampleCompare(
  s: Sampler2DShadow, uv: V2f, depthRef: number,
): number;

export declare function textureSize(s: Sampler2D, lod: number): import("./vectors.js").V2i;
export declare function textureSize(s: Sampler3D, lod: number): import("./vectors.js").V3i;
