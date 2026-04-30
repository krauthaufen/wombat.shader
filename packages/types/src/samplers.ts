// Sampler / Texture types. The IR splits these (WGSL needs them
// separate); for GLSL targets the frontend / legaliseTypes folds a
// matched `Sampler*` + `Texture*` declaration into a combined `sampler*`.

import type { V2f, V3f, V4f } from "./vectors.js";

declare const __wombatShaderBrand: unique symbol;

// ─── Combined GLSL-style samplers (the most common shape users write) ─

export declare class Sampler2D {
  static readonly [__wombatShaderBrand]: "Sampler2D";
}

export declare class SamplerCube {
  static readonly [__wombatShaderBrand]: "SamplerCube";
}

export declare class Sampler3D {
  static readonly [__wombatShaderBrand]: "Sampler3D";
}

export declare class Sampler2DArray {
  static readonly [__wombatShaderBrand]: "Sampler2DArray";
}

export declare class SamplerCubeArray {
  static readonly [__wombatShaderBrand]: "SamplerCubeArray";
}

// Integer / unsigned-integer samplers
export declare class ISampler2D { static readonly [__wombatShaderBrand]: "ISampler2D"; }
export declare class USampler2D { static readonly [__wombatShaderBrand]: "USampler2D"; }

// Comparison samplers (depth comparison)
export declare class Sampler2DShadow {
  static readonly [__wombatShaderBrand]: "Sampler2DShadow";
}

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

export declare function textureSize(s: Sampler2D, lod: number): import("./vectors.js").V2i;
export declare function textureSize(s: Sampler3D, lod: number): import("./vectors.js").V3i;
