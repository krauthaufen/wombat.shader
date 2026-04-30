// Vector types — frontend translates these to structural IR.
//
// `V2f` etc. exist purely for the type checker. There is no runtime
// implementation; the frontend recognises these classes by their brand
// symbol and lowers references to `Vector(Float, n)` IR types.

declare const __wombatShaderBrand: unique symbol;

// ─── 2-component ─────────────────────────────────────────────────────

export declare class V2b {
  static readonly [__wombatShaderBrand]: "V2b";
  x: boolean; y: boolean;
  constructor(x: boolean, y: boolean);
}

export declare class V2i {
  static readonly [__wombatShaderBrand]: "V2i";
  x: number; y: number;
  constructor(x: number, y: number);
  add(other: V2i): V2i;
  sub(other: V2i): V2i;
  mul(s: number): V2i; mul(other: V2i): V2i;
  div(s: number): V2i; div(other: V2i): V2i;
  neg(): V2i;
  dot(other: V2i): number;
  toArray(): readonly [number, number];
}

export declare class V2u {
  static readonly [__wombatShaderBrand]: "V2u";
  x: number; y: number;
  constructor(x: number, y: number);
  add(other: V2u): V2u;
  sub(other: V2u): V2u;
  mul(s: number): V2u; mul(other: V2u): V2u;
  div(s: number): V2u; div(other: V2u): V2u;
  toArray(): readonly [number, number];
}

export declare class V2f {
  static readonly [__wombatShaderBrand]: "V2f";
  x: number; y: number;
  constructor(x: number, y: number);
  add(other: V2f): V2f;
  sub(other: V2f): V2f;
  mul(s: number): V2f; mul(other: V2f): V2f;
  div(s: number): V2f; div(other: V2f): V2f;
  neg(): V2f;
  dot(other: V2f): number;
  length(): number;
  normalize(): V2f;
  toArray(): readonly [number, number];
}

// ─── 3-component ─────────────────────────────────────────────────────

export declare class V3b {
  static readonly [__wombatShaderBrand]: "V3b";
  x: boolean; y: boolean; z: boolean;
  constructor(x: boolean, y: boolean, z: boolean);
}

export declare class V3i {
  static readonly [__wombatShaderBrand]: "V3i";
  x: number; y: number; z: number;
  constructor(x: number, y: number, z: number);
  add(other: V3i): V3i;
  sub(other: V3i): V3i;
  mul(s: number): V3i; mul(other: V3i): V3i;
  div(s: number): V3i; div(other: V3i): V3i;
  neg(): V3i;
  dot(other: V3i): number;
  cross(other: V3i): V3i;
}

export declare class V3u {
  static readonly [__wombatShaderBrand]: "V3u";
  x: number; y: number; z: number;
  constructor(x: number, y: number, z: number);
  add(other: V3u): V3u;
  sub(other: V3u): V3u;
  mul(s: number): V3u; mul(other: V3u): V3u;
  div(s: number): V3u; div(other: V3u): V3u;
}

export declare class V3f {
  static readonly [__wombatShaderBrand]: "V3f";
  x: number; y: number; z: number;
  /** Read-only swizzle: `v.xyz`. */
  readonly xyz: V3f;
  /** Read-only swizzle: `v.xy`. */
  readonly xy: V2f;
  /** Read-only swizzle: `v.yz`. */
  readonly yz: V2f;
  /** Read-only swizzle: `v.xx` etc. work for any combination. */
  readonly [k: string]: number | V2f | V3f | V4f | unknown;

  constructor(x: number, y: number, z: number);

  add(other: V3f): V3f;
  sub(other: V3f): V3f;
  mul(s: number): V3f; mul(other: V3f): V3f;
  div(s: number): V3f; div(other: V3f): V3f;
  neg(): V3f;

  dot(other: V3f): number;
  cross(other: V3f): V3f;
  length(): number;
  normalize(): V3f;
}

// ─── 4-component ─────────────────────────────────────────────────────

export declare class V4b {
  static readonly [__wombatShaderBrand]: "V4b";
  x: boolean; y: boolean; z: boolean; w: boolean;
  constructor(x: boolean, y: boolean, z: boolean, w: boolean);
}

export declare class V4i {
  static readonly [__wombatShaderBrand]: "V4i";
  x: number; y: number; z: number; w: number;
  constructor(x: number, y: number, z: number, w: number);
  add(other: V4i): V4i;
  sub(other: V4i): V4i;
  mul(s: number): V4i; mul(other: V4i): V4i;
  div(s: number): V4i; div(other: V4i): V4i;
  neg(): V4i;
  dot(other: V4i): number;
}

export declare class V4u {
  static readonly [__wombatShaderBrand]: "V4u";
  x: number; y: number; z: number; w: number;
  constructor(x: number, y: number, z: number, w: number);
  add(other: V4u): V4u;
  sub(other: V4u): V4u;
  mul(s: number): V4u; mul(other: V4u): V4u;
  div(s: number): V4u; div(other: V4u): V4u;
}

export declare class V4f {
  static readonly [__wombatShaderBrand]: "V4f";
  x: number; y: number; z: number; w: number;
  readonly xyz: V3f;
  readonly xy: V2f;

  constructor(x: number, y: number, z: number, w: number);
  /** Promote a V3f + scalar → V4f: `new V4f(v3, w)`. */
  // overloaded constructors are awkward in declare-class; the frontend
  // also recognises `vec4(v3, w)` via `vec4` below.

  add(other: V4f): V4f;
  sub(other: V4f): V4f;
  mul(s: number): V4f; mul(other: V4f): V4f;
  div(s: number): V4f; div(other: V4f): V4f;
  neg(): V4f;

  dot(other: V4f): number;
  length(): number;
  normalize(): V4f;
}

// ─── functional constructors recognised by the frontend ──────────────

export declare function vec2(x: number, y: number): V2f;
export declare function vec3(x: number, y: number, z: number): V3f;
export declare function vec3(xy: V2f, z: number): V3f;
export declare function vec4(x: number, y: number, z: number, w: number): V4f;
export declare function vec4(xyz: V3f, w: number): V4f;
export declare function vec4(xy: V2f, z: number, w: number): V4f;

export declare function ivec2(x: number, y: number): V2i;
export declare function ivec3(x: number, y: number, z: number): V3i;
export declare function ivec4(x: number, y: number, z: number, w: number): V4i;

export declare function uvec2(x: number, y: number): V2u;
export declare function uvec3(x: number, y: number, z: number): V3u;
export declare function uvec4(x: number, y: number, z: number, w: number): V4u;
