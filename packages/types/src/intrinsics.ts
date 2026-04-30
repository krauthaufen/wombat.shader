// Built-in functions that translate to IR `CallIntrinsic` nodes.
// The frontend recognises calls to these by name and emits the
// corresponding intrinsic IR node.
//
// All declarations are `declare function` — they have no runtime
// implementation. Calling them outside shader source is a runtime
// error.

import type { V2f, V3f, V4f } from "./vectors.js";

// ─── trig / transcendental ───────────────────────────────────────────

export declare function sin(x: number): number;
export declare function sin<T extends V2f | V3f | V4f>(x: T): T;
export declare function cos(x: number): number;
export declare function cos<T extends V2f | V3f | V4f>(x: T): T;
export declare function tan(x: number): number;
export declare function tan<T extends V2f | V3f | V4f>(x: T): T;
export declare function asin(x: number): number;
export declare function acos(x: number): number;
export declare function atan(x: number): number;
export declare function atan2(y: number, x: number): number;

export declare function sinh(x: number): number;
export declare function sinh<T extends V2f | V3f | V4f>(x: T): T;
export declare function cosh(x: number): number;
export declare function cosh<T extends V2f | V3f | V4f>(x: T): T;
export declare function tanh(x: number): number;
export declare function tanh<T extends V2f | V3f | V4f>(x: T): T;
export declare function asinh(x: number): number;
export declare function acosh(x: number): number;
export declare function atanh(x: number): number;

export declare function degrees(x: number): number;
export declare function degrees<T extends V2f | V3f | V4f>(x: T): T;
export declare function radians(x: number): number;
export declare function radians<T extends V2f | V3f | V4f>(x: T): T;

export declare function trunc(x: number): number;
export declare function trunc<T extends V2f | V3f | V4f>(x: T): T;

export declare function exp(x: number): number;
export declare function exp<T extends V2f | V3f | V4f>(x: T): T;
export declare function exp2(x: number): number;
export declare function log(x: number): number;
export declare function log2(x: number): number;
export declare function pow(x: number, y: number): number;
export declare function pow<T extends V2f | V3f | V4f>(x: T, y: T): T;
export declare function sqrt(x: number): number;
export declare function sqrt<T extends V2f | V3f | V4f>(x: T): T;
export declare function inversesqrt(x: number): number;
export declare function inversesqrt<T extends V2f | V3f | V4f>(x: T): T;

// ─── element-wise utility ───────────────────────────────────────────
//
// `abs` / `sign` / `floor` / `ceil` / `round` / `fract` / `mod` /
// `min` / `max` / `clamp` / `mix` (= `lerp` on V*f) all have method
// forms on wombat.base's vector classes. The intrinsic stays as a
// free function for **scalar** use (`abs(0.5)`); vectors should
// prefer `v.abs()` etc.

export declare function abs(x: number): number;
export declare function sign(x: number): number;
export declare function floor(x: number): number;
export declare function ceil(x: number): number;
export declare function round(x: number): number;
export declare function fract(x: number): number;
export declare function mod(x: number, y: number): number;

export declare function min(x: number, y: number): number;
export declare function max(x: number, y: number): number;
export declare function clamp(x: number, min: number, max: number): number;
export declare function mix(x: number, y: number, t: number): number;

// `step` and `smoothstep` have no V*f method equivalent, so they
// keep their vector overloads.
export declare function step(edge: number, x: number): number;
export declare function step<T extends V2f | V3f | V4f>(edge: T, x: T): T;
export declare function smoothstep(edge0: number, edge1: number, x: number): number;
export declare function smoothstep<T extends V2f | V3f | V4f>(e0: T, e1: T, x: T): T;

// ─── geometric ──────────────────────────────────────────────────────
//
// `length` / `dot` / `cross` / `distance` / `normalize` are provided
// as **methods** on wombat.base's `V*f` (`v.length()`, `a.dot(b)`,
// `a.cross(b)`, `a.distance(b)`, `v.normalize()`) and are no longer
// shipped as free-function intrinsics. The frontend translates the
// method form identically.

export declare function reflect<T extends V2f | V3f | V4f>(i: T, n: T): T;
export declare function refract<T extends V2f | V3f | V4f>(i: T, n: T, eta: number): T;
export declare function faceforward<T extends V2f | V3f | V4f>(n: T, i: T, nref: T): T;

// ─── derivatives (fragment-only) ─────────────────────────────────────

export declare function dFdx(x: number): number;
export declare function dFdx<T extends V2f | V3f | V4f>(x: T): T;
export declare function dFdy(x: number): number;
export declare function dFdy<T extends V2f | V3f | V4f>(x: T): T;
export declare function fwidth(x: number): number;
export declare function fwidth<T extends V2f | V3f | V4f>(x: T): T;

// ─── fragment-only ───────────────────────────────────────────────────

export declare function discard(): never;

// ─── atomics (compute-only, WGSL only) ───────────────────────────────
//
// First argument is a storage-buffer element (`buf[i]` typically). The
// frontend infers atomic typing on the buffer from these calls; the
// WGSL emitter prepends `&` to the first argument. WebGL2 GLSL ES 3.00
// does not expose atomics — using these in a GLSL-targeted entry is a
// compile-time error.

export declare function atomicLoad(target: number): number;
export declare function atomicStore(target: number, value: number): void;
export declare function atomicAdd(target: number, value: number): number;
export declare function atomicSub(target: number, value: number): number;
export declare function atomicMin(target: number, value: number): number;
export declare function atomicMax(target: number, value: number): number;
export declare function atomicAnd(target: number, value: number): number;
export declare function atomicOr(target: number, value: number): number;
export declare function atomicXor(target: number, value: number): number;
export declare function atomicExchange(target: number, value: number): number;
export declare function atomicCompareExchangeWeak(
  target: number, compare: number, value: number,
): { readonly old_value: number; readonly exchanged: boolean };

// ─── workgroup barriers (compute-only) ───────────────────────────────

export declare function workgroupBarrier(): void;
export declare function storageBarrier(): void;

// ─── storage textures (compute / fragment, WGSL only) ────────────────
//
// `textureLoad` reads a texel; `textureStore` writes one. Both take
// the storage-texture binding as the first argument and an integer
// coord (and array layer for 2DArray). WebGL2 GLSL ES 3.00 has no
// surface for these — the GLSL emitter throws.

import type { StorageTexture2D, StorageTexture3D, StorageTexture2DArray } from "./storage.js";
import type { V4i, V4ui } from "./vectors.js";

export declare function textureLoad<F extends string, A extends "read" | "read_write">(
  tex: StorageTexture2D<F, A>, coord: import("./vectors.js").V2i,
): V4f;
export declare function textureLoad<F extends string, A extends "read" | "read_write">(
  tex: StorageTexture3D<F, A>, coord: import("./vectors.js").V3i,
): V4f;
export declare function textureLoad<F extends string, A extends "read" | "read_write">(
  tex: StorageTexture2DArray<F, A>, coord: import("./vectors.js").V2i, layer: number,
): V4f;

export declare function textureStore<F extends string, A extends "write" | "read_write">(
  tex: StorageTexture2D<F, A>, coord: import("./vectors.js").V2i, value: V4f,
): void;
export declare function textureStore<F extends string, A extends "write" | "read_write">(
  tex: StorageTexture3D<F, A>, coord: import("./vectors.js").V3i, value: V4f,
): void;
export declare function textureStore<F extends string, A extends "write" | "read_write">(
  tex: StorageTexture2DArray<F, A>, coord: import("./vectors.js").V2i, layer: number, value: V4f,
): void;
// Touch the int/uint vector imports so TS doesn't drop them; users
// reading int/uint storage textures get the right return type via
// alternative overloads added later.
type _Touch = V4i | V4ui;

// ─── packing / unpacking ─────────────────────────────────────────────
//
// WGSL-style names; the IR emitter maps to GLSL ES 3.00 spellings
// (`packHalf2x16`, `packUnorm4x8`, …). Inputs and outputs are u32 or
// vec*<f32> per the spec.

export declare function pack2x16float(v: V2f): number;
export declare function unpack2x16float(v: number): V2f;
export declare function pack2x16unorm(v: V2f): number;
export declare function unpack2x16unorm(v: number): V2f;
export declare function pack2x16snorm(v: V2f): number;
export declare function unpack2x16snorm(v: number): V2f;
export declare function pack4x8unorm(v: V4f): number;
export declare function unpack4x8unorm(v: number): V4f;
export declare function pack4x8snorm(v: V4f): number;
export declare function unpack4x8snorm(v: number): V4f;

// ─── integer bit ops ─────────────────────────────────────────────────

export declare function countOneBits(x: number): number;
export declare function extractBits(value: number, offset: number, count: number): number;
export declare function insertBits(value: number, replacement: number, offset: number, count: number): number;
export declare function reverseBits(x: number): number;
export declare function firstLeadingBit(x: number): number;
export declare function firstTrailingBit(x: number): number;
