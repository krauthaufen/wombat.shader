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

export declare function abs(x: number): number;
export declare function abs<T extends V2f | V3f | V4f>(x: T): T;
export declare function sign(x: number): number;
export declare function sign<T extends V2f | V3f | V4f>(x: T): T;
export declare function floor(x: number): number;
export declare function floor<T extends V2f | V3f | V4f>(x: T): T;
export declare function ceil(x: number): number;
export declare function ceil<T extends V2f | V3f | V4f>(x: T): T;
export declare function round(x: number): number;
export declare function fract(x: number): number;
export declare function fract<T extends V2f | V3f | V4f>(x: T): T;
export declare function mod(x: number, y: number): number;
export declare function mod<T extends V2f | V3f | V4f>(x: T, y: T): T;
export declare function mod<T extends V2f | V3f | V4f>(x: T, y: number): T;

export declare function min(x: number, y: number): number;
export declare function min<T extends V2f | V3f | V4f>(x: T, y: T): T;
export declare function max(x: number, y: number): number;
export declare function max<T extends V2f | V3f | V4f>(x: T, y: T): T;
export declare function clamp(x: number, min: number, max: number): number;
export declare function clamp<T extends V2f | V3f | V4f>(x: T, min: T, max: T): T;

export declare function mix(x: number, y: number, t: number): number;
export declare function mix<T extends V2f | V3f | V4f>(x: T, y: T, t: number): T;
export declare function mix<T extends V2f | V3f | V4f>(x: T, y: T, t: T): T;

export declare function step(edge: number, x: number): number;
export declare function step<T extends V2f | V3f | V4f>(edge: T, x: T): T;
export declare function smoothstep(edge0: number, edge1: number, x: number): number;
export declare function smoothstep<T extends V2f | V3f | V4f>(e0: T, e1: T, x: T): T;

// ─── geometric ──────────────────────────────────────────────────────

export declare function length(v: V2f): number;
export declare function length(v: V3f): number;
export declare function length(v: V4f): number;
export declare function dot(a: V2f, b: V2f): number;
export declare function dot(a: V3f, b: V3f): number;
export declare function dot(a: V4f, b: V4f): number;
export declare function cross(a: V3f, b: V3f): V3f;
export declare function normalize<T extends V2f | V3f | V4f>(v: T): T;
export declare function distance(a: V2f, b: V2f): number;
export declare function distance(a: V3f, b: V3f): number;
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
