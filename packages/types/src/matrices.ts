// Matrix types — frontend translates these to structural IR
// `Matrix(Float, R, C)`. As with vectors, no runtime; method calls
// are pattern-matched by the frontend.
//
// Naming convention: `M{R}{C}f` means R rows × C columns of f32.

import type { V2f, V3f, V4f } from "./vectors.js";

declare const __wombatShaderBrand: unique symbol;

// ─── square matrices ─────────────────────────────────────────────────

export declare class M22f {
  static readonly [__wombatShaderBrand]: "M22f";
  add(other: M22f): M22f;
  sub(other: M22f): M22f;
  mul(s: number): M22f; mul(other: M22f): M22f; mul(v: V2f): V2f;
  neg(): M22f;
  transpose(): M22f;
  inverse(): M22f;
  determinant(): number;
}

export declare class M33f {
  static readonly [__wombatShaderBrand]: "M33f";
  add(other: M33f): M33f;
  sub(other: M33f): M33f;
  mul(s: number): M33f; mul(other: M33f): M33f; mul(v: V3f): V3f;
  neg(): M33f;
  transpose(): M33f;
  inverse(): M33f;
  determinant(): number;
}

export declare class M44f {
  static readonly [__wombatShaderBrand]: "M44f";
  add(other: M44f): M44f;
  sub(other: M44f): M44f;
  mul(s: number): M44f; mul(other: M44f): M44f; mul(v: V4f): V4f;
  neg(): M44f;
  transpose(): M44f;
  inverse(): M44f;
  determinant(): number;
}

// ─── rectangular matrices (less common but supported by the IR) ─────
//
// Naming: M{R}{C}f, transformation rule v_out = m * v_in maps a length-C
// input vector to a length-R output.

export declare class M23f {
  static readonly [__wombatShaderBrand]: "M23f";
  mul(v: V3f): V2f;
  mul(s: number): M23f;
}

export declare class M24f {
  static readonly [__wombatShaderBrand]: "M24f";
  mul(v: V4f): V2f;
  mul(s: number): M24f;
}

export declare class M32f {
  static readonly [__wombatShaderBrand]: "M32f";
  mul(v: V2f): V3f;
  mul(s: number): M32f;
}

export declare class M34f {
  static readonly [__wombatShaderBrand]: "M34f";
  mul(v: V4f): V3f;
  mul(s: number): M34f;
}

export declare class M42f {
  static readonly [__wombatShaderBrand]: "M42f";
  mul(v: V2f): V4f;
  mul(s: number): M42f;
}

export declare class M43f {
  static readonly [__wombatShaderBrand]: "M43f";
  mul(v: V3f): V4f;
  mul(s: number): M43f;
}

// ─── functional constructors ────────────────────────────────────────

export declare function mat2(c0: V2f, c1: V2f): M22f;
export declare function mat3(c0: V3f, c1: V3f, c2: V3f): M33f;
export declare function mat4(c0: V4f, c1: V4f, c2: V4f, c3: V4f): M44f;
