// Branded scalar aliases.
//
// TypeScript has only `number`, which the frontend defaults to `f32`.
// To declare an int / unsigned-int uniform or capture, annotate it
// with one of these branded aliases:
//
//   declare const time: f32;
//   declare const frame: i32;
//   declare const flags: u32;
//   const animFrame: aval<i32> = ...;
//
// At runtime they're plain numbers — the brand is purely for the
// type checker, and the frontend's `tryResolveTypeName` table maps
// the alias names to the right IR scalar.

declare const __scalarBrand: unique symbol;

/** 32-bit signed integer. Plain `number` at runtime. */
export type i32 = number & { readonly [__scalarBrand]?: "i32" };
/** 32-bit unsigned integer. Plain `number` at runtime. */
export type u32 = number & { readonly [__scalarBrand]?: "u32" };
/** 32-bit float. Plain `number` at runtime. */
export type f32 = number & { readonly [__scalarBrand]?: "f32" };

// ── Scalar operator-method sugar (IDE accommodation) ────────────────
//
// The shader frontend recognises `a.mul(b)` / `.add` / `.sub` / `.div`
// / `.neg()` on ANY operand (see frontend/translate.ts) and emits the
// corresponding WGSL operator — including on scalars. But the IDE /
// tsserver only sees `number`, which has no such methods, so it
// red-squiggles otherwise-valid shader code (`f32` is `number & brand`).
//
// Declaring the methods on the global `Number` interface silences that.
// It must go on `Number` (not the `f32` brand): adding *required*
// methods to the brand would break `number → f32` assignability, so
// `const t: f32 = 0.5` and `x.mul(0.5)` (numeric-literal arg) would stop
// type-checking. On `Number`, plain literals keep working and the method
// is still callable.
//
// TYPE-ONLY — these do NOT exist on `Number.prototype` at runtime. Shader
// bodies compile to WGSL and never run as JS, so it's safe there; but
// calling e.g. `(5).mul(2)` in ordinary CPU code WILL throw. Treat the
// method style as shader-only. (Stopgap; the long-term ergonomic answer
// is operators via boperators or free-function math.)
declare global {
  interface Number {
    /** Shader-only: emits `a * b`. Not a real runtime method. */
    mul(o: number): number;
    /** Shader-only: emits `a + b`. Not a real runtime method. */
    add(o: number): number;
    /** Shader-only: emits `a - b`. Not a real runtime method. */
    sub(o: number): number;
    /** Shader-only: emits `a / b`. Not a real runtime method. */
    div(o: number): number;
    /** Shader-only: emits `-a`. Not a real runtime method. */
    neg(): number;
  }
}
