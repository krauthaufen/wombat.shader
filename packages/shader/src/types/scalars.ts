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
// This declaration is the TYPE side. The matching RUNTIME side lives in
// `@aardworx/wombat.base` (src/numberMethods.ts), which installs these on
// `Number.prototype` as a side effect of importing the package — so the
// methods also WORK in ordinary CPU code wherever wombat.base is loaded
// (which is everywhere in this stack: shader → base). In shader bodies the
// frontend lowers them to WGSL operators, so the runtime impl is never
// even called there. Net: one method-based scalar vocabulary, no plugin.
declare global {
  interface Number {
    /** Scalar `a * b` (frontend lowers to `*`; runtime impl in wombat.base). */
    mul(o: number): number;
    /** Scalar `a + b` (frontend lowers to `+`; runtime impl in wombat.base). */
    add(o: number): number;
    /** Scalar `a - b` (frontend lowers to `-`; runtime impl in wombat.base). */
    sub(o: number): number;
    /** Scalar `a / b` (frontend lowers to `/`; runtime impl in wombat.base). */
    div(o: number): number;
    /** Scalar `-a` (frontend lowers to unary `-`; runtime impl in wombat.base). */
    neg(): number;
  }
}
