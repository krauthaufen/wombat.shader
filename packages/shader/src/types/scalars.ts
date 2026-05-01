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
