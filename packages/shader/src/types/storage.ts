// Storage-buffer and storage-texture brands.
//
// These are zero-runtime markers — the build plugin recognises
// `Storage<T, A>` and `StorageTexture*<F, A>` annotations on
// declarations and emits the appropriate `StorageBuffer` /
// `Sampler<StorageTexture>` ValueDef in the IR.
//
// Access mode `A` defaults to `"read_write"`; the
// `inferStorageAccess` pass narrows it to `"read"` when the body
// only reads the resource.

declare const __storageBrand: unique symbol;

/**
 * Storage buffer of element shape `T`. `T` should be an
 * IR-mappable type — typically a runtime-sized array
 * (`i32[]`, `V4f[]`, etc.) or a fixed-size shape.
 *
 * Usage:
 *
 *   declare const heights: Storage<f32[]>;
 *   declare const counters: Storage<u32[], "read_write">;
 *
 *   compute(b => { heights[i] = …; });
 */
export type Storage<T, A extends "read" | "read_write" = "read_write"> =
  T & { readonly [__storageBrand]?: ["buffer", A] };

/**
 * 2D storage texture. `F` is a WGSL format string
 * (`"rgba8unorm"`, `"r32float"`, …); `A` is access mode.
 *
 *   declare const out: StorageTexture2D<"rgba8unorm", "write">;
 *   compute(b => { textureStore(out, ivec2(b.globalInvocationId.xy), color); });
 */
export type StorageTexture2D<F extends string, A extends "read" | "write" | "read_write" = "write"> =
  { readonly [__storageBrand]?: ["texture-2d", F, A] };

export type StorageTexture3D<F extends string, A extends "read" | "write" | "read_write" = "write"> =
  { readonly [__storageBrand]?: ["texture-3d", F, A] };

export type StorageTexture2DArray<F extends string, A extends "read" | "write" | "read_write" = "write"> =
  { readonly [__storageBrand]?: ["texture-2d-array", F, A] };
