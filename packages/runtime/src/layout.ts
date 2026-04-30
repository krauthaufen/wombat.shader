// Compute byte offsets / sizes / alignments for IR `Type`s under each
// backend's uniform-block layout rules.
//
//   GLSL ES 3.00 — std140 (the only block layout WebGL2 supports without
//   extensions). Specifically:
//     - scalar (i32/u32/f32/bool) — size 4, align 4
//     - vec2  — size 8, align 8
//     - vec3  — size 12, align 16   (!)  the size doesn't include the pad
//     - vec4  — size 16, align 16
//     - matNxM (column-major) — array of N vec4s; 16N bytes, align 16
//     - array<T, n>  — each element padded up to align-of-vec4 (16 bytes)
//     - struct — each member at the next aligned offset, total size
//                rounded to the largest member's alignment.
//
//   WGSL — uniform layout matches std140 *very* closely but vec3 is
//   typically 12-byte size + 16-byte align (same), arrays of scalars
//   don't get padded to vec4 (unlike std140), and the trailing padding
//   to "biggest member" is the same. We use the WGSL spec's formulas
//   (https://www.w3.org/TR/WGSL/#alignment-and-size).
//
// References:
//   - https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf §2.12.6.4
//   - https://www.w3.org/TR/WGSL/#address-space-layout-constraints

import type { StructField, Type } from "@aardworx/wombat.shader-ir";

export type LayoutTarget = "glsl-std140" | "wgsl-uniform" | "wgsl-storage";

export interface FieldLayout {
  readonly name: string;
  readonly type: Type;
  /** Byte offset from the start of the enclosing struct/buffer. */
  readonly offset: number;
  /** Resolved layout for this field. */
  readonly layout: LayoutInfo;
}

export interface LayoutInfo {
  /** Logical size of the value (without trailing struct alignment padding). */
  readonly size: number;
  /** Required alignment of this value's start address. */
  readonly align: number;
  /** For struct types only — resolved field layouts. */
  readonly fields?: readonly FieldLayout[] | undefined;
  /** For array types — element stride (already rounded up). */
  readonly stride?: number | undefined;
  /** For array types — element layout (without stride padding). */
  readonly element?: LayoutInfo | undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────────────

export function computeLayout(type: Type, target: LayoutTarget): LayoutInfo {
  switch (target) {
    case "glsl-std140":   return std140(type);
    case "wgsl-uniform":  return wgsl(type, /* uniform */ true);
    case "wgsl-storage":  return wgsl(type, /* uniform */ false);
  }
}

// ─────────────────────────────────────────────────────────────────────
// std140 (GLSL ES 3.00)
// ─────────────────────────────────────────────────────────────────────

function std140(t: Type): LayoutInfo {
  switch (t.kind) {
    case "Bool":
    case "Int":
    case "Float":
      return { size: 4, align: 4 };
    case "Vector": {
      const elem = std140(t.element);
      const size = elem.size * t.dim;
      const align = t.dim === 2 ? 2 * elem.align : 4 * elem.align;
      return { size, align };
    }
    case "Matrix": {
      // std140 stores each column padded to vec4. Stride is 16; total
      // size is `cols * 16`. Align matches a vec4.
      const stride = 16;
      return {
        size: stride * t.cols,
        align: 16,
        stride,
        element: { size: t.rows * 4, align: 16 },
      };
    }
    case "Array": {
      const elem = std140(t.element);
      // std140 rounds the element size up to a multiple of vec4 alignment (16).
      const stride = roundUp(Math.max(elem.size, 16), 16);
      const length = t.length === "runtime" ? 0 : t.length;
      return { size: stride * length, align: 16, stride, element: elem };
    }
    case "Struct":
      return std140Struct(t.fields);
    case "Sampler":
    case "Texture":
    case "AtomicI32":
    case "AtomicU32":
    case "Intrinsic":
    case "Void":
      // Opaque types don't live inside std140 blocks.
      return { size: 0, align: 1 };
  }
}

function std140Struct(fields: readonly StructField[]): LayoutInfo {
  const out: FieldLayout[] = [];
  let offset = 0;
  let maxAlign = 1;
  for (const f of fields) {
    const fl = std140(f.type);
    offset = roundUp(offset, fl.align);
    out.push({ name: f.name, type: f.type, offset, layout: fl });
    offset += fl.size;
    if (fl.align > maxAlign) maxAlign = fl.align;
  }
  // Struct size is rounded up to its largest member's alignment.
  const size = roundUp(offset, maxAlign);
  return { size, align: maxAlign, fields: out };
}

// ─────────────────────────────────────────────────────────────────────
// WGSL layout
// ─────────────────────────────────────────────────────────────────────

function wgsl(t: Type, isUniform: boolean): LayoutInfo {
  switch (t.kind) {
    case "Bool":
      // bool is host-shareable only via i32 for uniform blocks; we
      // surface size 4 / align 4 for symmetry with i32.
      return { size: 4, align: 4 };
    case "Int":
    case "Float":
      return { size: 4, align: 4 };
    case "Vector": {
      const elem = wgsl(t.element, isUniform);
      const size = elem.size * t.dim;
      const align = t.dim === 2 ? 2 * elem.align : 4 * elem.align;
      return { size, align };
    }
    case "Matrix": {
      // WGSL matCxR: array of C column vectors of length R, with the
      // column alignment matching vec(R). For matCx3 this still pads
      // each column to vec4 alignment (16 bytes) like std140.
      const rows = t.rows;
      const elementSize = rows * 4;
      const elementAlign = rows === 2 ? 8 : 16;
      const stride = roundUp(elementSize, elementAlign);
      return {
        size: stride * t.cols,
        align: elementAlign,
        stride,
        element: { size: elementSize, align: elementAlign },
      };
    }
    case "Array": {
      const elem = wgsl(t.element, isUniform);
      // Uniform arrays: stride padded to multiple of 16 (same as std140).
      // Storage arrays: stride is the natural element size aligned to elem.align.
      const stride = isUniform
        ? roundUp(Math.max(elem.size, 16), 16)
        : roundUp(elem.size, elem.align);
      const length = t.length === "runtime" ? 0 : t.length;
      return { size: stride * length, align: elem.align, stride, element: elem };
    }
    case "Struct":
      return wgslStruct(t.fields, isUniform);
    case "Sampler":
    case "Texture":
    case "AtomicI32":
    case "AtomicU32":
    case "Intrinsic":
    case "Void":
      return { size: 0, align: 1 };
  }
}

function wgslStruct(fields: readonly StructField[], isUniform: boolean): LayoutInfo {
  const out: FieldLayout[] = [];
  let offset = 0;
  let maxAlign = 1;
  for (const f of fields) {
    const fl = wgsl(f.type, isUniform);
    offset = roundUp(offset, fl.align);
    out.push({ name: f.name, type: f.type, offset, layout: fl });
    offset += fl.size;
    if (fl.align > maxAlign) maxAlign = fl.align;
  }
  const size = roundUp(offset, maxAlign);
  return { size, align: maxAlign, fields: out };
}

// ─────────────────────────────────────────────────────────────────────

function roundUp(value: number, multiple: number): number {
  return Math.ceil(value / multiple) * multiple;
}
