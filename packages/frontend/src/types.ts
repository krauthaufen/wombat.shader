// Mapping from shader-types TypeScript names to IR `Type`.
//
// The frontend recognises types by name only (matching the shipped
// declarations in `@aardworx/wombat.shader-types`). A real frontend
// would use the TS type checker to resolve the symbol; the lite path
// works for hand-written shader source that imports the right names.

import type { Type } from "@aardworx/wombat.shader-ir";

const Tvoid: Type = { kind: "Void" };
const Tbool: Type = { kind: "Bool" };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };
const Tu32: Type = { kind: "Int", signed: false, width: 32 };
const Tf32: Type = { kind: "Float", width: 32 };

const Tvec = (e: Type, dim: 2 | 3 | 4): Type => ({ kind: "Vector", element: e, dim });
const Tmat = (e: Type, rows: 2 | 3 | 4, cols: 2 | 3 | 4): Type => ({
  kind: "Matrix", element: e, rows, cols,
});

const TYPE_TABLE: Record<string, Type> = {
  // primitives
  void: Tvoid,
  bool: Tbool,
  number: Tf32,        // default JS number → f32 in shader code
  i32: Ti32,
  u32: Tu32,
  f32: Tf32,

  // wombat.base / wombat.shader-types names
  V2b: Tvec(Tbool, 2), V3b: Tvec(Tbool, 3), V4b: Tvec(Tbool, 4),
  V2i: Tvec(Ti32, 2),  V3i: Tvec(Ti32, 3),  V4i: Tvec(Ti32, 4),
  // wombat.base uses `Vxui`; older shipped shims used `Vxu`. Both
  // resolve to the same IR type — we keep both names for
  // back-compat. New code should prefer `Vxui` (the wombat.base
  // canonical form).
  V2u: Tvec(Tu32, 2),  V3u: Tvec(Tu32, 3),  V4u: Tvec(Tu32, 4),
  V2ui: Tvec(Tu32, 2), V3ui: Tvec(Tu32, 3), V4ui: Tvec(Tu32, 4),
  V2f: Tvec(Tf32, 2),  V3f: Tvec(Tf32, 3),  V4f: Tvec(Tf32, 4),

  M22f: Tmat(Tf32, 2, 2), M33f: Tmat(Tf32, 3, 3), M44f: Tmat(Tf32, 4, 4),
  M23f: Tmat(Tf32, 2, 3), M24f: Tmat(Tf32, 2, 4),
  M32f: Tmat(Tf32, 3, 2), M34f: Tmat(Tf32, 3, 4),
  M42f: Tmat(Tf32, 4, 2), M43f: Tmat(Tf32, 4, 3),

  Sampler2D: { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false },
  Sampler3D: { kind: "Sampler", target: "3D", sampled: { kind: "Float" }, comparison: false },
  SamplerCube: { kind: "Sampler", target: "Cube", sampled: { kind: "Float" }, comparison: false },
  Sampler2DArray: { kind: "Sampler", target: "2DArray", sampled: { kind: "Float" }, comparison: false },
  SamplerCubeArray: { kind: "Sampler", target: "CubeArray", sampled: { kind: "Float" }, comparison: false },
  ISampler2D: { kind: "Sampler", target: "2D", sampled: { kind: "Int", signed: true }, comparison: false },
  USampler2D: { kind: "Sampler", target: "2D", sampled: { kind: "Int", signed: false }, comparison: false },
  Sampler2DShadow: { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: true },
  Sampler2DMS: { kind: "Sampler", target: "2DMS", sampled: { kind: "Float" }, comparison: false },
  ISampler2DMS: { kind: "Sampler", target: "2DMS", sampled: { kind: "Int", signed: true }, comparison: false },
  USampler2DMS: { kind: "Sampler", target: "2DMS", sampled: { kind: "Int", signed: false }, comparison: false },
};

/** Resolve a textual TS type name to an IR Type. Throws on unknown names. */
export function resolveTypeName(name: string): Type {
  const t = TYPE_TABLE[name];
  if (!t) throw new Error(`frontend: unknown shader type "${name}"`);
  return t;
}

/** Best-effort type lookup; returns undefined if the name isn't in the table. */
export function tryResolveTypeName(name: string): Type | undefined {
  return TYPE_TABLE[name];
}

/** Read the IR-level vector dimension from a name like `V3f`. */
export function vectorDimOf(name: string): 2 | 3 | 4 | undefined {
  const m = name.match(/^V([234])[buif]$/);
  if (!m) return undefined;
  return parseInt(m[1]!, 10) as 2 | 3 | 4;
}

/** Recognise vector / matrix construction calls (`vec3`, `mat4`, …). */
export function constructorTargetType(name: string): Type | undefined {
  const m = /^(i?vec|uvec|vec)([234])$/.exec(name);
  if (m) {
    const prefix = m[1];
    const dim = parseInt(m[2]!, 10) as 2 | 3 | 4;
    const element = prefix === "ivec" ? Ti32 : prefix === "uvec" ? Tu32 : Tf32;
    return Tvec(element, dim);
  }
  const mm = /^(i?mat|mat)([234])(?:x([234]))?$/.exec(name);
  if (mm) {
    const rows = parseInt(mm[2]!, 10) as 2 | 3 | 4;
    const cols = (mm[3] ? parseInt(mm[3], 10) : rows) as 2 | 3 | 4;
    return Tmat(Tf32, rows, cols);
  }
  return undefined;
}
