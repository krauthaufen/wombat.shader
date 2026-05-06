// Semantic type brands.
//
// TypeScript has no `[<Position>]`-style attributes that can attach
// to a parameter or interface property the way F# / C# / Rust do. The
// inline-shader plugin needs *some* way to learn that a field of an
// effect's input/output record carries the canonical
// `DefaultSemantic` meaning ("Positions", "Normals", "Colors", …)
// without forcing the user to spell that exact name as the field key:
// inputs and outputs of the same effect often want different names
// for the same semantic (a vertex attribute `Positions` is V3f
// object-space; the homonymous output is V4f clip-space).
//
// We use TypeScript's intersection-with-phantom-symbol pattern. The
// `__semantic` symbol is `unique`, never assigned, and never
// referenced at runtime — it exists purely to thread a string-literal
// type through structural unification. The plugin's type-resolver
// walks the type-checker's view of each entry-parameter field; if it
// sees an intersection that includes `{ readonly [__semantic]: "X" }`,
// it treats `X` as the field's `DefaultSemantic` name regardless of
// the field's user-chosen identifier.
//
// Example use, with the plugin honoring the brand:
//
//     import type { Position, Normal, Color } from "@aardworx/wombat.shader/types";
//
//     const trafo = vertex((v: {
//       pos:  Position<V3f>;     // semantic: "Positions"; field: "pos"
//       norm: Normal;            // semantic: "Normals";   default V3f
//       col:  Color;             // semantic: "Colors";    default V4f
//     }): {
//       clipPos: Position<V4f>;  // → @builtin(position) on the entry's
//                                //   output struct, regardless of name
//       norm:    Normal;
//       col:     Color;
//     } => ({
//       clipPos: ViewProjTrafo.mul(new V4f(v.pos.x, v.pos.y, v.pos.z, 1)),
//       norm:    v.norm,
//       col:     v.col,
//     }));
//
// Everything outside `vertex(...)` (e.g. the runtime-side aval
// closure that supplies the V4f) sees plain V*f because the brand is
// erased at compile time.

// V*f / V*ui imports for the alias defaults below — type-only.
import type { V2f, V3f, V4f, V3ui } from "@aardworx/wombat.base";

/**
 * Phantom field on `Semantic<T, N>`. `unique symbol` plus the
 * `declare const` (no value) means TS treats it as a brand that's
 * structurally unique, while runtime sees nothing.
 */
declare const __semantic: unique symbol;

/**
 * Carry a `DefaultSemantic`-style name `N` on top of any value
 * type `T`. Type-only — disappears at emit time.
 *
 * The plugin recognises this brand by walking intersection types
 * looking for a `{ readonly [__semantic]: N }` member with `N` a
 * string-literal type, then treats `N` as the field's semantic
 * name. The underlying `T` becomes the field's IR type.
 */
export type Semantic<T, N extends string> = T & {
  readonly [__semantic]: N;
};

// ─────────────────────────────────────────────────────────────────────
// DefaultSemantic aliases — one per Aardvark `DefaultSemantic` entry.
//
// All take a generic `T` so the user can override the type. Defaults
// match the most common stage-of-use:
//   - Position<T = V4f>     — V3f for object-space vertex attributes,
//                              V4f for clip-space outputs (default).
//   - Normal<T = V3f>       — V3f world/view-space normal.
//   - Color<T = V4f>        — V4f RGBA.
//   - Texcoord<T = V2f>     — V2f UV.
//   - Tangent<T = V3f>      — V3f tangent.
//   - BiNormal<T = V3f>     — V3f bitangent.
// ─────────────────────────────────────────────────────────────────────

export type Position<T = V4f> = Semantic<T, "Positions">;
export type WorldPosition<T = V4f> = Semantic<T, "WorldPositions">;
export type ViewPosition<T = V3f> = Semantic<T, "ViewPosition">;
export type Normal<T = V3f> = Semantic<T, "Normals">;
export type ViewNormal<T = V3f> = Semantic<T, "ViewNormal">;
export type Tangent<T = V3f> = Semantic<T, "Tangents">;
export type BiNormal<T = V3f> = Semantic<T, "BiNormals">;
export type DiffuseColorUTangent<T = V3f> = Semantic<T, "DiffuseColorUTangents">;
export type DiffuseColorVTangent<T = V3f> = Semantic<T, "DiffuseColorVTangents">;
export type Color<T = V4f> = Semantic<T, "Colors">;
export type Texcoord<T = V2f> = Semantic<T, "DiffuseColorCoordinates">;
export type Pick<T = number> = Semantic<T, "Pick">;

// ─────────────────────────────────────────────────────────────────────
// Builtin<T, K> — GPU-provided / GPU-consumed built-in slots.
//
// Distinct brand from `Semantic` so the plugin can tell a
// user-facing semantic (which lowers to a vertex attribute or
// inter-stage varying location) from a hardware builtin (which
// lowers to `@builtin(K)` in WGSL or the equivalent
// `gl_*` global in GLSL ES). Like `Semantic`, the brand is type-
// only — runtime sees plain V*f / number / boolean.
//
// Aliases below cover every WGSL builtin slot. Use them on
// entry-parameter records:
//
//   const fs = fragment((
//     v: { col: Color },
//     b: { fc: FragCoord; ff: FrontFacing },
//   ) => ({ outColor: v.col, depth: b.fc.z as Depth }));
//
// The plugin extracts the brand `K` from the field's declared
// type, applies `@builtin(K)` to the corresponding entry parameter
// at IR-build time, and uses `T` as the underlying value type.
// ─────────────────────────────────────────────────────────────────────

declare const __builtin: unique symbol;

export type Builtin<T, K extends string> = T & {
  readonly [__builtin]: K;
};

// Vertex stage — inputs and outputs.
/** Vertex-stage clip-space output (`@builtin(position)`). */
export type ClipPosition<T = V4f> = Builtin<T, "position">;
/** Per-vertex index (`@builtin(vertex_index)`, u32). */
export type VertexIndex<T = number> = Builtin<T, "vertex_index">;
/** Per-instance index (`@builtin(instance_index)`, u32). */
export type InstanceIndex<T = number> = Builtin<T, "instance_index">;

// Fragment stage — inputs and outputs.
/** Fragment-stage screen-space coord (`@builtin(position)` on FS input, vec4f). */
export type FragCoord<T = V4f> = Builtin<T, "position">;
/** True for front-facing primitives (`@builtin(front_facing)`, bool). */
export type FrontFacing<T = boolean> = Builtin<T, "front_facing">;
/** Sample index in MSAA shading (`@builtin(sample_index)`, u32). */
export type SampleIndex<T = number> = Builtin<T, "sample_index">;
/** Sample mask (in / out) — `@builtin(sample_mask)`, u32. */
export type SampleMask<T = number> = Builtin<T, "sample_mask">;
/** Custom depth output (`@builtin(frag_depth)`, f32). */
export type FragDepth<T = number> = Builtin<T, "frag_depth">;
/**
 * Conventional `Depth` output — same `frag_depth` slot as
 * `FragDepth`, kept under the `DefaultSemantic` name so user code
 * that already thinks in those terms can stay consistent. The
 * plugin treats both as identical at IR time.
 */
export type Depth<T = number> = Builtin<T, "frag_depth">;

// Compute stage builtins. WGSL spec specifies these as `vec3<u32>`,
// so the defaults use `V3ui` (unsigned 3-component int) — not V3f.
// User can override `T` if they want a different concrete type
// (e.g. plain number tuples for testing harnesses).
/** Local thread coords inside a workgroup (`@builtin(local_invocation_id)`, vec3<u32>). */
export type LocalInvocationId<T = V3ui> = Builtin<T, "local_invocation_id">;
/** Linear local index (`@builtin(local_invocation_index)`, u32). */
export type LocalInvocationIndex<T = number> = Builtin<T, "local_invocation_index">;
/** Global thread coords (`@builtin(global_invocation_id)`, vec3<u32>). */
export type GlobalInvocationId<T = V3ui> = Builtin<T, "global_invocation_id">;
/** Workgroup coords (`@builtin(workgroup_id)`, vec3<u32>). */
export type WorkgroupId<T = V3ui> = Builtin<T, "workgroup_id">;
/** Number of workgroups in the dispatch (`@builtin(num_workgroups)`, vec3<u32>). */
export type NumWorkgroups<T = V3ui> = Builtin<T, "num_workgroups">;

// ─────────────────────────────────────────────────────────────────────
// Stage-direction registry
//
// For each `@builtin(K)` slot, the `(stage, direction)` pairs where
// it's actually allowed to appear. The plugin uses this at IR-build
// time to error out cleanly when a user puts e.g. `FragCoord` on a
// vertex input or `VertexIndex` on a fragment input — those would
// otherwise silently survive into the WGSL emit and get rejected by
// the GPU's pipeline-creation path with an obscure validation
// error.
// ─────────────────────────────────────────────────────────────────────

export type BuiltinStage = "vertex" | "fragment" | "compute";
export type BuiltinDirection = "in" | "out";

export interface BuiltinSlot {
  readonly stage: BuiltinStage;
  readonly direction: BuiltinDirection;
}

/**
 * Map of builtin name (matches WGSL `@builtin(...)` argument) → the
 * legal `(stage, direction)` slots. A builtin with multiple legal
 * slots (e.g. `position` is a vertex output AND a fragment input,
 * `sample_mask` is fragment in AND out) lists each. Plugin and
 * frontend validate against this; consumer code can also import the
 * map to introspect.
 */
export const BUILTIN_SLOTS: Readonly<Record<string, readonly BuiltinSlot[]>> = {
  // Position: vertex stage emits clip-space; fragment stage receives
  // the rasteriser-interpolated screen-space coord under the same
  // builtin slot.
  position: [
    { stage: "vertex", direction: "out" },
    { stage: "fragment", direction: "in" },
  ],
  vertex_index: [{ stage: "vertex", direction: "in" }],
  instance_index: [{ stage: "vertex", direction: "in" }],
  front_facing: [{ stage: "fragment", direction: "in" }],
  sample_index: [{ stage: "fragment", direction: "in" }],
  sample_mask: [
    { stage: "fragment", direction: "in" },
    { stage: "fragment", direction: "out" },
  ],
  frag_depth: [{ stage: "fragment", direction: "out" }],
  local_invocation_id: [{ stage: "compute", direction: "in" }],
  local_invocation_index: [{ stage: "compute", direction: "in" }],
  global_invocation_id: [{ stage: "compute", direction: "in" }],
  workgroup_id: [{ stage: "compute", direction: "in" }],
  num_workgroups: [{ stage: "compute", direction: "in" }],
};

/**
 * True iff `builtin` is allowed on the given `(stage, direction)`
 * slot. Returns `true` for unknown builtin names (so the plugin
 * doesn't reject experimental extensions); narrow to known names
 * via `BUILTIN_SLOTS` if you want strict checks.
 */
export function isBuiltinAllowed(
  builtin: string,
  stage: BuiltinStage,
  direction: BuiltinDirection,
): boolean {
  const slots = BUILTIN_SLOTS[builtin];
  if (slots === undefined) return true;
  return slots.some((s) => s.stage === stage && s.direction === direction);
}
