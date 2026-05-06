// Standard-uniform namespace — mirrors Aardvark's `UniformScope`
// extension on `FShade.Core` (FShade.fs:27-77).
//
// Aardvark gives F# code a `uniform.ModelTrafo`, `uniform.LightLocation`
// etc. lookup that resolves at IR-build time to a `ReadInput("Uniform",
// "ModelTrafo")` and at runtime against whatever the rendering layer
// auto-injects (Trafo3d composition, viewport size, headlight default).
// We mirror the same shape in TypeScript.
//
// Usage in inline-marker shaders:
//
//   import { uniform } from "@aardworx/wombat.shader/uniforms";
//
//   const trafo = vertex((v: { Positions: V3f }) => ({
//     gl_Position: uniform.ViewProjTrafo.mul(uniform.ModelTrafo.mul(
//       new V4f(v.Positions.x, v.Positions.y, v.Positions.z, 1.0))),
//   }));
//
// The plugin recognises the `uniform.X` access path via the
// existing ambient-namespace classifier — `uniform`'s declaration
// is `declare const`, ambient, with an object-typed shape, so the
// plugin sees a uniform-namespace and lowers `uniform.ModelTrafo`
// to `ReadInput("Uniform", "ModelTrafo", M44f)` directly. No extra
// runtime, no extra wiring.
//
// The set covers exactly what the wombat.dom scene-graph
// auto-injects (`autoInjectedUniforms` in
// `wombat.dom/src/scene/compile.ts`). Consumers add domain-specific
// uniforms via TypeScript module augmentation:
//
//   declare module "@aardworx/wombat.shader/uniforms" {
//     interface UniformScope {
//       readonly MyAppTransform: M44f;
//       readonly Time: number;
//     }
//   }

import type { V3f, V4f, M44f } from "@aardworx/wombat.base";

/**
 * The standard uniform namespace. Properties match the names the
 * runtime auto-injects (or that the user explicitly binds via
 * `<Sg Uniform={{...}}>...</Sg>`). Augment via TS module merging
 * to add app-specific uniforms.
 */
export interface UniformScope {
  // ─── core trafos (CPU-side `Trafo3d`s, uploaded as `M44f`) ──────
  readonly ModelTrafo: M44f;
  readonly ViewTrafo: M44f;
  readonly ProjTrafo: M44f;
  readonly ModelViewTrafo: M44f;
  readonly ViewProjTrafo: M44f;
  readonly ModelViewProjTrafo: M44f;

  // ─── inverses (each one auto-derived from its forward) ──────────
  readonly ModelTrafoInv: M44f;
  readonly ViewTrafoInv: M44f;
  readonly ProjTrafoInv: M44f;
  readonly ModelViewTrafoInv: M44f;
  readonly ViewProjTrafoInv: M44f;
  readonly ModelViewProjTrafoInv: M44f;

  // ─── derived from the trafos ────────────────────────────────────
  /**
   * `ModelTrafo.Backward.Transposed`, padded to M44f. Shaders use
   * the upper-3×3 (or pad V3f normals to (n, 0)) for normal
   * transforms when the model trafo has non-uniform scale.
   */
  readonly NormalMatrix: M44f;
  /** Camera position in world space — `ViewTrafoInv` applied to the origin. */
  readonly CameraLocation: V3f;
  /**
   * Light position in world space. Defaults to `CameraLocation`
   * (Aardvark's "headlight" convention) unless the user explicitly
   * binds it via `<Sg Uniform={{ LightLocation: cval(...) }}>...</Sg>`.
   */
  readonly LightLocation: V3f;

  // ─── pipeline state ─────────────────────────────────────────────
  /** `(width, height)` of the current viewport in pixels. */
  readonly ViewportSize: V4f;
}

/**
 * The `uniform` namespace. Inline-marker shader bodies reference
 * this via `uniform.<name>`; the plugin lowers each access to
 * `ReadInput("Uniform", <name>, <type>)` at IR-build time and the
 * rendering backend matches by name to whatever the scene-graph
 * or user explicitly bound.
 *
 * Runtime stub: a Proxy that throws on access. Valid usage hits
 * inside `vertex(...)` / `fragment(...)` / `compute(...)` markers,
 * which the plugin transforms at build time so the stub never
 * runs. Invalid usage (touching `uniform.X` outside a marker)
 * gets a clear diagnostic instead of `undefined`. The TS type
 * stays `UniformScope` so editor tooling sees the same shape
 * shader bodies see.
 */
export const uniform: UniformScope = new Proxy({} as UniformScope, {
  get(_target, prop) {
    throw new Error(
      `wombat.shader: \`uniform.${String(prop)}\` was accessed outside a ` +
      `vertex/fragment/compute marker. Wrap the access in `+
      `\`vertex(...)\` / \`fragment(...)\` / \`compute(...)\` and run the ` +
      `wombat.shader-vite plugin so the access is rewritten at build time.`,
    );
  },
});
