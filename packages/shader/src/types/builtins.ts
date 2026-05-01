// Stage-builtin variables. The frontend recognises these by identity
// and emits `LInput`/`ReadInput` with `scope: "Builtin"`.

import type { V2f, V4f } from "./vectors.js";

// ─── vertex stage builtins ────────────────────────────────────────────

export interface VertexBuiltinIn {
  /** GL_VertexID / @builtin(vertex_index). */
  readonly vertexIndex: number;
  /** Instance ID / @builtin(instance_index). */
  readonly instanceIndex: number;
}

export interface VertexBuiltinOut {
  /** Clip-space position. Required output. */
  position: V4f;
  /** Point size (only meaningful for `gl.POINTS` topology). */
  pointSize?: number;
}

// ─── fragment stage builtins ──────────────────────────────────────────

export interface FragmentBuiltinIn {
  /** Window-space pixel coordinates (xy, plus z=depth, w=1/clip.w). */
  readonly fragCoord: V4f;
  /** True if rendering the front face of the primitive. */
  readonly frontFacing: boolean;
  /** Point coordinates (only meaningful for points). */
  readonly pointCoord: V2f;
}

export interface FragmentBuiltinOut {
  /** Override the depth value written to the depth buffer. */
  fragDepth?: number;
}

// ─── compute stage builtins ───────────────────────────────────────────

import type { V3ui } from "./vectors.js";

export interface ComputeBuiltins {
  /** Global invocation in `[0, dispatchSize)`. */
  readonly globalInvocationId: V3ui;
  /** Local invocation in `[0, workgroupSize)`. */
  readonly localInvocationId: V3ui;
  /** Workgroup ID in `[0, dispatchSize / workgroupSize)`. */
  readonly workgroupId: V3ui;
  /** Number of workgroups in the dispatch. */
  readonly numWorkgroups: V3ui;
}
