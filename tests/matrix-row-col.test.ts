// Matrix `m.R0` / `m.C0` accessors + the row/col-major flip in
// the WGSL/GLSL emit path.
//
// Background: wombat.base stores matrices row-major. The shader
// pipeline uploads the bytes verbatim and the GPU interprets them
// column-major (the "FShade trick" implemented in
// `reverseMatrixOps`). Net effect on element extraction:
//
//   - CPU's `m.row(r)` lines up bytewise with GPU's `m[r]`
//     (the c-th column read returns CPU row c).
//   - CPU's `m.col(c)` doesn't have a single GPU column; you
//     have to pick the c-th element from each GPU column
//     (= each CPU row): `vecN(m[0][c], m[1][c], …)`.
//
// `reverseMatrixOps` swaps `MatrixRow` ↔ `MatrixCol` so the
// default emit path yields the cheap single-column read for
// CPU-row access, and the per-row pick for CPU-column access.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import { Mat, Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";

const TM44f: Type = Mat(Tf32, 4, 4);
const Tvec4f: Type = Vec(Tf32, 4);

describe("frontend: m.R0 / m.C0 lower to MatrixRow / MatrixCol IR", () => {
  it("WGSL: `m.R0` after row/col flip emits the cheap `M[0]` form", () => {
    const source = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: M.R0 };
      }
    `;
    const r = compileShaderSource(source, [{
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }], { target: "wgsl",
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
    const src = r.stages[0]!.source;
    // After reverseMatrixOps: MatrixRow(M, 0) → MatrixCol(M, 0) →
    // emits `M[0]`. The expensive per-row pick form is avoided.
    expect(src).toMatch(/out\.gl_Position = M\[0(i|u)?\];?/);
  });

  it("WGSL: `m.C0` after row/col flip emits the per-row pick", () => {
    const source = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: M.C0 };
      }
    `;
    const r = compileShaderSource(source, [{
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }], { target: "wgsl",
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
    const src = r.stages[0]!.source;
    // CPU's column 0 → vec4(M[0][0], M[1][0], M[2][0], M[3][0]) on GPU.
    expect(src).toMatch(/M\[0i?\]\[0i?\]/);
    expect(src).toMatch(/M\[1i?\]\[0i?\]/);
    expect(src).toMatch(/M\[2i?\]\[0i?\]/);
    expect(src).toMatch(/M\[3i?\]\[0i?\]/);
  });

  it("with `skipMatrixReversal: true`, no flip happens — `R0` emits the per-row pick", () => {
    const source = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: M.R0 };
      }
    `;
    const r = compileShaderSource(source, [{
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }], { target: "wgsl", skipMatrixReversal: true,
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
    const src = r.stages[0]!.source;
    // No reversal: MatrixRow stays MatrixRow → emits the per-row
    // pick (column-major-aware).
    expect(src).toMatch(/M\[0i?\]\[0i?\]/);
    expect(src).toMatch(/M\[3i?\]\[0i?\]/);
  });
});
