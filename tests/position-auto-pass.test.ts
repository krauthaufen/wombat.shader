// FShade-style Position auto-pass.
//
// The GPU's `@builtin(position)` on FS input is screen-space
// pixel coords (gl_FragCoord), NOT the rasteriser-interpolated
// clip-space the user expects from a "Positions" semantic. When
// a fragment effect declares a `Positions`-semantic input but no
// upstream effect explicitly emitted that varying, the cross-
// stage linker synthesises an extra Location-decorated VS output
// whose value is the same expression the VS already wrote to
// `@builtin(position)`. The rasteriser then interpolates it as a
// regular vec4 varying and the FS gets clip-space.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import { Tf32, Vec, Mat, type Type } from "@aardworx/wombat.shader/ir";

const TM44f: Type = Mat(Tf32, 4, 4);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

describe("linkCrossStage: Position auto-pass", () => {
  it("synthesises a varying for FS Positions input when only @builtin(position) was written", () => {
    const source = `
      function vsMain(input: { Positions: V3f }): { gl_Position: V4f } {
        return { gl_Position: M.mul(new V4f(input.Positions.x, input.Positions.y, input.Positions.z, 1.0)) };
      }
      function fsMain(input: { Positions: V4f }): { outColor: V4f } {
        return { outColor: input.Positions };
      }
    `;
    const r = compileShaderSource(source, [
      {
        name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec3f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], { target: "wgsl",
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });

    const vs = r.stages.find((s) => s.stage === "vertex")!.source;
    const fs = r.stages.find((s) => s.stage === "fragment")!.source;

    // VS emits BOTH `@builtin(position)` AND a Location-decorated
    // `Positions` varying. The two writes share the same RHS
    // expression (or DCE/CSE shares it).
    expect(vs).toContain("@builtin(position)");
    expect(vs).toMatch(/@location\(\d+\)\s+Positions/);

    // FS reads `Positions` from the input struct (a regular varying),
    // not from `@builtin(position)`.
    expect(fs).toMatch(/@location\(\d+\)\s+Positions/);
    // The FS struct does NOT declare `@builtin(position)` for the
    // `Positions` field — that would route screen-space coords.
    const fsStruct = fs.match(/struct[^{]*Input\s*\{[\s\S]*?\}/)?.[0] ?? "";
    expect(fsStruct).not.toMatch(/@builtin\(position\)\s+Positions/);
  });

  it("no auto-pass when the VS already provides Positions as a varying", () => {
    // VS emits Positions as a regular Location output AND
    // gl_Position. linkCrossStage uses the existing varying — no
    // additional synthesis needed.
    const source = `
      function vsMain(input: { Positions: V3f }): { gl_Position: V4f; Positions: V4f } {
        const wp = M.mul(new V4f(input.Positions.x, input.Positions.y, input.Positions.z, 1.0));
        return { gl_Position: wp, Positions: wp };
      }
      function fsMain(input: { Positions: V4f }): { outColor: V4f } {
        return { outColor: input.Positions };
      }
    `;
    const r = compileShaderSource(source, [
      {
        name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec3f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], { target: "wgsl",
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });

    const vs = r.stages.find((s) => s.stage === "vertex")!.source;
    // Exactly ONE `Positions` Location output in the OutputStruct
    // (no duplicate synthesised on top of the user's explicit one).
    // Scope the regex to the output struct only — the input struct
    // has a `Positions: vec3` location too.
    const outStruct = vs.match(/struct[^{]*Output\s*\{[\s\S]*?\}/)?.[0] ?? "";
    const outPositions = outStruct.match(/@location\(\d+\)\s+Positions/g) ?? [];
    expect(outPositions).toHaveLength(1);
  });
});
