// Same-stage fusion with builtin outputs (FShade composition semantics):
// every spelling of `@builtin(position)` is one port, and a later vertex
// stage reading `Positions` reads the earlier stage's position write, not
// the vertex attribute. Same for `@builtin(frag_depth)` / `Depth`.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import { Tf32, Vec, Mat, type Type } from "@aardworx/wombat.shader/ir";

const TM44f: Type = Mat(Tf32, 4, 4);
const Tvec4f: Type = Vec(Tf32, 4);
const Tf: Type = Tf32;

const vsOf = (name: string, out: string) => ({
  name, stage: "vertex" as const,
  inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location" as const, value: 0 }] }],
  outputs: [{ name: out, type: Tvec4f, semantic: out === "gl_Position" ? "Position" : "Positions", decorations: [{ kind: "Builtin" as const, value: "position" as const }] }],
});
const fsWhite = {
  name: "fsMain", stage: "fragment" as const,
  inputs: [],
  outputs: [{ name: "outColor", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 0 }] }],
};

describe("fusion: builtin outputs are one port per builtin", () => {
  it("a stage reading Positions after a gl_Position write reads that write", () => {
    const source = `
      function vsA(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: M.mul(input.Positions) };
      }
      function vsB(input: { Positions: V4f }): { Positions: V4f } {
        return { Positions: new V4f(input.Positions.x, input.Positions.y, input.Positions.z * 0.25, input.Positions.w) };
      }
      function fsMain(input: {}): { outColor: V4f } {
        return { outColor: new V4f(1.0, 1.0, 1.0, 1.0) };
      }
    `;
    const r = compileShaderSource(source, [vsOf("vsA", "gl_Position"), vsOf("vsB", "Positions"), fsWhite], {
      target: "wgsl",
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }],
    });
    const vs = r.stages.find((s) => s.stage === "vertex")!;
    // one position output (two spellings used to fuse into two)
    expect(vs.source.match(/@builtin\(position\)/g)?.length).toBe(1);
    // one attribute
    expect(vs.bindings.inputs).toHaveLength(1);
    // vsB scales the transformed z: its helper reads the state's
    // gl_Position (written by vsA), never the Positions attribute
    const helperB = vs.source.split(/\bfn /).find((f) => f.includes("0.25"))!;
    expect(helperB).toMatch(/gl_Position/);
    expect(helperB).not.toMatch(/\.Positions\b/);
  });

  it("a later fragment stage reading Depth reads an earlier gl_FragDepth write", () => {
    const source = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: input.Positions };
      }
      function fsA(input: {}): { outColor: V4f; gl_FragDepth: f32 } {
        return { outColor: new V4f(1.0, 1.0, 1.0, 1.0), gl_FragDepth: 0.25 };
      }
      function fsB(input: { Depth: f32 }): { outColor: V4f } {
        return { outColor: new V4f(input.Depth, input.Depth, input.Depth, 1.0) };
      }
    `;
    const r = compileShaderSource(source, [
      vsOf("vsMain", "gl_Position"),
      { name: "fsA", stage: "fragment",
        inputs: [],
        outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location", value: 0 }] },
          { name: "gl_FragDepth", type: Tf, semantic: "Depth", decorations: [{ kind: "Builtin", value: "frag_depth" }] },
        ] },
      { name: "fsB", stage: "fragment",
        inputs: [{ name: "Depth", type: Tf, semantic: "Depth", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location", value: 0 }] }] },
    ], { target: "wgsl" });
    const fs = r.stages.find((s) => s.stage === "fragment")!;
    // Depth is not a varying: fsB reads fsA's write
    expect(fs.bindings.inputs).toHaveLength(0);
  });

  it("fragment stages writing gl_FragDepth and Depth fuse into one frag_depth output", () => {
    const source = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: input.Positions };
      }
      function fsA(input: {}): { outColor: V4f; gl_FragDepth: f32 } {
        return { outColor: new V4f(1.0, 1.0, 1.0, 1.0), gl_FragDepth: 0.25 };
      }
      function fsB(input: {}): { Depth: f32 } {
        return { Depth: 0.5 };
      }
    `;
    const r = compileShaderSource(source, [
      vsOf("vsMain", "gl_Position"),
      { name: "fsA", stage: "fragment",
        inputs: [],
        outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location", value: 0 }] },
          { name: "gl_FragDepth", type: Tf, semantic: "Depth", decorations: [{ kind: "Builtin", value: "frag_depth" }] },
        ] },
      { name: "fsB", stage: "fragment",
        inputs: [],
        outputs: [{ name: "Depth", type: Tf, semantic: "Depth", decorations: [{ kind: "Builtin", value: "frag_depth" }] }] },
    ], { target: "wgsl" });
    const fs = r.stages.find((s) => s.stage === "fragment")!;
    expect(fs.source.match(/@builtin\(frag_depth\)/g)?.length).toBe(1);
  });
});
