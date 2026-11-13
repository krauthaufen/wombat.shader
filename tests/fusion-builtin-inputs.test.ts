// Same-stage fusion with builtin inputs: composed stages may each declare
// the same builtin under their own name (an F# fragment taking the vertex
// record declares `position`, the pick chain reads `fragCoord`). There is
// one state field per builtin, and the wrapper must fill THAT field.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";

const Tvec4f: Type = Vec(Tf32, 4);
const Tf: Type = Tf32;

const vs = {
  name: "vsMain", stage: "vertex" as const,
  inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location" as const, value: 0 }] }],
  outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin" as const, value: "position" as const }] }],
};
const position = { name: "position", type: Tvec4f, semantic: "position", decorations: [{ kind: "Builtin" as const, value: "position" as const }] };
const color = { name: "outColor", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 0 }] };
const depth = { name: "Depth", type: Tf, semantic: "Depth", decorations: [{ kind: "Builtin" as const, value: "frag_depth" as const }] };

const source = (aReads: boolean) => `
  function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
    return { gl_Position: input.Positions };
  }
  function fsA(input: { position: V4f }): { outColor: V4f } {
    return { outColor: new V4f(${aReads ? "input.position.x" : "1.0"}, 1.0, 1.0, 1.0) };
  }
  function fsB(input: {}, b: FragmentBuiltinIn): { Depth: f32 } {
    return { Depth: b.fragCoord.z };
  }
`;

const fragment = (aReads: boolean): string => {
  const r = compileShaderSource(source(aReads), [
    vs,
    { name: "fsA", stage: "fragment", inputs: [position], outputs: [color] },
    { name: "fsB", stage: "fragment", inputs: [], outputs: [depth] },
  ], { target: "wgsl" });
  return r.stages.find((s) => s.stage === "fragment")!.source;
};

/** The state field the depth is read from, and whether the wrapper fills it. */
const check = (fs: string): void => {
  expect(fs.match(/@builtin\(position\)/g)?.length).toBe(1);
  const field = /out\.Depth = out\.(\w+)\.z/.exec(fs)?.[1];
  expect(field).toBeDefined();
  expect(fs).toMatch(new RegExp(`s\\.${field} = position;`));
};

describe("fusion: builtin inputs are one state field per builtin", () => {
  it("a later stage reads the fragment position an earlier stage only declares", () => {
    check(fragment(false));
  });

  it("and the one both stages read", () => {
    const fs = fragment(true);
    check(fs);
    // filled once
    expect(fs.match(/= position;/g)?.length).toBe(1);
  });
});
