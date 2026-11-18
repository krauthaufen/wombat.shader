// The F# (Fable) plugin decorates `[<Position>]` with the position
// BUILTIN on inputs as well as outputs (Translator.fs's
// `fshadeAttrLookup`), so a vertex shader that reads the position
// attribute arrives as `Positions` + `@builtin(position)` on the input
// side. A vertex input can never be `@builtin(position)` — that slot
// only exists on a fragment input, where it means gl_FragCoord — so
// this must lower to a plain attribute, and the stage's own position
// write must still be the one and only position output.
//
// Reproduces the skybox of unified-wombat: wombat.dom's
// `viewSpaceNormalVertexEffect` runs first and writes gl_Position =
// Positions, then the user's stage overwrites the position.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";

const Tvec4f: Type = Vec(Tf32, 4);
const Tvec3f: Type = Vec(Tf32, 3);

// the position attribute as the F# plugin spells it: Builtin, not Location
const posInBuiltin = {
  name: "Positions", type: Tvec4f, semantic: "Positions",
  decorations: [{ kind: "Builtin" as const, value: "position" as const }],
};

const viewSpaceNormalVs = {
  name: "vsNormal", stage: "vertex" as const,
  inputs: [posInBuiltin, { name: "Normals", type: Tvec3f, semantic: "Normals", decorations: [{ kind: "Location" as const, value: 1 }] }],
  outputs: [
    { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin" as const, value: "position" as const }] },
    { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location" as const, value: 1 }] },
  ],
};

const skyVs = {
  name: "skyVertex", stage: "vertex" as const,
  inputs: [posInBuiltin],
  outputs: [
    { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin" as const, value: "position" as const }] },
    { name: "SkyNdc", type: Vec(Tf32, 2), semantic: "SkyNdc", decorations: [{ kind: "Location" as const, value: 2 }] },
  ],
};

const skyFs = {
  name: "skyFragment", stage: "fragment" as const,
  inputs: [{ name: "SkyNdc", type: Vec(Tf32, 2), semantic: "SkyNdc", decorations: [{ kind: "Location" as const, value: 2 }] }],
  outputs: [{ name: "Colors", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 0 }] }],
};

const source = `
  function vsNormal(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; ViewSpaceNormal: V3f } {
    return { gl_Position: input.Positions, ViewSpaceNormal: input.Normals.normalize() };
  }
  function skyVertex(input: { Positions: V4f }): { Positions: V4f; SkyNdc: V2f } {
    return { Positions: new V4f(input.Positions.x, input.Positions.y, 0.0, 1.0),
             SkyNdc: new V2f(input.Positions.x, input.Positions.y) };
  }
  function skyFragment(input: { SkyNdc: V2f }): { Colors: V4f } {
    return { Colors: new V4f(input.SkyNdc.x, input.SkyNdc.y, 1.0, 1.0) };
  }
`;

describe("a vertex input spelled @builtin(position)", () => {
  it("emits one position output and keeps the position attribute", () => {
    const r = compileShaderSource(source, [viewSpaceNormalVs, skyVs, skyFs], { target: "wgsl" });
    const vs = r.stages.find((s) => s.stage === "vertex")!;
    expect(vs.source.match(/@builtin\(position\)/g)?.length).toBe(1);
    // the quad's position must survive as a vertex ATTRIBUTE
    expect(vs.bindings.inputs.map((i) => i.name)).toContain("Positions");
  });
});
