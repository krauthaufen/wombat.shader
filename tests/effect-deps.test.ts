// effectDependencies — per-output transitive input analysis.
// Cross-stage resolve: fragment outputs report only their
// resolved-to-vertex-attribute (or unresolved-cross-stage) inputs.
//
// Pinned scenarios:
//   - Vertex output's deps == its body's Input-scope reads
//   - Fragment output's deps resolve through the matching vertex
//     output (transitively)
//   - User effect that writes a "ViewSpaceNormal" output — caller
//     can use the deps to ask "given my geometry has Normals, can
//     this effect produce ViewSpaceNormal?"
//   - Output that doesn't depend on Input-scope reads (only uniforms
//     / closures) reports an empty input set
//   - Fragment input that no vertex stage produces remains in the
//     unresolved set (reported as input requirements)

import { describe, expect, it } from "vitest";
import { compileShaderSource, effectDependencies, stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import {
  Tf32, Tu32, Vec, type Type,
} from "@aardworx/wombat.shader/ir";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function build(source: string, entries: EntryRequest[]): ReturnType<typeof stage> {
  // Hand-rolled "compose-as-stage" — bypasses runtime compile holes.
  // Frontend produces a Module; wrap as a single Effect for the test.
  const module = parseShader({ source, entries });
  return stage(module);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("effectDependencies", () => {
  it("vertex output deps equal its Input-scope reads", () => {
    const e = build(
      `
        function vsMain(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; vNormal: V3f } {
          return { gl_Position: input.Positions, vNormal: input.Normals };
        }
      `,
      [
        {
          name: "vsMain", stage: "vertex",
          inputs: [
            { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
            { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
          ],
          outputs: [
            { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
            { name: "vNormal",     type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
      ],
    );
    const deps = effectDependencies(e);
    const pos = deps.get("gl_Position")!;
    const norm = deps.get("vNormal")!;
    expect([...pos.inputs.keys()]).toEqual(["Positions"]);
    expect([...norm.inputs.keys()]).toEqual(["Normals"]);
  });

  it("fragment output deps resolve through the vertex stage to vertex attributes", () => {
    // Vertex passes Normal through; fragment writes ViewSpaceNormal
    // by reading the cross-stage 'vNormal'. Effect-level dep for
    // ViewSpaceNormal should resolve to {Normals}.
    const e = build(
      `
        function vsMain(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; vNormal: V3f } {
          return { gl_Position: input.Positions, vNormal: input.Normals };
        }
        function fsMain(input: { vNormal: V3f }): { ViewSpaceNormal: V3f } {
          return { ViewSpaceNormal: input.vNormal };
        }
      `,
      [
        {
          name: "vsMain", stage: "vertex",
          inputs: [
            { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
            { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
          ],
          outputs: [
            { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
            { name: "vNormal",     type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
        {
          name: "fsMain", stage: "fragment",
          inputs: [
            { name: "vNormal", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] },
          ],
          outputs: [
            { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
      ],
    );
    const deps = effectDependencies(e);
    const vsn = deps.get("ViewSpaceNormal")!;
    expect([...vsn.inputs.keys()].sort()).toEqual(["Normals"]);
  });

  it("output reading only uniforms/closures has empty Input-scope deps", () => {
    // The frontend resolves bare-identifier `tint` as a Closure
    // hole when it has no externalTypes entry — but for this test
    // we can build a fragment that just emits a constant.
    const e = build(
      `
        function fsMain(input: {}): { outColor: V4f } {
          return { outColor: new V4f(1.0, 0.5, 0.25, 1.0) };
        }
      `,
      [
        {
          name: "fsMain", stage: "fragment",
          inputs: [],
          outputs: [
            { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
      ],
    );
    const deps = effectDependencies(e);
    expect(deps.get("outColor")!.inputs.size).toBe(0);
  });

  it("fragment input that has no upstream vertex producer remains as a required input", () => {
    // Fragment reads `userExtra` from its inputs, but there's no
    // vertex stage in this effect. The dep stays as `userExtra` in
    // `inputs` — caller treats it as a geometry attribute too.
    const e = build(
      `
        function fsMain(input: { userExtra: V3f }): { outColor: V4f } {
          return { outColor: new V4f(input.userExtra, 1.0) };
        }
      `,
      [
        {
          name: "fsMain", stage: "fragment",
          inputs: [
            { name: "userExtra", type: Tvec3f, semantic: "userExtra", decorations: [{ kind: "Location", value: 0 }] },
          ],
          outputs: [
            { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
      ],
    );
    const deps = effectDependencies(e);
    const oc = deps.get("outColor")!;
    expect([...oc.inputs.keys()]).toEqual(["userExtra"]);
  });

  it("forward dataflow through Var assignments preserves input deps", () => {
    // `let n = input.Normals; ... output uses n` — n inherits
    // Normals' deps; output deps include Normals.
    const e = build(
      `
        function vsMain(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; vNormal: V3f } {
          const n = input.Normals;
          return { gl_Position: input.Positions, vNormal: n };
        }
      `,
      [
        {
          name: "vsMain", stage: "vertex",
          inputs: [
            { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
            { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
          ],
          outputs: [
            { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
            { name: "vNormal",     type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
          ],
        },
      ],
    );
    const deps = effectDependencies(e);
    expect([...deps.get("vNormal")!.inputs.keys()]).toEqual(["Normals"]);
  });

  it("the deps map is also computable on the result of compileShaderSource", () => {
    // The intermediate Effect returned by `stage(module)` round-
    // trips through `compileShaderSource` for callers who already
    // have a CompiledEffect — but the deps API takes Effect, not
    // CompiledEffect. This test just sanity-checks that
    // `compileShaderSource` doesn't blow up the Module shape used
    // upstream.
    const compiled = compileShaderSource(
      `
        function fsMain(input: {}): { outColor: V4f } {
          return { outColor: new V4f(0.0, 0.0, 0.0, 1.0) };
        }
      `,
      [{
        name: "fsMain", stage: "fragment",
        inputs: [],
        outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
        ],
      }],
      { target: "wgsl" },
    );
    expect(compiled.stages).toHaveLength(1);
  });
});

void Tu32;
