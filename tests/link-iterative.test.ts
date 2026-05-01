// Multi-step DCE through linkFragmentOutputs + pruneCrossStage:
// dropping a fragment output cascades into dropped cross-stage inputs,
// which cascades into dropped vertex outputs, which (after VS DCE) drops
// vertex-input reads. The pipeline must reach a fixed point.

import { describe, expect, it } from "vitest";
import {
  Tf32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
  type Module,
  type Stmt,
  type Type,
} from "@aardworx/wombat.shader/ir";
import {
  linkCrossStage,
  linkFragmentOutputs,
  pruneCrossStage,
} from "@aardworx/wombat.shader/passes";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });

function entriesOf(m: Module): EntryDef[] {
  return m.values.flatMap((x) => x.kind === "Entry" ? [x.entry] : []);
}
function flat(s: Stmt): Stmt[] { return s.kind === "Sequential" ? s.body.flatMap(flat) : [s]; }

describe("iterative cross-stage DCE", () => {
  it("dropping a fragment output cascades two levels back", () => {
    // VS reads three vertex attributes (Position, Normal, Color), writes
    // gl_Position + vn + vc (Normal/Color carriers).
    // FS has TWO outputs: outColor (uses vn), debug (uses vc).
    // We then `linkFragmentOutputs` with only outColor. After:
    //  - debug fragment output dropped, its body DCE'd → FS no longer
    //    reads vc.
    //  - pruneCrossStage drops vc from VS outputs, DCE's the vc write,
    //    which removes the read of `Color` from VS.
    //  - VS no longer has `Color` as a *used* attribute (the input
    //    declaration may remain — vertex inputs aren't pruned by this
    //    pass — but the body no longer references it).

    const a_pos: EntryDef["inputs"][number] = {
      name: "Position", type: Tvec3f, semantic: "Position",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const a_normal: EntryDef["inputs"][number] = {
      name: "Normal", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 1 }],
    };
    const a_color: EntryDef["inputs"][number] = {
      name: "Color", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 2 }],
    };

    const gl_Position: EntryDef["outputs"][number] = {
      name: "gl_Position", type: Tvec4f, semantic: "Position",
      decorations: [{ kind: "Builtin", value: "position" }],
    };
    const vn: EntryDef["outputs"][number] = {
      name: "vn", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const vc: EntryDef["outputs"][number] = {
      name: "vc", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 1 }],
    };

    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [a_pos, a_normal, a_color],
      outputs: [gl_Position, vn, vc],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential", body: [
          { kind: "WriteOutput", name: "gl_Position",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [readI("Position", Tvec3f), constF(1)], type: Tvec4f } } },
          { kind: "WriteOutput", name: "vn",
            value: { kind: "Expr", value: readI("Normal", Tvec3f) } },
          { kind: "WriteOutput", name: "vc",
            value: { kind: "Expr", value: readI("Color", Tvec3f) } },
        ],
      },
      decorations: [],
    };

    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [
        { name: "vn", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] },
        { name: "vc", type: Tvec3f, semantic: "Color",  decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
        { name: "debug",    type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 1 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential", body: [
          { kind: "WriteOutput", name: "outColor",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [readI("vn", Tvec3f), constF(1)], type: Tvec4f } } },
          { kind: "WriteOutput", name: "debug",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [readI("vc", Tvec3f), constF(1)], type: Tvec4f } } },
        ],
      },
      decorations: [],
    };

    const m: Module = {
      types: [],
      values: [
        { kind: "Entry", entry: vertex },
        { kind: "Entry", entry: fragment },
      ],
    };

    // Pipeline order under test.
    const a = linkCrossStage(m);
    const b = linkFragmentOutputs(a, { locations: new Map([["outColor", 0]]) });
    const c = pruneCrossStage(b);

    const vs2 = entriesOf(c).find((e) => e.stage === "vertex")!;
    const fs2 = entriesOf(c).find((e) => e.stage === "fragment")!;

    // Final FS: only outColor. Reads vn but not vc.
    expect(fs2.outputs.map((o) => o.name)).toEqual(["outColor"]);
    expect(JSON.stringify(fs2.body)).not.toContain("\"vc\"");

    // Final VS: gl_Position + vn (vc dropped). Body reads Position + Normal,
    // but no longer reads Color.
    expect(vs2.outputs.map((o) => o.name)).toEqual(["gl_Position", "vn"]);
    const writes = flat(vs2.body).flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writes).toContain("gl_Position");
    expect(writes).toContain("vn");
    expect(writes).not.toContain("vc");
    // The Color attribute read should be gone after DCE.
    const bodyJson = JSON.stringify(vs2.body);
    expect(bodyJson).not.toContain("\"Color\"");
    expect(bodyJson).toContain("\"Normal\"");
    expect(bodyJson).toContain("\"Position\"");
  });
});
