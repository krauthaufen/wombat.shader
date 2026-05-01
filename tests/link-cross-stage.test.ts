// linkCrossStage — semantic-based VS<->FS matching + auto pass-through.

import { describe, expect, it, vi } from "vitest";
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

function flatStmts(s: Stmt): Stmt[] {
  return s.kind === "Sequential" ? s.body.flatMap(flatStmts) : [s];
}

const a_normal: EntryDef["inputs"][number] = {
  name: "Normal", type: Tvec3f, semantic: "Normal",
  decorations: [{ kind: "Location", value: 0 }],
};
const a_pos: EntryDef["inputs"][number] = {
  name: "Position", type: Tvec3f, semantic: "Position",
  decorations: [{ kind: "Location", value: 1 }],
};
const gl_Position: EntryDef["outputs"][number] = {
  name: "gl_Position", type: Tvec4f, semantic: "Position",
  decorations: [{ kind: "Builtin", value: "position" }],
};

describe("linkCrossStage", () => {
  it("renames FS input to VS-output name when semantics match", () => {
    // VS produces `vn` with semantic "Normal".
    // FS reads `nrm` with semantic "Normal".
    const vn: EntryDef["outputs"][number] = {
      name: "vn", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [a_normal],
      outputs: [gl_Position, vn],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential", body: [
          { kind: "WriteOutput", name: "gl_Position",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [readI("Position", Tvec3f), constF(1)], type: Tvec4f } } },
          { kind: "WriteOutput", name: "vn",
            value: { kind: "Expr", value: readI("Normal", Tvec3f) } },
        ],
      },
      decorations: [],
    };
    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [{ name: "nrm", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("nrm", Tvec3f), constF(1)], type: Tvec4f } },
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

    const linked = linkCrossStage(m);
    const vs2 = entriesOf(linked).find((e) => e.stage === "vertex")!;
    const fs2 = entriesOf(linked).find((e) => e.stage === "fragment")!;

    // VS unchanged.
    expect(vs2.outputs.map((o) => o.name)).toContain("vn");
    // FS input renamed nrm -> vn.
    expect(fs2.inputs.map((i) => i.name)).toEqual(["vn"]);
    // FS body now reads `vn`, not `nrm`.
    expect(JSON.stringify(fs2.body)).toContain("\"vn\"");
    expect(JSON.stringify(fs2.body)).not.toContain("\"nrm\"");
  });

  it("auto-injects a pass-through when no VS output but VS attribute matches", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [a_pos, a_normal],
      outputs: [gl_Position],
      arguments: [], returnType: Tvoid,
      body: { kind: "Sequential", body: [
        { kind: "WriteOutput", name: "gl_Position",
          value: { kind: "Expr", value: { kind: "NewVector",
            components: [readI("Position", Tvec3f), constF(1)], type: Tvec4f } } },
      ] },
      decorations: [],
    };
    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [{ name: "nrm", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("nrm", Tvec3f), constF(1)], type: Tvec4f } },
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

    const linked = linkCrossStage(m);
    const vs2 = entriesOf(linked).find((e) => e.stage === "vertex")!;
    const fs2 = entriesOf(linked).find((e) => e.stage === "fragment")!;

    // VS now has cross-stage output `nrm` (no rename — FS keeps its name).
    expect(vs2.outputs.map((o) => o.name)).toContain("nrm");

    // VS body contains a pass-through Write of `Normal` attribute -> _pt_nrm
    // and a WriteOutput("nrm", ...) of the carrier.
    const flat = flatStmts(vs2.body);
    const declared = flat.flatMap((s) => s.kind === "Declare" ? [s.var.name] : []);
    expect(declared).toContain("_pt_nrm");
    const writeOuts = flat.flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writeOuts).toContain("nrm");

    // FS unchanged (no rename necessary).
    expect(fs2.inputs.map((i) => i.name)).toEqual(["nrm"]);

    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("VS extra outputs are pruned by pruneCrossStage after linking", () => {
    // VS writes vn AND vextra; FS reads only vn.
    const vn: EntryDef["outputs"][number] = {
      name: "vn", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const vextra: EntryDef["outputs"][number] = {
      name: "vextra", type: Tvec3f, semantic: "Extra",
      decorations: [{ kind: "Location", value: 1 }],
    };
    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [a_normal, a_pos],
      outputs: [gl_Position, vn, vextra],
      arguments: [], returnType: Tvoid,
      body: { kind: "Sequential", body: [
        { kind: "WriteOutput", name: "gl_Position",
          value: { kind: "Expr", value: { kind: "NewVector",
            components: [readI("Position", Tvec3f), constF(1)], type: Tvec4f } } },
        { kind: "WriteOutput", name: "vn",
          value: { kind: "Expr", value: readI("Normal", Tvec3f) } },
        { kind: "WriteOutput", name: "vextra",
          value: { kind: "Expr", value: { kind: "NewVector",
            components: [constF(1), constF(0), constF(0)], type: Tvec3f } } },
      ] },
      decorations: [],
    };
    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [{ name: "vn", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("vn", Tvec3f), constF(1)], type: Tvec4f } },
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

    const linked = linkCrossStage(m);
    const linkedFb = linkFragmentOutputs(linked, { locations: new Map([["outColor", 0]]) });
    const pruned = pruneCrossStage(linkedFb);
    const vs2 = entriesOf(pruned).find((e) => e.stage === "vertex")!;
    expect(vs2.outputs.map((o) => o.name)).toEqual(["gl_Position", "vn"]);
    const flat = flatStmts(vs2.body);
    const writeOuts = flat.flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writeOuts).not.toContain("vextra");
  });

  it("does not inject pass-through when neither VS output nor matching attribute exists", () => {
    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [a_pos],
      outputs: [gl_Position],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "gl_Position",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("Position", Tvec3f), constF(1)], type: Tvec4f } },
      },
      decorations: [],
    };
    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      // FS reads "nrm" w/ semantic Normal, but VS has no Normal in inputs/outputs.
      inputs: [{ name: "nrm", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("nrm", Tvec3f), constF(1)], type: Tvec4f } },
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
    const linked = linkCrossStage(m);
    const vs2 = entriesOf(linked).find((e) => e.stage === "vertex")!;
    // No pass-through synthesised — VS unchanged.
    expect(vs2.outputs.map((o) => o.name)).toEqual(["gl_Position"]);
    const flat = flatStmts(vs2.body);
    expect(flat.flatMap((s) => s.kind === "Declare" ? [s.var.name] : [])).not.toContain("_pt_nrm");
  });
});
