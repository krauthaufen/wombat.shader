// Cross-helper liveness: an input read only while building a LOCAL struct
// value (`i.mp = in.mp`, as the F# plugin lowers a record argument) stays
// live — the field write is not a State write.

import { describe, expect, it } from "vitest";
import { composeStages, linkFragmentOutputs, linkHelpers } from "@aardworx/wombat.shader/passes";
import { Tf32, Tvoid, Vec, type EntryDef, type Module, type Type, type Var } from "@aardworx/wombat.shader/ir";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Trec: Type = { kind: "Struct", name: "Rec", fields: [{ name: "mp", type: Tvec3f }] };

describe("linkHelpers: local struct field writes", () => {
  it("keeps an input that only feeds a local struct field", () => {
    const rec: Var = { name: "r", type: Trec, mutable: true };
    const A: EntryDef = {
      name: "A", stage: "fragment",
      inputs: [{ name: "mp", type: Tvec3f, semantic: "mp", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: { kind: "Sequential", body: [
        { kind: "Declare", var: rec },
        { kind: "Write",
          target: { kind: "LField", target: { kind: "LVar", var: rec, type: Trec }, name: "mp", type: Tvec3f },
          value: { kind: "ReadInput", scope: "Input", name: "mp", type: Tvec3f } },
        { kind: "WriteOutput", name: "outColor", value: { kind: "Expr", value: { kind: "NewVector", type: Tvec4f, components: [
          { kind: "Field", target: { kind: "Var", var: rec, type: Trec }, name: "mp", type: Tvec3f },
          { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 } ] } } },
      ] },
      decorations: [],
    };
    const B: EntryDef = {
      name: "B", stage: "fragment",
      inputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: { kind: "WriteOutput", name: "outColor", value: { kind: "Expr", value: { kind: "ReadInput", scope: "Input", name: "outColor", type: Tvec4f } } },
      decorations: [],
    };
    let m: Module = { types: [Trec as never], values: [{ kind: "Entry", entry: A }, { kind: "Entry", entry: B }] };
    m = composeStages(m);
    m = linkFragmentOutputs(m, { locations: new Map([["outColor", 0]]) });
    m = linkHelpers(m);
    const wrapper = m.values.flatMap((v) => v.kind === "Entry" ? [v.entry] : [])[0]!;
    expect(wrapper.inputs.map((p) => p.name)).toContain("mp");
    const state = m.types.find((t) => t.kind === "Struct" && t.name !== "Rec");
    expect(state?.kind === "Struct" && state.fields.map((f) => f.name)).toContain("mp");
  });
});
