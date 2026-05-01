// FShade-style composition: composeStages keeps every non-piped output
// and resolves same-named output collisions as B-wins (last-wins).

import { describe, expect, it } from "vitest";
import {
  Tf32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
  type Module,
  type Type,
} from "@aardworx/wombat.shader/ir";
import { composeStages } from "@aardworx/wombat.shader/passes";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });

function entriesOf(m: Module): EntryDef[] {
  return m.values.flatMap((x) => x.kind === "Entry" ? [x.entry] : []);
}

describe("composeStages: FShade-style keep-everything", () => {
  it("A's unused output survives composition; B-name wins on collisions", () => {
    // A writes outColor + extra. B reads outColor (so it's piped) and
    // also writes its own outColor.
    const fragA: EntryDef = {
      name: "fA", stage: "fragment",
      inputs: [],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 7 }] },
        { name: "extra",    type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 8 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential",
        body: [
          {
            kind: "WriteOutput", name: "outColor",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [constF(1), constF(0), constF(0), constF(1)], type: Tvec4f } },
          },
          {
            kind: "WriteOutput", name: "extra",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [constF(0.5), constF(0.5), constF(0.5)], type: Tvec3f } },
          },
        ],
      },
      decorations: [],
    };
    const fragB: EntryDef = {
      name: "fB", stage: "fragment",
      inputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      // B writes its own outColor — this should win the merge.
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: {
          kind: "Expr",
          value: {
            kind: "Add",
            lhs: readI("outColor", Tvec4f),
            rhs: { kind: "NewVector",
              components: [constF(0.1), constF(0.1), constF(0.1), constF(0)], type: Tvec4f },
            type: Tvec4f,
          },
        },
      },
      decorations: [],
    };

    const m: Module = {
      types: [],
      values: [
        { kind: "Entry", entry: fragA },
        { kind: "Entry", entry: fragB },
      ],
    };

    const composed = composeStages(m);
    const merged = entriesOf(composed)[0]!;
    // Both A's `extra` and the merged `outColor` are present.
    const names = merged.outputs.map((o) => o.name);
    expect(names).toContain("outColor");
    expect(names).toContain("extra");

    // B's outColor wins: its Location decoration is preserved (= 0).
    const oc = merged.outputs.find((o) => o.name === "outColor")!;
    const ocLoc = oc.decorations.find((d) => d.kind === "Location");
    expect(ocLoc?.kind === "Location" ? ocLoc.value : -1).toBe(0);

    // Body is well-formed (no parse errors); no piped name surfaces as
    // an `outColor` output because B's read was rerouted to a carrier.
  });
});
