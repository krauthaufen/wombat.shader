// A fragment composed from stages that each declare @builtin(position)
// under their own name gets ONE parameter (reads lower to the semantic).

import { describe, expect, it } from "vitest";
import { Tf32, Tvoid, Vec, type EntryDef, type Module, type Type } from "@aardworx/wombat.shader/ir";
import { emitWgsl } from "@aardworx/wombat.shader/wgsl";

const Tvec4f: Type = Vec(Tf32, 4);
const fragCoord = (name: string) => ({ name, type: Tvec4f, semantic: "FragCoord", decorations: [{ kind: "Builtin" as const, value: "position" }] });

describe("wgsl emit: builtin inputs", () => {
  it("declares a builtin shared by several inputs once", () => {
    const entry: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [fragCoord("pos"), fragCoord("fragCoord")],
      outputs: [{ name: "Colors", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "Colors",
        value: { kind: "Expr", value: { kind: "ReadInput", scope: "Builtin", name: "position", type: Tvec4f } },
      },
      decorations: [],
    };
    const mod: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const wgsl = emitWgsl(mod).source;
    expect(wgsl.match(/@builtin\(position\)/g)?.length).toBe(1);
  });
});
