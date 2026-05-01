// linkFragmentOutputs: re-pin fragment outputs to a target framebuffer
// signature; drop+DCE outputs not in the layout; pass builtins through.

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
import { linkFragmentOutputs } from "@aardworx/wombat.shader/passes";

const Tvec4f: Type = Vec(Tf32, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });

function entriesOf(m: Module): EntryDef[] {
  return m.values.flatMap((x) => x.kind === "Entry" ? [x.entry] : []);
}

describe("linkFragmentOutputs", () => {
  it("re-pins surviving outputs and DCEs the dropped ones", () => {
    const frag: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 99 }] },
        { name: "pickId",   type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 100 }] },
        { name: "junk",     type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 5 }] },
        // Builtin depth output — must pass through unchanged.
        { name: "Depth",    type: Tf32,   semantic: "Depth", decorations: [{ kind: "Builtin", value: "frag_depth" }] },
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
            kind: "WriteOutput", name: "junk",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [constF(9), constF(9), constF(9), constF(9)], type: Tvec4f } },
          },
          {
            kind: "WriteOutput", name: "pickId",
            value: { kind: "Expr", value: { kind: "NewVector",
              components: [constF(0.5), constF(0), constF(0), constF(1)], type: Tvec4f } },
          },
          {
            kind: "WriteOutput", name: "Depth",
            value: { kind: "Expr", value: constF(0.25) },
          },
        ],
      },
      decorations: [],
    };

    const m: Module = { types: [], values: [{ kind: "Entry", entry: frag }] };

    const linked = linkFragmentOutputs(m, {
      locations: new Map([["outColor", 0], ["pickId", 1]]),
    });
    const out = entriesOf(linked)[0]!;
    const byName = (n: string) => out.outputs.find((p) => p.name === n);

    // junk is dropped; outColor / pickId survive with their layout
    // locations; Depth (builtin) passes through.
    const names = out.outputs.map((p) => p.name);
    expect(names).toContain("outColor");
    expect(names).toContain("pickId");
    expect(names).toContain("Depth");
    expect(names).not.toContain("junk");

    const ocLoc = byName("outColor")!.decorations.find((d) => d.kind === "Location");
    const piLoc = byName("pickId")!.decorations.find((d) => d.kind === "Location");
    expect(ocLoc?.kind === "Location" ? ocLoc.value : -1).toBe(0);
    expect(piLoc?.kind === "Location" ? piLoc.value : -1).toBe(1);

    // Depth's Builtin decoration intact.
    const depthBI = byName("Depth")!.decorations.find((d) => d.kind === "Builtin");
    expect(depthBI?.kind === "Builtin" ? depthBI.value : "").toBe("frag_depth");

    // The body no longer mentions `junk`.
    expect(JSON.stringify(out.body)).not.toContain("junk");
  });

  it("non-fragment entries are passed through unchanged", () => {
    const vs: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [],
      outputs: [{ name: "v_extra", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: { kind: "Nop" },
      decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry: vs }] };
    const linked = linkFragmentOutputs(m, { locations: new Map() });
    const out = entriesOf(linked)[0]!;
    expect(out.outputs.map((o) => o.name)).toEqual(["v_extra"]);
  });
});
