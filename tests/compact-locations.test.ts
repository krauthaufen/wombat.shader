// Tests for compactLocations — densely renumber surviving cross-stage
// varyings after pruning may have left gaps / out-of-cap numbers.

import { describe, expect, it } from "vitest";
import {
  Tf32,
  Tu32,
  Tvoid,
  Vec,
  type EntryParameter,
  type Module,
  type ParamDecoration,
} from "@aardworx/wombat.shader/ir";
import { compactLocations } from "@aardworx/wombat.shader/passes";

const Tvec2f = Vec(Tf32, 2);
const Tvec3f = Vec(Tf32, 3);

function out(name: string, type = Tvec3f, location?: number, flat = false): EntryParameter {
  const decorations: ParamDecoration[] = [];
  if (location !== undefined) decorations.push({ kind: "Location", value: location });
  if (flat) decorations.push({ kind: "Interpolation", mode: "flat" });
  return { name, type, semantic: name, decorations };
}

function builtinPos(): EntryParameter {
  return { name: "gl_Position", type: Vec(Tf32, 4), semantic: "position", decorations: [{ kind: "Builtin", value: "position" }] };
}

function locOf(p: EntryParameter): number | undefined {
  const d = p.decorations.find((x) => x.kind === "Location");
  return d && d.kind === "Location" ? d.value : undefined;
}

describe("compactLocations", () => {
  it("compacts sparse surviving VS-output locations to a dense 0..n-1 run", () => {
    // Mirrors the heap decoder case: dead uniform carriers were pruned,
    // leaving survivors pinned at 0,1,3,20,21 — location 20/21 exceed the
    // 16-location cap even though only 5 varyings cross the boundary.
    const m: Module = {
      types: [],
      values: [
        {
          kind: "Entry", entry: {
            name: "vs", stage: "vertex", inputs: [],
            outputs: [
              out("_h_drawIdx", Tu32, 0, true),
              out("ViewSpaceNormal", Tvec3f, 1),
              builtinPos(),
              out("ViewNormal", Tvec3f, 3),
              out("ViewPos", Tvec3f, 20),
              out("DiffuseColorCoordinates", Tvec2f, 21),
            ],
            arguments: [], returnType: Tvoid,
            body: { kind: "Sequential", body: [] }, decorations: [],
          },
        },
        {
          kind: "Entry", entry: {
            name: "fs", stage: "fragment",
            inputs: [
              out("_h_drawIdx", Tu32, 0, true),
              out("ViewSpaceNormal", Tvec3f, 1),
              out("ViewNormal", Tvec3f, 3),
              out("ViewPos", Tvec3f, 20),
              out("DiffuseColorCoordinates", Tvec2f, 21),
            ],
            outputs: [], arguments: [], returnType: Tvoid,
            body: { kind: "Sequential", body: [] }, decorations: [],
          },
        },
      ],
    };

    const r = compactLocations(m);
    const vs = r.values.find((x) => x.kind === "Entry" && x.entry.name === "vs")!;
    const fs = r.values.find((x) => x.kind === "Entry" && x.entry.name === "fs")!;
    if (vs.kind !== "Entry" || fs.kind !== "Entry") throw new Error("bad");

    // VS outputs densely renumbered in declaration order (builtin skipped).
    const vsByName = Object.fromEntries(vs.entry.outputs.map((p) => [p.name, locOf(p)]));
    expect(vsByName["_h_drawIdx"]).toBe(0);
    expect(vsByName["ViewSpaceNormal"]).toBe(1);
    expect(vsByName["ViewNormal"]).toBe(2);
    expect(vsByName["ViewPos"]).toBe(3);
    expect(vsByName["DiffuseColorCoordinates"]).toBe(4);
    // Builtin keeps its @builtin slot (no Location).
    expect(locOf(vs.entry.outputs.find((p) => p.name === "gl_Position")!)).toBeUndefined();

    // FS inputs stay aligned to the VS outputs by name.
    const fsByName = Object.fromEntries(fs.entry.inputs.map((p) => [p.name, locOf(p)]));
    expect(fsByName).toEqual({
      _h_drawIdx: 0, ViewSpaceNormal: 1, ViewNormal: 2, ViewPos: 3, DiffuseColorCoordinates: 4,
    });

    // Every surviving location is within the 16-location cap.
    for (const p of vs.entry.outputs) {
      const l = locOf(p);
      if (l !== undefined) expect(l).toBeLessThan(16);
    }
  });

  it("is a no-op on an already-dense module (idempotent)", () => {
    const m: Module = {
      types: [],
      values: [
        {
          kind: "Entry", entry: {
            name: "vs", stage: "vertex", inputs: [],
            outputs: [builtinPos(), out("a", Tvec3f, 0), out("b", Tvec2f, 1)],
            arguments: [], returnType: Tvoid,
            body: { kind: "Sequential", body: [] }, decorations: [],
          },
        },
      ],
    };
    expect(compactLocations(m)).toBe(m);
  });
});
