// resolveHoles — closure holes inlined as IR constants.

import { describe, expect, it } from "vitest";
import type { EntryDef, Module, Stmt, Type } from "@aardworx/wombat.shader/ir";
import { resolveHoles } from "@aardworx/wombat.shader/passes";
import { compileModule } from "@aardworx/wombat.shader";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };
const Tvec3f: Type = { kind: "Vector", element: Tf32, dim: 3 };

function fragmentTemplate(): Module {
  // Equivalent to:
  //   fragment(() => ({ outColor: new V4f(tint, 1.0) }))
  // where `tint` is a closure-captured V3f hole.
  const tintRead = { kind: "ReadInput", scope: "Closure", name: "tint", type: Tvec3f } as const;
  const out: Stmt = {
    kind: "WriteOutput",
    name: "outColor",
    value: {
      kind: "Expr",
      value: {
        kind: "NewVector",
        components: [
          { kind: "VecSwizzle", value: tintRead, comps: ["x"], type: Tf32 },
          { kind: "VecSwizzle", value: tintRead, comps: ["y"], type: Tf32 },
          { kind: "VecSwizzle", value: tintRead, comps: ["z"], type: Tf32 },
          { kind: "Const", value: { kind: "Float", value: 1.0 }, type: Tf32 },
        ],
        type: Tvec4f,
      },
    },
  };
  const entry: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [], outputs: [{
      name: "outColor", type: Tvec4f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    arguments: [],
    returnType: { kind: "Void" },
    body: out,
    decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
}

describe("resolveHoles", () => {
  it("inlines a V3f hole into NewVector", () => {
    const filled = resolveHoles(fragmentTemplate(), {
      tint: [0.25, 0.5, 0.75],
    });
    const json = JSON.stringify(filled);
    expect(json).not.toContain('"Closure"');
    // The hole was lowered into a NewVector(0.25, 0.5, 0.75) substituted
    // for each occurrence of `tint`.
    expect(json).toContain('"value":0.25');
    expect(json).toContain('"value":0.5');
    expect(json).toContain('"value":0.75');
  });

  it("end-to-end: filled template compiles to GLSL/WGSL with no closure references", () => {
    const filled = resolveHoles(fragmentTemplate(), { tint: { x: 0.1, y: 0.2, z: 0.3 } });
    const glsl = compileModule(filled, { target: "glsl" });
    const wgsl = compileModule(filled, { target: "wgsl" });
    for (const r of [glsl, wgsl]) {
      const src = r.stages[0]!.source;
      expect(src).not.toMatch(/tint/);
      expect(src).toMatch(/0\.1/);
      expect(src).toMatch(/0\.2/);
      expect(src).toMatch(/0\.3/);
    }
  });

  it("missing hole value raises a clear error", () => {
    expect(() => resolveHoles(fragmentTemplate(), {})).toThrow(/missing value for closure hole "tint"/);
  });

  it("scalar holes accept plain numbers", () => {
    // Build a tiny module: fragment that emits new V4f(t, t, t, 1).
    const t = { kind: "ReadInput", scope: "Closure", name: "t", type: Tf32 } as const;
    const body: Stmt = {
      kind: "WriteOutput",
      name: "outColor",
      value: { kind: "Expr", value: {
        kind: "NewVector",
        components: [t, t, t, { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 }],
        type: Tvec4f,
      }},
    };
    const m: Module = { types: [], values: [{
      kind: "Entry", entry: {
        name: "fsMain", stage: "fragment", inputs: [],
        outputs: [{
          name: "outColor", type: Tvec4f, semantic: "Color",
          decorations: [{ kind: "Location", value: 0 }],
        }],
        arguments: [], returnType: { kind: "Void" }, body, decorations: [],
      },
    }] };
    const filled = resolveHoles(m, { t: 0.42 });
    const json = JSON.stringify(filled);
    expect(json).not.toContain('"Closure"');
    expect(json).toContain('"value":0.42');
  });
});
