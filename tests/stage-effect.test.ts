// Stage / Effect — closure-hole-driven compile path.

import { describe, expect, it } from "vitest";
import type { EntryDef, Module, Stmt, Type } from "@aardworx/wombat.shader/ir";
import { effect, stage } from "@aardworx/wombat.shader";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec2f: Type = { kind: "Vector", element: Tf32, dim: 2 };
const Tvec3f: Type = { kind: "Vector", element: Tf32, dim: 3 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };

function fragmentTemplate(): Module {
  // fragment(({ v_uv }) => ({ outColor: new V4f(v_uv, tint.x, 1.0).mul(tint) }))
  // simplified: outColor = new V4f(tint.x, tint.y, tint.z, 1.0)
  const tintRead = { kind: "ReadInput", scope: "Closure", name: "tint", type: Tvec3f } as const;
  const body: Stmt = {
    kind: "WriteOutput",
    name: "outColor",
    value: { kind: "Expr", value: {
      kind: "NewVector",
      components: [
        { kind: "VecSwizzle", value: tintRead, comps: ["x"], type: Tf32 },
        { kind: "VecSwizzle", value: tintRead, comps: ["y"], type: Tf32 },
        { kind: "VecSwizzle", value: tintRead, comps: ["z"], type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 1.0 }, type: Tf32 },
      ],
      type: Tvec4f,
    }},
  };
  const entry: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [{
      name: "v_uv", type: Tvec2f, semantic: "Uv",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    outputs: [{
      name: "outColor", type: Tvec4f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    arguments: [],
    returnType: { kind: "Void" },
    body, decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
}

describe("stage / effect", () => {
  it("compiles a stage with closure holes via getters", () => {
    let r = 0.1, g = 0.2, b = 0.3;
    const fx = effect(stage(fragmentTemplate(), { tint: () => [r, g, b] }));
    const compiled = fx.compile({ target: "wgsl" });
    const wgsl = compiled.stages[0]!.source;
    expect(wgsl).not.toContain("tint");
    expect(wgsl).toMatch(/0\.1/);
    expect(wgsl).toMatch(/0\.2/);
    expect(wgsl).toMatch(/0\.3/);
  });

  it("re-sampling the getters reflects new values on next compile", () => {
    const ref = { tint: [1.0, 0.0, 0.0] as [number, number, number] };
    const fx = effect(stage(fragmentTemplate(), { tint: () => ref.tint }));

    const a = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(a).toMatch(/1\.0/);

    ref.tint = [0.0, 1.0, 0.0];
    const b = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(b).toMatch(/0\.0f?,\s*1\.0f?,\s*0\.0f?/);
  });

  it("missing hole getter throws at compile time", () => {
    const fx = effect(stage(fragmentTemplate(), {}));
    expect(() => fx.compile({ target: "wgsl" })).toThrow(/closure hole "tint"/);
  });

  it("effect(...) flattens single-stage Effects (vertex + fragment)", () => {
    // stage() returns a single-stage Effect; effect(e1, e2) flattens
    // their stage lists into a single Effect with two stages.
    const fxA = stage(fragmentTemplate(), { tint: () => [0.1, 0.2, 0.3] });
    const fxB = stage(fragmentTemplate(), { tint: () => [0.4, 0.5, 0.6] });
    const merged = effect(fxA, fxB);
    expect(merged.stages).toHaveLength(2);
  });
});
