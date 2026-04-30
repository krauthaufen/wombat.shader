// Per-Effect compile cache. Cache key = effect.id + sampled-hole
// values hash + target. Same key on a second `compile()` returns
// the same `CompiledEffect` reference (i.e. no re-emit).

import { describe, expect, it } from "vitest";
import { effect, stage } from "@aardworx/wombat.shader-runtime";
import type { EntryDef, Module, Stmt, Type } from "@aardworx/wombat.shader-ir";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec3f: Type = { kind: "Vector", element: Tf32, dim: 3 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };

function fragmentTemplate(): Module {
  // outColor = new V4f(tint, 1.0) where tint is a closure hole.
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
        { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 },
      ],
      type: Tvec4f,
    }},
  };
  const entry: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [], outputs: [{
      name: "outColor", type: Tvec4f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    arguments: [], returnType: { kind: "Void" },
    body, decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
}

describe("Effect compile cache", () => {
  it("same hole values + target → same CompiledEffect (cache hit)", () => {
    const v: number[] = [0.1, 0.2, 0.3];
    const fx = effect(stage(fragmentTemplate(), { tint: () => v }));
    const a = fx.compile({ target: "wgsl" });
    const b = fx.compile({ target: "wgsl" });
    expect(a).toBe(b);
  });

  it("different hole values → distinct CompiledEffects", () => {
    const ref = { tint: [0.1, 0.2, 0.3] as [number, number, number] };
    const fx = effect(stage(fragmentTemplate(), { tint: () => ref.tint }));
    const a = fx.compile({ target: "wgsl" });
    ref.tint = [0.4, 0.5, 0.6];
    const b = fx.compile({ target: "wgsl" });
    expect(a).not.toBe(b);
    // First key is still cached: revert to original values → same a.
    ref.tint = [0.1, 0.2, 0.3];
    const c = fx.compile({ target: "wgsl" });
    expect(c).toBe(a);
  });

  it("different target → distinct cache entries", () => {
    const fx = effect(stage(fragmentTemplate(), { tint: () => [0.1, 0.2, 0.3] }));
    const w = fx.compile({ target: "wgsl" });
    const g = fx.compile({ target: "glsl" });
    expect(w).not.toBe(g);
    expect(w.target).toBe("wgsl");
    expect(g.target).toBe("glsl");
  });

  it("`skipOptimisations` and the optimised path don't collide", () => {
    const fx = effect(stage(fragmentTemplate(), { tint: () => [0.1, 0.2, 0.3] }));
    const o = fx.compile({ target: "wgsl" });
    const r = fx.compile({ target: "wgsl", skipOptimisations: true });
    expect(o).not.toBe(r);
  });
});
