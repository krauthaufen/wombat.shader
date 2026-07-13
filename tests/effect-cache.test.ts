// Effect compile cache. Cache key = effect.id + target (+ skipOpt +
// fragment-output-layout + instance attrs) + the sampled closure-hole
// values. Same key on a second `compile()` returns the *same*
// `CompiledEffect` reference (no re-emit). Hole values ARE part of the
// key: they're specialised into the emitted source as literals, so a
// changed capture must produce a fresh compile rather than stale
// source.

import { describe, expect, it } from "vitest";
import { effect, stage } from "@aardworx/wombat.shader";
import type { EntryDef, Module, Stmt, Type } from "@aardworx/wombat.shader/ir";

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

// Each test uses a fresh explicit stage id so it gets its own cache
// slot — otherwise structurally-identical templates across tests (same
// `hashModule`) would collide on the now-hole-independent cache key.
let idSeq = 0;
const freshId = (): string => `test/effect-cache/${++idSeq}`;

describe("Effect compile cache", () => {
  it("second compile with same options → same CompiledEffect (cache hit)", () => {
    const v: number[] = [0.1, 0.2, 0.3];
    const fx = effect(stage(fragmentTemplate(), { tint: () => v }, freshId()));
    const a = fx.compile({ target: "wgsl" });
    const b = fx.compile({ target: "wgsl" });
    expect(a).toBe(b);
  });

  it("hole-getter values ARE part of the key — changed holes ⇒ fresh compile", () => {
    // Holes are inlined as constants, so mutating what a getter returns
    // must miss the cache and re-emit; restoring the original values
    // hits the original cache entry again.
    const ref = { tint: [0.1, 0.2, 0.3] as [number, number, number] };
    const fx = effect(stage(fragmentTemplate(), { tint: () => ref.tint }, freshId()));
    const a = fx.compile({ target: "wgsl" });
    ref.tint = [0.4, 0.5, 0.6];
    const b = fx.compile({ target: "wgsl" });
    expect(b).not.toBe(a);
    expect(b.stages[0]!.source).toMatch(/0\.4/);
    ref.tint = [0.1, 0.2, 0.3];
    const c = fx.compile({ target: "wgsl" });
    expect(c).toBe(a);
  });

  it("different target → distinct cache entries", () => {
    const fx = effect(stage(fragmentTemplate(), { tint: () => [0.1, 0.2, 0.3] }, freshId()));
    const w = fx.compile({ target: "wgsl" });
    const g = fx.compile({ target: "glsl" });
    expect(w).not.toBe(g);
    expect(w.target).toBe("wgsl");
    expect(g.target).toBe("glsl");
  });

  it("`skipOptimisations` and the optimised path don't collide", () => {
    const fx = effect(stage(fragmentTemplate(), { tint: () => [0.1, 0.2, 0.3] }, freshId()));
    const o = fx.compile({ target: "wgsl" });
    const r = fx.compile({ target: "wgsl", skipOptimisations: true });
    expect(o).not.toBe(r);
  });
});
