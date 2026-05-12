// Effect compile cache. Cache key = effect.id + target (+ skipOpt +
// fragment-output-layout). Same key on a second `compile()` returns
// the *same* `CompiledEffect` reference (no re-emit). The sampled
// closure-hole values are NOT part of the key — for a given effect id
// the holes are assumed invariant (they're module-level-const captures
// in every realistic effect), so two compiles of the same id share a
// result regardless of what the hole getters return between them.

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

  it("hole-getter values are NOT part of the key — same id ⇒ same result", () => {
    // The compile cache is keyed on the effect id, not the sampled
    // holes; once compiled, mutating what a hole getter returns does
    // not change the cached `CompiledEffect`.
    const ref = { tint: [0.1, 0.2, 0.3] as [number, number, number] };
    const fx = effect(stage(fragmentTemplate(), { tint: () => ref.tint }, freshId()));
    const a = fx.compile({ target: "wgsl" });
    ref.tint = [0.4, 0.5, 0.6];
    const b = fx.compile({ target: "wgsl" });
    expect(b).toBe(a);
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
