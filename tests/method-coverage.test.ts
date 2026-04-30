// wombat.base method names translate via the frontend method
// dispatcher. Coverage for the post-Phase-9 surface so users can
// import V*/M* directly from wombat.base and use its full method API
// in shader bodies.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader-runtime";

function compileFragment(source: string): { glsl: string; wgsl: string } {
  const out = ["glsl", "wgsl"].map((target) =>
    compileShaderSource(
      source,
      [{ name: "fsMain", stage: "fragment", outputs: [{
        name: "outColor",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }]}],
      { target: target as "glsl" | "wgsl" },
    ).stages[0]!.source,
  );
  return { glsl: out[0]!, wgsl: out[1]! };
}

describe("vector method coverage (wombat.base API)", () => {
  it("`v.lengthSquared()` lowers to dot(v, v)", () => {
    const { glsl, wgsl } = compileFragment(`
      function fsMain(input: { v_uv: V2f }): V4f {
        const ls = input.v_uv.lengthSquared();
        return new V4f(ls, 0, 0, 1);
      }
    `);
    // Both targets have native dot.
    expect(glsl).toMatch(/dot\(/);
    expect(wgsl).toMatch(/dot\(/);
  });

  it("`v.distance(other)` and `v.distanceSquared(other)` use distance intrinsic", () => {
    const { glsl, wgsl } = compileFragment(`
      function fsMain(input: { v_uv: V2f }): V4f {
        const a = input.v_uv;
        const b = new V2f(0.5, 0.5);
        const d = a.distance(b);
        const d2 = a.distanceSquared(b);
        return new V4f(d + d2, 0, 0, 1);
      }
    `);
    for (const code of [glsl, wgsl]) {
      expect(code).toMatch(/distance\(/);
    }
  });

  it("element-wise method forms route to intrinsics: abs, floor, fract, sign", () => {
    const { glsl, wgsl } = compileFragment(`
      function fsMain(input: { v_uv: V2f }): V4f {
        const a = input.v_uv.abs();
        const b = a.floor();
        const c = b.fract();
        const d = c.sign();
        return new V4f(d.x, d.y, 0, 1);
      }
    `);
    for (const code of [glsl, wgsl]) {
      expect(code).toContain("abs(");
      expect(code).toContain("floor(");
      expect(code).toContain("fract(");
      expect(code).toContain("sign(");
    }
  });

  it("min/max/clamp/lerp method forms route correctly", () => {
    const { glsl, wgsl } = compileFragment(`
      function fsMain(input: { v_uv: V2f }): V4f {
        const lo = new V2f(0, 0);
        const hi = new V2f(1, 1);
        const c = input.v_uv.clamp(lo, hi);
        const m = c.min(hi);
        const x = m.max(lo);
        const l = x.lerp(hi, 0.5);
        return new V4f(l.x, l.y, 0, 1);
      }
    `);
    for (const code of [glsl, wgsl]) {
      expect(code).toContain("clamp(");
      expect(code).toContain("min(");
      expect(code).toContain("max(");
      expect(code).toContain("mix(");  // lerp → mix in IR
    }
  });
});
