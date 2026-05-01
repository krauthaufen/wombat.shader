// Sanity: the V*f / M*f types re-exported via
// `@aardworx/wombat.shader-types` are real wombat.base runtime
// classes. Constructing them on the CPU works; the same identifier
// resolves to the right IR type in shader code.

import { describe, expect, it } from "vitest";
import { V3f, V4f, M44f } from "@aardworx/wombat.shader/types";
import { compileShaderSource } from "@aardworx/wombat.shader";

describe("wombat.base reuse", () => {
  it("V3f / V4f / M44f are real runtime classes", () => {
    const v = new V3f(1, 2, 3);
    expect(v.x).toBe(1);
    expect(v.y).toBe(2);
    expect(v.z).toBe(3);
    const w = new V4f(0.1, 0.2, 0.3, 1.0);
    expect(w.toArray()).toEqual([0.1, 0.2, 0.3, 1.0].map((n) => Math.fround(n)));
    const m = M44f.identity;
    expect(typeof m).toBe("object");
  });

  it("V3f.add returns a new V3f and the IR translates the same call form", () => {
    const a = new V3f(1, 0, 0);
    const b = new V3f(0, 1, 0);
    const c = a.add(b);
    expect(c.toArray().slice(0, 3)).toEqual([1, 1, 0]);

    // Same .add(...) call form lowered by the shader frontend.
    const compiled = compileShaderSource(
      `
        function fsMain(input: { v_color: V3f }): V4f {
          const tinted = input.v_color.add(input.v_color);
          return new V4f(tinted.x, tinted.y, tinted.z, 1.0);
        }
      `,
      [{ name: "fsMain", stage: "fragment", outputs: [{
        name: "outColor",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }] }],
      { target: "wgsl" },
    );
    const wgsl = compiled.stages[0]!.source;
    expect(wgsl).toMatch(/in\.v_color\s*\+\s*in\.v_color/);
  });
});
