// End-to-end test for the frontend: TypeScript source → IR Module
// → GLSL / WGSL emit.

import { describe, expect, it } from "vitest";
import { parseShader } from "@aardworx/wombat.shader/frontend";
import { emitGlsl } from "@aardworx/wombat.shader/glsl";
import { emitWgsl } from "@aardworx/wombat.shader/wgsl";

describe("frontend — minimal fragment shader", () => {
  it("translates a simple fragment to IR + emits GLSL", () => {
    const source = `
      function fsMain(input: { v_color: V3f }): V4f {
        const c = new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0);
        return c;
      }
    `;
    // We can't actually use the shipped types in this test (they're
    // declarations); just run the parser, which is name-based.
    const m = parseShader({
      source,
      entries: [{
        name: "fsMain",
        stage: "fragment",
        outputs: [
          {
            name: "outColor",
            type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
            semantic: "Color",
            decorations: [{ kind: "Location", value: 0 }],
          },
        ],
      }],
    });
    expect(m.values.length).toBeGreaterThan(0);
    const entry = m.values.find((v) => v.kind === "Entry");
    expect(entry?.kind).toBe("Entry");
    if (entry?.kind === "Entry") {
      // The body should reference v_color via Field/Swizzle, then a vec4 NewVector.
      const json = JSON.stringify(entry.entry.body);
      expect(json).toContain("VecSwizzle");
      expect(json).toContain("NewVector");
    }
  });

  it("end-to-end: TS source → GLSL", () => {
    const source = `
      function fsMain(input: { v_uv: V2f }): V4f {
        const tinted = new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0);
        return tinted;
      }
    `;
    const m = parseShader({
      source,
      entries: [{
        name: "fsMain", stage: "fragment",
        outputs: [{
          name: "outColor",
          type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
          semantic: "Color",
          decorations: [{ kind: "Location", value: 0 }],
        }],
      }],
    });
    // The frontend leaves the function body returning `tinted`. To
    // actually produce a fragment that writes the output, callers
    // would post-process with a "lift return → WriteOutput" pass; for
    // this v0.1 we just verify it emits without crashing.
    const glsl = emitGlsl(m).source;
    expect(glsl).toContain("vec4(");
    expect(glsl).toContain("v_uv");
    const wgsl = emitWgsl(m).source;
    expect(wgsl).toContain("vec4<f32>(");
  });
});

describe("frontend — control flow", () => {
  it("translates if/else", () => {
    const source = `
      function fsMain(input: { u_t: number }): V4f {
        if (input.u_t > 0.5) {
          return new V4f(1.0, 0.0, 0.0, 1.0);
        } else {
          return new V4f(0.0, 1.0, 0.0, 1.0);
        }
      }
    `;
    const m = parseShader({
      source,
      entries: [{ name: "fsMain", stage: "fragment" }],
    });
    const entry = m.values.find((v) => v.kind === "Entry");
    if (entry?.kind === "Entry") {
      expect(entry.entry.body.kind).toBe("If");
    }
  });

  it("translates a for loop with mutation", () => {
    const source = `
      function fsMain(input: { u_n: number }): V4f {
        let acc = 0.0;
        for (let i = 0; i < 10; i++) {
          acc = acc + 1.0;
        }
        return new V4f(acc, acc, acc, 1.0);
      }
    `;
    const m = parseShader({
      source,
      entries: [{ name: "fsMain", stage: "fragment" }],
    });
    // Expect the body to contain a For statement.
    const entry = m.values.find((v) => v.kind === "Entry");
    expect(entry?.kind).toBe("Entry");
    if (entry?.kind === "Entry") {
      const json = JSON.stringify(entry.entry.body);
      expect(json).toContain("\"For\"");
    }
  });
});

describe("frontend — method calls map to IR ops", () => {
  it("a.add(b), a.dot(b), a.normalize() translate", () => {
    const source = `
      function fsMain(input: { a: V3f; b: V3f }): V4f {
        const sum = input.a.add(input.b);
        const d = input.a.dot(input.b);
        const n = input.a.normalize();
        return new V4f(d, d, d, 1.0);
      }
    `;
    const m = parseShader({
      source,
      entries: [{ name: "fsMain", stage: "fragment" }],
    });
    const entry = m.values.find((v) => v.kind === "Entry");
    if (entry?.kind === "Entry") {
      const json = JSON.stringify(entry.entry.body);
      expect(json).toContain("\"Add\"");
      expect(json).toContain("\"Dot\"");
      expect(json).toContain("\"normalize\""); // CallIntrinsic
    }
  });

  it("matrix * vector dispatches to MulMatVec", () => {
    const source = `
      function vsMain(input: { mvp: M44f; pos: V4f }): V4f {
        return input.mvp.mul(input.pos);
      }
    `;
    const m = parseShader({
      source,
      entries: [{ name: "vsMain", stage: "vertex" }],
    });
    const entry = m.values.find((v) => v.kind === "Entry");
    if (entry?.kind === "Entry") {
      const json = JSON.stringify(entry.entry.body);
      expect(json).toContain("MulMatVec");
    }
  });
});

describe("frontend — intrinsics", () => {
  it("sin / mix / texture all become CallIntrinsic", () => {
    const source = `
      function fsMain(input: { v_uv: V2f; tex: Sampler2D; t: number }): V4f {
        const wave = sin(input.t);
        const sampled = texture(input.tex, input.v_uv);
        const tinted = mix(sampled, new V4f(wave, wave, wave, 1.0), 0.5);
        return tinted;
      }
    `;
    const m = parseShader({
      source,
      entries: [{ name: "fsMain", stage: "fragment" }],
    });
    const entry = m.values.find((v) => v.kind === "Entry");
    if (entry?.kind === "Entry") {
      const json = JSON.stringify(entry.entry.body);
      expect(json).toContain("\"sin\"");
      expect(json).toContain("\"texture\"");
      expect(json).toContain("\"mix\"");
    }
  });
});
