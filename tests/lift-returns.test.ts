// Tests for the liftReturns pass + the full compile pipeline.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";

describe("liftReturns + compile pipeline", () => {
  const source = `
    function fsMain(input: { v_color: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
    }
  `;

  it("compiles to GLSL with WriteOutput-shaped body", () => {
    const r = compileShaderSource(source, [{
      name: "fsMain",
      stage: "fragment",
      outputs: [{
        name: "outColor",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }],
    }], { target: "glsl" });
    const stage = r.stages[0]!;
    expect(stage.source).toContain("layout(location = 0) out vec4 outColor;");
    expect(stage.source).toContain("outColor = vec4(");
  });

  it("compiles the same source to WGSL", () => {
    const r = compileShaderSource(source, [{
      name: "fsMain",
      stage: "fragment",
      outputs: [{
        name: "outColor",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }],
    }], { target: "wgsl" });
    const stage = r.stages[0]!;
    expect(stage.source).toContain("@fragment");
    expect(stage.source).toContain("out.outColor = vec4<f32>(");
  });

  it("bare V4f return in fragment with one declared output lifts to WriteOutput", () => {
    const src = `
      function fsMain(input: { v_color: V3f }): V4f {
        return new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0);
      }
    `;
    const r = compileShaderSource(src, [{
      name: "fsMain",
      stage: "fragment",
      outputs: [{
        name: "outColor",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }],
    }], { target: "wgsl" });
    const stage = r.stages[0]!;
    // Should be a normal WriteOutput-shaped body, not a `return vec4<f32>(...)`.
    expect(stage.source).toContain("out.outColor = vec4<f32>(");
    expect(stage.source).not.toMatch(/return\s+vec4<f32>/);
  });

  it("bare V4f return in vertex with single position output lifts to WriteOutput", () => {
    const src = `
      function vsMain(input: { a_position: V3f }): V4f {
        return new V4f(input.a_position.x, input.a_position.y, input.a_position.z, 1.0);
      }
    `;
    const r = compileShaderSource(src, [{
      name: "vsMain",
      stage: "vertex",
      outputs: [{
        name: "gl_Position",
        type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
        semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }],
      }],
    }], { target: "wgsl" });
    const stage = r.stages[0]!;
    expect(stage.source).toContain("out.gl_Position = vec4<f32>(");
    expect(stage.source).not.toMatch(/return\s+vec4<f32>/);
  });

  it("hello-triangle vertex+fragment compiles end-to-end (GLSL)", () => {
    const src = `
      function vsMain(input: { a_position: V3f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
        return {
          gl_Position: new V4f(input.a_position.x, input.a_position.y, input.a_position.z, 1.0),
          v_color: input.a_color,
        };
      }
      function fsMain(input: { v_color: V3f }): { outColor: V4f } {
        return { outColor: new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
      }
    `;
    const r = compileShaderSource(src, [
      {
        name: "vsMain", stage: "vertex",
        outputs: [
          {
            name: "gl_Position",
            type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
            semantic: "Position",
            decorations: [{ kind: "Builtin", value: "position" }],
          },
          {
            name: "v_color",
            type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 },
            semantic: "Color",
            decorations: [{ kind: "Location", value: 0 }],
          },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        outputs: [{
          name: "outColor",
          type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
          semantic: "Color",
          decorations: [{ kind: "Location", value: 0 }],
        }],
      },
    ], { target: "glsl" });
    expect(r.stages.length).toBe(2);
    const vs = r.stages.find((s) => s.stage === "vertex")!.source;
    const fs = r.stages.find((s) => s.stage === "fragment")!.source;
    expect(vs).toContain("gl_Position = vec4(");
    expect(vs).toContain("v_color = a_color;");
    expect(fs).toContain("outColor = vec4(");
  });

  describe("imperative early-return", () => {
    const earlyReturnSource = `
      function fsMain(input: { v_color: V4f }): { outColor: V4f } {
        if (input.v_color.w < 0.5) {
          return { outColor: new V4f(0.0, 0.0, 0.0, 1.0) };
        }
        return { outColor: input.v_color };
      }
    `;
    const fsEntry = {
      name: "fsMain", stage: "fragment" as const,
      outputs: [{
        name: "outColor",
        type: { kind: "Vector" as const, element: { kind: "Float" as const, width: 32 }, dim: 4 },
        semantic: "Color",
        decorations: [{ kind: "Location" as const, value: 0 }],
      }],
    };

    it("WGSL: early `return { … }` inside `if` actually exits the entry", () => {
      const r = compileShaderSource(earlyReturnSource, [fsEntry], { target: "wgsl" });
      const fs = r.stages[0]!.source;
      // Before the fix, the early-exit silently fell through to the
      // tail write, so the conditional black colour was always
      // overwritten. Now we expect a `return out;` inside the if.
      const ifIdx = fs.indexOf("if (");
      const endIfIdx = fs.indexOf("}", ifIdx);
      expect(ifIdx).toBeGreaterThanOrEqual(0);
      const ifBlock = fs.slice(ifIdx, endIfIdx);
      expect(ifBlock).toContain("return out;");
    });

    it("GLSL: same early-return path emits a bare `return;` inside the branch", () => {
      const r = compileShaderSource(earlyReturnSource, [fsEntry], { target: "glsl" });
      const fs = r.stages[0]!.source;
      const ifIdx = fs.indexOf("if (");
      const endIfIdx = fs.indexOf("}", ifIdx);
      const ifBlock = fs.slice(ifIdx, endIfIdx);
      expect(ifBlock).toContain("return;");
    });

    it("tail return at the natural end emits NO duplicate `return out;`", () => {
      // The tail write doesn't get a Return — emitEntryFunction's
      // synth handles it. We assert the entry function body has
      // exactly one `return out;` (synthesized at the end), not two
      // adjacent ones from the lifted tail + synth.
      const tailOnlySource = `
        function fsMain(input: { v_color: V4f }): { outColor: V4f } {
          return { outColor: input.v_color };
        }
      `;
      const r = compileShaderSource(tailOnlySource, [fsEntry], { target: "wgsl" });
      const fs = r.stages[0]!.source;
      const matches = fs.match(/return out;/g) ?? [];
      expect(matches.length).toBe(1);
    });
  });
});
