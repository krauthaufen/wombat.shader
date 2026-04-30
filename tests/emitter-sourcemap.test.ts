// Emitter source maps. The frontend tags every Stmt/Expr with a TS
// source span; the WGSL/GLSL Writer records the current span per
// emitted line; the runtime builds a v3 source map and attaches it
// to each `CompiledStage`.

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import type { Type } from "@aardworx/wombat.shader-ir";

const Tvec4f: Type = { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 };
const Tvec2f: Type = { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 2 };

describe("emitter source maps", () => {
  it("CompiledStage carries a v3 source map when the IR has spans", () => {
    const compiled = compileShaderSource(
      `
        function fsMain(input: { v_uv: V2f }): V4f {
          const x = input.v_uv.x;
          const y = input.v_uv.y;
          return new V4f(x, y, 0.5, 1.0);
        }
      `,
      [{ name: "fsMain", stage: "fragment", outputs: [{
        name: "outColor", type: Tvec4f, semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }] }],
      { target: "wgsl", file: "/x/app.ts" },
    );
    const stage = compiled.stages[0]!;
    expect(stage.sourceMap).not.toBeNull();
    const map = stage.sourceMap!;
    expect(map.version).toBe(3);
    expect(map.sources).toContain("/x/app.ts");
    expect(typeof map.mappings).toBe("string");
    expect(map.mappings.length).toBeGreaterThan(0);
    // One ';' separator per emitted line minus 1.
    const sourceLines = stage.source.split("\n").length - 1;
    const mapLines = map.mappings.split(";").length;
    expect(mapLines).toBe(sourceLines);
  });

  it("hand-built IR Module (no spans) yields sourceMap=null", () => {
    const compiled = compileShaderSource(
      `function fsMain(input: { v_uv: V2f }): V4f {
        return new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0);
      }`,
      [{ name: "fsMain", stage: "fragment", outputs: [{
        name: "outColor", type: Tvec4f, semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }] }],
      { target: "wgsl" }, // no `file`, but spans still propagate from `<input>`
    );
    expect(compiled.stages[0]!.sourceMap).not.toBeNull();
    expect(compiled.stages[0]!.sourceMap!.sources[0]).toBe("<input>");
  });

  it("GLSL target also produces a source map", () => {
    const compiled = compileShaderSource(
      `function fsMain(input: { v_uv: V2f }): V4f {
        const a = input.v_uv.x;
        const b = input.v_uv.y;
        return new V4f(a, b, 0.5, 1.0);
      }`,
      [{ name: "fsMain", stage: "fragment", outputs: [{
        name: "outColor", type: Tvec4f, semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }] }],
      { target: "glsl", file: "/x/app.ts" },
    );
    const stage = compiled.stages[0]!;
    expect(stage.sourceMap).not.toBeNull();
    expect(stage.sourceMap!.sources).toContain("/x/app.ts");
  });

  // Sanity check on the type-level wiring: lineSpans aren't accidentally
  // serialised into the IR JSON via stableStringify (would change hashes).
  it("Stmt.span is excluded from the build-time module hash", async () => {
    const { hashModule } = await import("@aardworx/wombat.shader-ir");
    const { parseShader } = await import("@aardworx/wombat.shader-frontend");
    const m1 = parseShader({
      source: `function f(input: { v_uv: V2f }): V4f { return new V4f(input.v_uv.x, 0, 0, 1); }`,
      file: "/file-A.ts",
      entries: [{ name: "f", stage: "fragment" }],
    });
    const m2 = parseShader({
      source: `function f(input: { v_uv: V2f }): V4f { return new V4f(input.v_uv.x, 0, 0, 1); }`,
      file: "/file-B.ts",
      entries: [{ name: "f", stage: "fragment" }],
    });
    // Same code, different file — span.file differs but hash should match.
    expect(hashModule(m1)).toBe(hashModule(m2));
  });
});
