// Vite-plugin inline-shader transform: `vertex/fragment/compute(arrow)`
// call sites are rewritten to `__wombat_stage(template, holes)`.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";

describe("transformInlineShaders", () => {
  it("returns null for files with no markers", () => {
    const r = transformInlineShaders(
      `console.log("hi");`,
      "/x/app.ts",
    );
    expect(r).toBeNull();
  });

  it("returns null for marker-shaped strings without runtime import", () => {
    const r = transformInlineShaders(
      `function vertex() { /* user-defined helper, not the marker */ }`,
      "/x/app.ts",
    );
    expect(r).toBeNull();
  });

  it("rewrites a fragment marker call with no captures", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.code).toContain("__wombat_stage(");
    expect(r!.code).toContain('import { stage as __wombat_stage }');
    // No captures → empty holes object.
    expect(r!.code).toMatch(/__wombat_stage\([\s\S]+, \{\}, "[0-9a-f]{16}", \{\}\)/);
    // Original `fragment(...)` call removed.
    expect(r!.code).not.toMatch(/\bfragment\s*\(/);
  });

  it("rewrites a fragment marker call with a closure capture", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tint: V3f = new V3f(0.5, 0.6, 0.7);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, input.v_uv.y, tint.x, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.code).toContain("__wombat_stage(");
    expect(r!.code).toContain("tint: () => tint");
    // Template's IR JSON should record the closure scope reference.
    expect(r!.code).toContain('"scope":"Closure"');
    expect(r!.code).toContain('"name":"tint"');
  });

  it("infers the closure type from a `new V3f(...)` initializer", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tint = new V3f(1, 0, 0);
      const fs = fragment(() => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.code).toContain("tint: () => tint");
    // V3f resolves to a Vector<f32, 3>.
    expect(r!.code).toContain('"kind":"Vector"');
  });

  it("errors clearly on a closure capture with no resolvable type", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tint = whatever;
      const fs = fragment(() => ({ outColor: new V4f(tint, tint, tint, 1.0) }));
    `;
    expect(() => transformInlineShaders(src, "/x/app.ts"))
      .toThrow(/cannot determine type of "tint"/);
  });

  it("accepts a top-level function declaration by name", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      function fsMain(input: { v_uv: V2f }): { outColor: V4f } {
        return { outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0) };
      }
      const fs = fragment(fsMain);
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.code).toContain("__wombat_stage(");
    // `fragment(fsMain)` is replaced; the named function declaration
    // is left in place (callers may still reference it for tests).
    expect(r!.code).toContain("function fsMain");
    expect(r!.code).toContain('"stage":"fragment"');
  });

  it("accepts a top-level `const fn = (...) => ...` by name", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader-runtime";
      const vsMain = (v: { a_position: V3f }) => ({
        gl_Position: new V4f(v.a_position.x, v.a_position.y, v.a_position.z, 1.0),
      });
      const vs = vertex(vsMain);
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.code).toContain('"stage":"vertex"');
    expect(r!.code).toContain('"value":"position"');
  });

  it("errors clearly if the named function isn't found in this file", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const fs = fragment(missingFn);
    `;
    expect(() => transformInlineShaders(src, "/x/app.ts"))
      .toThrow(/could not find a top-level function declaration.*missingFn/);
  });

  it("ignores intrinsic / shipped-type identifiers (not captures)", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(sin(input.v_uv.x), 0.0, 0.0, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    // No closure references — `vec4` and `sin` are intrinsics, not captures.
    expect(r!.code).toMatch(/__wombat_stage\([\s\S]+, \{\}, "[0-9a-f]{16}", \{\}\)/);
  });
});
