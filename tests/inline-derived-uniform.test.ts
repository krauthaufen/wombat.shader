// The plugin recognises `derivedUniform((u) => …)` (wombat.rendering's derived-uniform
// marker) and appends a leaf-type-hint map gathered from the file's `UniformScope`
// augmentation, so `u.<Name>` reads its real WGSL type. Resource leaves are rejected.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";

const AUG = `
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Tint: V4f;
    readonly WorldUp: V3f;
    readonly Frame: u32;
    readonly Albedo: Sampler2D;
  }
}
`;

function tx(body: string, aug = AUG): string {
  const src = `import { effect } from "@aardworx/wombat.shader";\nimport { derivedUniform } from "@aardworx/wombat.rendering/runtime";\n${aug}\n${body}\n`;
  const r = transformInlineShaders(src, "/tmp/du-test.ts");
  return r ? r.code : src;
}

describe("inline plugin: derivedUniform marker", () => {
  it("appends a leaf-type-hint map for uniform names declared in UniformScope", () => {
    const out = tx(`const r = derivedUniform((u) => u.Tint.swizzle("xyz").add(u.WorldUp));`);
    expect(out).toContain(`derivedUniform((u) => u.Tint.swizzle("xyz").add(u.WorldUp), {`);
    expect(out).toMatch(/"Tint":"vec4"/);
    expect(out).toMatch(/"WorldUp":"vec3"/);
  });

  it("leaves names not in UniformScope alone (runtime defaults them to mat4)", () => {
    const out = tx(`const r = derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo));`);
    // ViewTrafo/ModelTrafo aren't in the augmentation → no hint map appended at all.
    expect(out).not.toContain(`derivedUniform((u) => u.ViewTrafo.mul(u.ModelTrafo), {`);
  });

  it("recognises scalar uniforms", () => {
    const out = tx(`const r = derivedUniform((u) => u.Frame);`);
    expect(out).toMatch(/derivedUniform\(\(u\) => u\.Frame, \{"Frame":"u32"\}\)/);
  });

  it("rejects a resource leaf with a clear diagnostic", () => {
    expect(() => tx(`const r = derivedUniform((u) => u.Albedo);`)).toThrow(/resources|texture\/sampler\/storage/i);
  });

  it("doesn't touch a derivedUniform call that already has a 2nd arg", () => {
    const out = tx(`const r = derivedUniform((u) => u.Tint, { Tint: "vec4" });`);
    // single-arg only — a 2-arg call is left as-is.
    expect(out).toContain(`derivedUniform((u) => u.Tint, { Tint: "vec4" })`);
  });
});
