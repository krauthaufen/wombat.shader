// Sampler captures: a closure-captured `Sampler2D` value lowers to
// a Sampler ValueDef + uniform-scope read, with the runtime handle
// preserved on the Effect's binding map.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";
import { effect, stage } from "@aardworx/wombat.shader-runtime";
import type { Module } from "@aardworx/wombat.shader-ir";

let workdir: string;
let resolver: TypeResolver;

function extractTemplate(code: string): Module {
  const idx = code.indexOf("__wombat_stage(");
  let i = code.indexOf("{", idx);
  const start = i;
  let depth = 0;
  for (; i < code.length; i++) {
    if (code[i] === "{") depth++;
    else if (code[i] === "}") { depth--; if (depth === 0) { i++; break; } }
  }
  return JSON.parse(code.slice(start, i)) as Module;
}

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-sampler-test-"));
  fs.writeFileSync(path.join(workdir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022", module: "ESNext", moduleResolution: "Bundler",
      strict: false, esModuleInterop: true, skipLibCheck: true, types: [],
    },
    include: ["**/*.ts", "**/*.d.ts"],
  }));
  fs.writeFileSync(path.join(workdir, "types.d.ts"), `
    declare class V2f { x: number; y: number; constructor(x: number, y: number); }
    declare class V3f { x: number; y: number; z: number; constructor(x: number, y: number, z: number); }
    declare class V4f { x: number; y: number; z: number; w: number; constructor(x: number, y: number, z: number, w: number); }
    declare class Sampler2D       { __brand: "Sampler2D"; }
    declare class Sampler3D       { __brand: "Sampler3D"; }
    declare class SamplerCube     { __brand: "SamplerCube"; }
    declare class Sampler2DArray  { __brand: "Sampler2DArray"; }
    declare class Sampler2DShadow { __brand: "Sampler2DShadow"; }
    declare class ISampler2D      { __brand: "ISampler2D"; }
    declare class USampler2D      { __brand: "USampler2D"; }
    declare class Sampler2DMS     { __brand: "Sampler2DMS"; }
    declare class V2i             { __brand: "V2i"; }
    declare function texelFetch(s: Sampler2DMS, ij: V2i, sample: number): V4f;
    declare function texture(s: Sampler2D, uv: V2f): V4f;
    declare function texture(s: Sampler3D, uv: V3f): V4f;
    declare function texture(s: SamplerCube, uv: V3f): V4f;
    declare function texture(s: Sampler2DArray, uv: V3f): V4f;
    declare function texture(s: Sampler2DShadow, uv: V3f): number;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("sampler captures", () => {
  it("`Sampler2D` capture emits a Sampler ValueDef, not a constant", () => {
    const file = path.join(workdir, "app.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tex: Sampler2D = ({} as any);  // pretend runtime handle
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: texture(tex, input.v_uv),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const sDef = tmpl.values.find((v) => v.kind === "Sampler");
    expect(sDef).toBeTruthy();
    if (sDef && sDef.kind === "Sampler") {
      expect(sDef.name).toBe("tex");
      expect(sDef.type.kind).toBe("Sampler");
    }
    // No closure scope ref — sampler reads through Uniform scope.
    const json = JSON.stringify(tmpl);
    expect(json).not.toContain('"scope":"Closure"');
    // The plugin emits the sampler getter as a binding (4th arg).
    expect(r!.code).toContain("tex: () => tex");
  });

  it("WGSL emit splits the sampler/texture pair via legaliseTypes", () => {
    const file = path.join(workdir, "wgsl.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tex: Sampler2D = ({} as any);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: texture(tex, input.v_uv),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fakeHandle = { __brand: "Sampler2D" };
    const fx = effect(stage(tmpl, {}, undefined, { tex: () => fakeHandle }));
    const compiled = fx.compile({ target: "wgsl" });
    expect(compiled.avalBindings.tex).toBeTypeOf("function");
    expect(compiled.avalBindings.tex!()).toBe(fakeHandle);
    // WGSL emits a separate texture_2d and sampler binding.
    const wgsl = compiled.stages[0]!.source;
    expect(wgsl).toMatch(/texture_2d/);
    expect(wgsl).toMatch(/var\s+tex:\s*sampler/);
  });

  it.each([
    ["Sampler3D",       /var\s+\w+:\s*sampler/, /texture_3d/, /sampler3D/],
    ["SamplerCube",     /var\s+\w+:\s*sampler/, /texture_cube/, /samplerCube/],
    ["Sampler2DArray",  /var\s+\w+:\s*sampler/, /texture_2d_array/, /sampler2DArray/],
    ["Sampler2DShadow", /var\s+\w+:\s*sampler_comparison/, /texture_depth_2d/, /sampler2DShadow/],
    ["ISampler2D",      /var\s+\w+:\s*sampler/, /texture_2d<i32>/, /isampler2D/],
    ["USampler2D",      /var\s+\w+:\s*sampler/, /texture_2d<u32>/, /usampler2D/],
  ])("captures of `%s` lower to the right binding shape on both targets",
    (typeName, samplerLine, textureLine, glslLine) => {
      const file = path.join(workdir, `${typeName}.ts`);
      // Use V3f for 3D/Cube/Array/Shadow uvs; V2f for 2D variants. Pick
      // a uv that the shipped texture overload accepts.
      const uvType = typeName === "Sampler3D" || typeName === "SamplerCube"
        || typeName === "Sampler2DArray" || typeName === "Sampler2DShadow"
        ? "V3f" : "V2f";
      const uvCtor = uvType === "V3f"
        ? `new V3f(input.v_uv.x, input.v_uv.y, 0.0)`
        : `input.v_uv`;
      // Sampler2DShadow returns scalar, others return V4f.
      const wrapped = typeName === "Sampler2DShadow"
        ? `new V4f(texture(t, ${uvCtor}), 0.0, 0.0, 1.0)`
        : `texture(t, ${uvCtor})`;
      const src = `
        import { fragment } from "@aardworx/wombat.shader-runtime";
        const t: ${typeName} = ({} as any);
        const fs = fragment((input: { v_uv: V2f }) => ({
          outColor: ${wrapped},
        }));
      `;
      fs.writeFileSync(file, src);
      const r = transformInlineShaders(src, file, resolver)!;
      const tmpl = extractTemplate(r.code);
      const fx = effect(stage(tmpl, {}, undefined, { t: () => ({}) }));
      const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
      expect(wgsl).toMatch(samplerLine);
      expect(wgsl).toMatch(textureLine);
      // GLSL: combined sampler with the right brand name.
      const glsl = fx.compile({ target: "glsl" }).stages[0]!.source;
      expect(glsl).toMatch(glslLine);
    });

  it("multisampled `Sampler2DMS` emits texture_multisampled_2d in WGSL with no companion sampler", () => {
    const file = path.join(workdir, "ms.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const ms: Sampler2DMS = ({} as any);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: texelFetch(ms, new V2i(0, 0), 0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { ms: () => ({}) }));
    const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toMatch(/texture_multisampled_2d<f32>/);
    // No sampler binding for MS — only the multisampled texture itself.
    expect(wgsl).not.toMatch(/var\s+ms_view/);
    // textureLoad takes (texture, coord, sample) — no sampler arg.
    expect(wgsl).toMatch(/textureLoad\(ms,/);
  });

  it("GLSL target rejects multisampled samplers with a clear error", () => {
    const file = path.join(workdir, "msfail.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const ms: Sampler2DMS = ({} as any);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: texelFetch(ms, new V2i(0, 0), 0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { ms: () => ({}) }));
    expect(() => fx.compile({ target: "glsl" })).toThrow(/multisampled samplers .* are not supported/);
  });

  it("GLSL emit uses a combined sampler", () => {
    const file = path.join(workdir, "glsl.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tex: Sampler2D = ({} as any);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: texture(tex, input.v_uv),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { tex: () => ({}) }));
    const glsl = fx.compile({ target: "glsl" }).stages[0]!.source;
    expect(glsl).toContain("uniform sampler2D tex");
    expect(glsl).toMatch(/texture\(tex/);
  });
});
