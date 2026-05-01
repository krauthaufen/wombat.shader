// Plugin recognises ambient `declare const u: { … }` declarations as
// uniform bindings. Uses the TypeResolver (full TS LanguageService) so
// uniforms can live in any file the project's tsconfig knows about —
// including .d.ts files in node_modules.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";
import type { Module } from "@aardworx/wombat.shader/ir";

let workdir: string;
let resolver: TypeResolver;

function extractTemplate(code: string): Module {
  const idx = code.indexOf("__wombat_stage(");
  let i = code.indexOf("{", idx);
  const start = i;
  let depth = 0;
  for (; i < code.length; i++) {
    if (code[i] === "{") depth++;
    else if (code[i] === "}") {
      depth--;
      if (depth === 0) { i++; break; }
    }
  }
  return JSON.parse(code.slice(start, i)) as Module;
}

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-inline-test-"));
  // Minimal tsconfig pointing at our shipped types so the resolver
  // can resolve V*f / M*f / Sampler* by symbol name.
  fs.writeFileSync(path.join(workdir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022",
      module: "ESNext",
      moduleResolution: "Bundler",
      strict: false,
      esModuleInterop: true,
      skipLibCheck: true,
      types: [],
    },
    include: ["**/*.ts", "**/*.d.ts"],
  }));
  // Shipped types stub so V3f / V4f resolve.
  fs.writeFileSync(path.join(workdir, "types.d.ts"), `
    declare class V2f { x: number; y: number; constructor(x: number, y: number); }
    declare class V3f { x: number; y: number; z: number; constructor(x: number, y: number, z: number); }
    declare class V4f { x: number; y: number; z: number; w: number; constructor(x: number, y: number, z: number, w: number); }
    declare class M44f { constructor(); }
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("inline shader: ambient uniforms via TS type checker", () => {
  it("recognises `declare const u: { … }` in the same file", () => {
    const file = path.join(workdir, "app.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      declare const u: { readonly tint: V3f; readonly time: number };
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(u.tint.x, u.tint.y, u.tint.z, u.time),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    // Uniform decl present.
    const uniformDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uniformDef).toBeTruthy();
    if (uniformDef && uniformDef.kind === "Uniform") {
      const fields = uniformDef.uniforms.map((u) => ({ name: u.name, buffer: u.buffer }));
      expect(fields).toContainEqual({ name: "tint", buffer: "u" });
      expect(fields).toContainEqual({ name: "time", buffer: "u" });
    }
    // Body reads as uniforms (ReadInput Uniform), no closure scope.
    const json = JSON.stringify(tmpl);
    expect(json).not.toContain('"scope":"Closure"');
    expect(json).toContain('"scope":"Uniform"');
    // No closure getters either.
    expect(r!.code).toMatch(/, \{\}, "[0-9a-f]{16}", \{\}\)/);
  });

  it("recognises a loose `declare const time: number` ambient uniform", () => {
    const file = path.join(workdir, "loose.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      declare const time: number;
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(time, time, time, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const uniformDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uniformDef).toBeTruthy();
    if (uniformDef && uniformDef.kind === "Uniform") {
      // Loose uniforms have no buffer attribute.
      const time = uniformDef.uniforms.find((u) => u.name === "time");
      expect(time).toBeTruthy();
      expect(time?.buffer).toBeUndefined();
    }
  });

  it("uniforms can live in another file via cross-file type resolution", () => {
    fs.writeFileSync(path.join(workdir, "uniforms.d.ts"), `
      declare const u: { readonly mvp: M44f; readonly tint: V3f };
    `);
    // Re-create the resolver so it picks up the new ambient file.
    const r2 = new TypeResolver({ rootDir: workdir });
    const file = path.join(workdir, "consumer.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(u.tint.x, u.tint.y, u.tint.z, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, r2);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const uniformDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uniformDef).toBeTruthy();
    if (uniformDef && uniformDef.kind === "Uniform") {
      const tint = uniformDef.uniforms.find((u) => u.name === "tint");
      expect(tint).toBeTruthy();
      expect(tint?.buffer).toBe("u");
    }
  });

  it("non-ambient `const tint: V3f` is still a closure capture", () => {
    const file = path.join(workdir, "closure.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const tint: V3f = new V3f(0.5, 0.5, 0.5);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    expect(r!.code).toContain("tint: () => tint");
    const tmpl = extractTemplate(r!.code);
    expect(JSON.stringify(tmpl)).toContain('"scope":"Closure"');
  });
});
