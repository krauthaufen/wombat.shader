// `aval<T>`-typed captures: the rule is type-driven —
// `aval<T>` ⇒ uniform binding (carried on Effect.avalBindings),
// plain `T` ⇒ specialize as constant (existing path).

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
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-aval-test-"));
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
    declare class M44f { constructor(); }
    declare interface aval<T> { __brand: "aval"; readonly value: T; }
    declare function aval<T>(initial: T): aval<T>;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("aval-typed captures → uniforms", () => {
  it("`const tint: aval<V3f>` capture becomes a Uniform decl", () => {
    const file = path.join(workdir, "app.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tint: aval<V3f> = aval(new V3f(1, 0, 0));
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    // Uniform decl emitted for `tint` with type V3f (unwrapped from aval).
    const uDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uDef).toBeTruthy();
    if (uDef && uDef.kind === "Uniform") {
      const tintU = uDef.uniforms.find((u) => u.name === "tint");
      expect(tintU).toBeTruthy();
      expect(tintU?.type).toEqual({ kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 });
    }
    // No closure scope ref — body uses Uniform.
    const json = JSON.stringify(tmpl);
    expect(json).not.toContain('"scope":"Closure"');
    expect(json).toContain('"scope":"Uniform"');
    // The plugin emits the aval getter as the 4th arg.
    expect(r!.code).toContain("tint: () => tint");
    expect(r!.code).toMatch(/, "[0-9a-f]{16}", \{ tint: \(\) => tint \}\)/);
  });

  it("plain V3f next to aval<V3f> — only the plain one specializes", () => {
    const file = path.join(workdir, "mixed.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const fixedTint: V3f = new V3f(0.5, 0.5, 0.5);
      const animTint: aval<V3f> = aval(new V3f(1, 0, 0));
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(fixedTint.x, animTint.y, fixedTint.z * animTint.x, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    // Closure hole present for fixedTint, but animTint is on the
    // uniform path.
    expect(r!.code).toContain("fixedTint: () => fixedTint");
    expect(r!.code).toContain("animTint: () => animTint");
    const tmpl = extractTemplate(r!.code);
    // Body should reference fixedTint via Closure scope, animTint via Uniform.
    const json = JSON.stringify(tmpl);
    expect(json).toContain('"scope":"Closure"');
    expect(json).toContain('"scope":"Uniform"');
    expect(json).toContain('"name":"animTint"'); // uniform decl
  });

  it("Effect.compile() exposes avalBindings on the CompiledEffect", () => {
    const file = path.join(workdir, "compiled.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const tint: aval<V3f> = aval(new V3f(1, 0, 0));
      const fs = fragment(() => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    // Hand the template + a dummy aval to the runtime stage().
    const fakeAval = { __brand: "aval", value: [1, 0, 0] };
    const fx = effect(stage(tmpl, {}, undefined, { tint: () => fakeAval }));
    const compiled = fx.compile({ target: "wgsl" });
    expect(compiled.avalBindings.tint).toBeTypeOf("function");
    expect(compiled.avalBindings.tint!()).toBe(fakeAval);
    // The IR shouldn't have inlined any value — `tint.*` reads stay
    // as uniform references.
    expect(compiled.stages[0]!.source).not.toMatch(/0\.\d+f?,\s*0\.\d+f?,\s*0\.\d+f?/);
    expect(compiled.stages[0]!.source).toContain("tint");
  });
});
