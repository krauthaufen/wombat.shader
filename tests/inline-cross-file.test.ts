// Cross-file helper resolution: a helper defined in another source
// file is followed through its `import` declaration via the TS
// LanguageService, walked, translated, and emitted as a `Function`
// ValueDef alongside the entry.

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
    else if (code[i] === "}") { depth--; if (depth === 0) { i++; break; } }
  }
  return JSON.parse(code.slice(start, i)) as Module;
}

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-cross-file-test-"));
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
    declare function mix<T>(a: T, b: T, t: number): T;
  `);
  fs.writeFileSync(path.join(workdir, "shaders-utils.ts"), `
    export function blendColors(a: V3f, b: V3f, t: number): V3f {
      return mix(a, b, t);
    }
    export const luminance = (c: V3f): number =>
      c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("cross-file helper resolution", () => {
  it("named import from another file is discovered and emitted", () => {
    const file = path.join(workdir, "app.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      import { blendColors } from "./shaders-utils.js";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(blendColors(new V3f(1, 0, 0), new V3f(0, 0, 1), input.v_uv.x).x, 0.5, 0.5, 1),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const fns = tmpl.values.filter((v) => v.kind === "Function");
    const names = fns.map((v) => v.kind === "Function" ? v.signature.name : "");
    expect(names).toContain("blendColors");
    // Body call resolves to the proper FunctionRef.
    const json = JSON.stringify(tmpl);
    expect(json).toContain('"id":"blendColors"');
  });

  it("imported `const fn = (…) => …` arrow helper also resolves cross-file", () => {
    const file = path.join(workdir, "app2.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      import { luminance } from "./shaders-utils.js";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(luminance(new V3f(input.v_uv.x, input.v_uv.y, 0.5)), 0, 0, 1),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const fns = tmpl.values
      .filter((v) => v.kind === "Function")
      .map((v) => v.kind === "Function" ? v.signature.name : "");
    expect(fns).toContain("luminance");
  });

  it("transitive cross-file: helper imports another helper from a third file", () => {
    fs.writeFileSync(path.join(workdir, "math-utils.ts"), `
      export function half(x: number): number { return x * 0.5; }
    `);
    fs.writeFileSync(path.join(workdir, "color-utils.ts"), `
      import { half } from "./math-utils.js";
      export function softTint(c: V3f): V3f {
        return new V3f(half(c.x), half(c.y), half(c.z));
      }
    `);
    const file = path.join(workdir, "app3.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      import { softTint } from "./color-utils.js";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(softTint(new V3f(input.v_uv.x, input.v_uv.y, 0.5)).x, 0, 0, 1),
      }));
    `;
    fs.writeFileSync(file, src);
    // Re-create resolver so it picks up the new files.
    const r2 = new TypeResolver({ rootDir: workdir });
    const r = transformInlineShaders(src, file, r2);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const fns = tmpl.values
      .filter((v) => v.kind === "Function")
      .map((v) => v.kind === "Function" ? v.signature.name : "");
    expect(fns).toContain("softTint");
    expect(fns).toContain("half");
  });
});
