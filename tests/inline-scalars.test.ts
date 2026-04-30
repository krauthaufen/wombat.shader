// Branded scalar aliases — `i32` / `u32` / `f32` route to the
// correct IR type when used in `declare const` ambient uniforms or
// closure annotations. Without these aliases TS can't distinguish
// int from float in user types, so every uniform defaults to f32.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";
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
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-scalars-test-"));
  fs.writeFileSync(path.join(workdir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022", module: "ESNext", moduleResolution: "Bundler",
      strict: false, esModuleInterop: true, skipLibCheck: true, types: [],
    },
    include: ["**/*.ts", "**/*.d.ts"],
  }));
  fs.writeFileSync(path.join(workdir, "types.d.ts"), `
    declare class V2f { x: number; y: number; constructor(x: number, y: number); }
    declare class V4f { x: number; y: number; z: number; w: number; constructor(x: number, y: number, z: number, w: number); }
    declare const __scalarBrand: unique symbol;
    type i32 = number & { readonly [__scalarBrand]?: "i32" };
    type u32 = number & { readonly [__scalarBrand]?: "u32" };
    type f32 = number & { readonly [__scalarBrand]?: "f32" };
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("scalar aliases", () => {
  it("ambient `declare const x: i32` becomes a Uniform of i32", () => {
    const file = path.join(workdir, "i32.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      declare const frame: i32;
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0),
      }));
      // touch frame so it isn't pruned
      const fs2 = fragment(() => ({ outColor: new V4f(frame as f32, 0, 0, 1) }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const code = r!.code;
    // Find the second __wombat_stage call (the one referencing frame).
    const tmpl = JSON.parse(code.slice(code.lastIndexOf("__wombat_stage(") + "__wombat_stage(".length).split(", {")[0]) as Module;
    const uDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uDef).toBeTruthy();
    if (uDef && uDef.kind === "Uniform") {
      const frameU = uDef.uniforms.find((u) => u.name === "frame");
      expect(frameU?.type).toEqual({ kind: "Int", signed: true, width: 32 });
    }
  });

  it("ambient `declare const x: u32` becomes a Uniform of u32", () => {
    const file = path.join(workdir, "u32.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      declare const flags: u32;
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(flags as f32, 0, 0, 1.0),
      }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const uDef = tmpl.values.find((v) => v.kind === "Uniform");
    expect(uDef).toBeTruthy();
    if (uDef && uDef.kind === "Uniform") {
      const flagsU = uDef.uniforms.find((u) => u.name === "flags");
      expect(flagsU?.type).toEqual({ kind: "Int", signed: false, width: 32 });
    }
  });

  it("plain `number` still defaults to f32 (existing behavior preserved)", () => {
    const file = path.join(workdir, "num.ts");
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      declare const time: number;
      const fs = fragment(() => ({ outColor: new V4f(time, time, time, 1.0) }));
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const uDef = tmpl.values.find((v) => v.kind === "Uniform");
    if (uDef && uDef.kind === "Uniform") {
      const timeU = uDef.uniforms.find((u) => u.name === "time");
      expect(timeU?.type).toEqual({ kind: "Float", width: 32 });
    }
  });
});
