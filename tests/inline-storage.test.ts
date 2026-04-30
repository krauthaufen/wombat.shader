// `Storage<T>` and `StorageTexture*<F, A>` captures lower to
// `StorageBuffer` / `Sampler<StorageTexture>` ValueDefs. Both ride
// the same binding-getter path as samplers/avals.

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
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-storage-test-"));
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
    declare class V2i { x: number; y: number; }
    declare class V3u { x: number; y: number; z: number; }

    declare const __scalarBrand: unique symbol;
    type i32 = number & { readonly [__scalarBrand]?: "i32" };
    type u32 = number & { readonly [__scalarBrand]?: "u32" };
    type f32 = number & { readonly [__scalarBrand]?: "f32" };

    declare const __storageBrand: unique symbol;
    type Storage<T, A extends "read" | "read_write" = "read_write"> =
      T & { readonly [__storageBrand]?: ["buffer", A] };
    type StorageTexture2D<F extends string, A extends "read" | "write" | "read_write" = "write"> =
      { readonly [__storageBrand]?: ["texture-2d", F, A] };

    interface ComputeBuiltins { readonly globalInvocationId: V3u; }
    declare function textureStore(t: StorageTexture2D<string, "write" | "read_write">, c: V2i, v: V4f): void;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

describe("inline storage buffers", () => {
  it("`declare const buf: Storage<i32[]>` becomes a StorageBuffer ValueDef", () => {
    const file = path.join(workdir, "buf.ts");
    const src = `
      import { compute } from "@aardworx/wombat.shader-runtime";
      declare const buf: Storage<i32[]>;
      /** @workgroupSize 64 */
      const cs = compute((b: ComputeBuiltins) => {
        const i = b.globalInvocationId.x as i32;
        buf[i] = i;
      });
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const sb = tmpl.values.find((v) => v.kind === "StorageBuffer");
    expect(sb).toBeTruthy();
    if (sb && sb.kind === "StorageBuffer") {
      expect(sb.name).toBe("buf");
      expect(sb.layout).toEqual({
        kind: "Array",
        element: { kind: "Int", signed: true, width: 32 },
        length: "runtime",
      });
      // inferStorageAccess flips access to read_write because of `buf[i] = …`.
    }
    // The plugin emits a binding-getter (4th arg) for the buffer.
    expect(r!.code).toContain("buf: () => buf");
  });

  it("compiles end-to-end into WGSL with `var<storage, read_write>`", () => {
    const file = path.join(workdir, "buf-wgsl.ts");
    const src = `
      import { compute } from "@aardworx/wombat.shader-runtime";
      declare const buf: Storage<u32[]>;
      /** @workgroupSize 64 */
      const cs = compute((b: ComputeBuiltins) => {
        const i = b.globalInvocationId.x as i32;
        buf[i] = (i as u32);
      });
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { buf: () => null }));
    const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toMatch(/var<storage,\s*read_write>\s+buf:\s*array<u32>/);
  });

  it("read-only access stays `read` thanks to inferStorageAccess", () => {
    const file = path.join(workdir, "buf-read.ts");
    const src = `
      import { compute } from "@aardworx/wombat.shader-runtime";
      declare const inBuf: Storage<u32[], "read">;
      declare const outBuf: Storage<u32[]>;
      /** @workgroupSize 64 */
      const cs = compute((b: ComputeBuiltins) => {
        const i = b.globalInvocationId.x as i32;
        outBuf[i] = inBuf[i];
      });
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { inBuf: () => null, outBuf: () => null }));
    const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toMatch(/var<storage,\s*read>\s+inBuf/);
    expect(wgsl).toMatch(/var<storage,\s*read_write>\s+outBuf/);
  });
});

describe("inline storage textures", () => {
  it("`StorageTexture2D<\"rgba8unorm\", \"write\">` lowers to `texture_storage_2d<rgba8unorm, write>` in WGSL", () => {
    const file = path.join(workdir, "tex.ts");
    const src = `
      import { compute } from "@aardworx/wombat.shader-runtime";
      declare const out: StorageTexture2D<"rgba8unorm", "write">;
      /** @workgroupSize 8 8 1 */
      const cs = compute((b: ComputeBuiltins) => {
        textureStore(out, new V2i(b.globalInvocationId.x as i32, b.globalInvocationId.y as i32), new V4f(1, 0, 0, 1));
      });
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver);
    expect(r).not.toBeNull();
    const tmpl = extractTemplate(r!.code);
    const samplerDef = tmpl.values.find((v) => v.kind === "Sampler");
    expect(samplerDef).toBeTruthy();
    if (samplerDef && samplerDef.kind === "Sampler") {
      expect(samplerDef.type.kind).toBe("StorageTexture");
    }
    const fx = effect(stage(tmpl, {}, undefined, { out: () => null }));
    const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toContain("texture_storage_2d<rgba8unorm, write>");
    expect(wgsl).toContain("textureStore(out,");
  });

  it("GLSL target rejects storage textures with a clear error", () => {
    const file = path.join(workdir, "tex-glsl.ts");
    const src = `
      import { compute } from "@aardworx/wombat.shader-runtime";
      declare const out: StorageTexture2D<"rgba8unorm", "write">;
      /** @workgroupSize 8 8 1 */
      const cs = compute((b: ComputeBuiltins) => {
        textureStore(out, new V2i(0, 0), new V4f(1, 0, 0, 1));
      });
    `;
    fs.writeFileSync(file, src);
    const r = transformInlineShaders(src, file, resolver)!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl, {}, undefined, { out: () => null }));
    expect(() => fx.compile({ target: "glsl" })).toThrow(/storage textures .* not supported/);
  });
});
