// End-to-end Dawn validation for shaders compiled through the
// inline-marker plugin. Mirrors `wgsl-validation.test.ts` (which uses
// the `parseShader` path) but exercises `transformInlineShaders`,
// where the for-loop closure-capture bug and the storage-buffer ↔
// uniform binding-slot collision both lived. Adding shapes here is
// the fastest way to lock in inline-plugin regressions.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";
import { effect, stage } from "@aardworx/wombat.shader";
import type { Module } from "@aardworx/wombat.shader/ir";
import { validateWgsl, formatFailure } from "./_wgsl-validator.js";

let workdir: string;
let resolver: TypeResolver;

function extractTemplates(code: string): Module[] {
  const out: Module[] = [];
  const re = /__wombat_(stage|compute)\(/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code)) !== null) {
    let i = code.indexOf("{", m.index);
    if (i < 0) continue;
    const start = i;
    let depth = 0;
    for (; i < code.length; i++) {
      if (code[i] === "{") depth++;
      else if (code[i] === "}") {
        depth--;
        if (depth === 0) { i++; break; }
      }
    }
    out.push(JSON.parse(code.slice(start, i)) as Module);
  }
  return out;
}

interface CompileExtras {
  closures?: Record<string, () => unknown>;
  bindings?: Record<string, () => unknown>;
}

async function emitAndValidate(
  src: string,
  extras: CompileExtras = {},
): Promise<void> {
  const file = path.join(workdir, `t_${Math.random().toString(36).slice(2, 8)}.ts`);
  fs.writeFileSync(file, src);
  const r = transformInlineShaders(src, file, resolver);
  if (!r) throw new Error("transformInlineShaders returned null");
  const tmpls = extractTemplates(r.code);
  if (tmpls.length === 0) throw new Error("no __wombat_stage/__wombat_compute templates emitted");
  const stages = tmpls.map((t) => stage(t, extras.closures ?? {}, undefined, extras.bindings ?? {}));
  const fx = effect(...stages);
  const compiled = fx.compile({ target: "wgsl" });
  for (const s of compiled.stages) {
    const v = await validateWgsl(s.source);
    if (!v.ok) {
      throw new Error(`${s.entryName} (${s.stage}) — ${formatFailure(v)}\n--- emitted source ---\n${s.source}\n--- end ---`);
    }
  }
}

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-inline-validation-"));
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
    declare class V4f { x: number; y: number; z: number; w: number; constructor(x: number, y: number, z: number, w: number); div(s: number): V4f; }
    declare class V2i { x: number; y: number; }
    declare class V3u { x: number; y: number; z: number; }
    declare class M44f { mul(v: V4f): V4f; }

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

    declare function clamp(x: number, a: number, b: number): number;
    declare function max(a: number, b: number): number;
    declare function min(a: number, b: number): number;
    declare function sqrt(x: number): number;
    declare function abs(x: number): number;
    declare function fract(x: number): number;
    declare function step(edge: number, x: number): number;
    declare function dFdx(x: number): number;
    declare function dFdy(x: number): number;
    declare function discard(): never;
    declare function atomicAdd(p: any, v: u32): u32;
    declare function atomicLoad(p: any): u32;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

// ─── For-loop scoping (regression: text-sdf processCand) ─────────────

describe("inline plugin: for-loop bound variables aren't closure-captured", () => {
  it("simple `for (let i = 0; i < N; i++)` validates", async () => {
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => {
        let acc: f32 = 0.0;
        for (let i: i32 = 0 as i32; i < (16 as i32); i = i + (1 as i32)) {
          acc = acc + input.v_uv.x;
        }
        return { outColor: new V4f(acc, 0.0, 0.0, 1.0) };
      });
    `);
  });

  it("`for (var i = …)` (mutable init) — Dawn requires u32 shift on indices", async () => {
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => {
        let mask: u32 = 0 as u32;
        for (let i: u32 = 0 as u32; i < (8 as u32); i = i + (1 as u32)) {
          mask = mask | ((1 as u32) << i);
        }
        return { outColor: new V4f((mask as f32) / 255.0, 0.0, 0.0, 1.0) };
      });
    `);
  });

  it("nested for-loops, inner reads outer's loop variable", async () => {
    // Mirrors text-sdf's processCand: `for (let s = …) { let t = s * …; for (let n = …) { … } }`
    // Before the for-statement scoping fix this captured \`s\` and \`n\`
    // as closures, runtime threw "ReferenceError: s is not defined".
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => {
        let best: f32 = 1.0e20;
        for (let s: u32 = 0 as u32; s < (5 as u32); s = s + (1 as u32)) {
          let t: f32 = (s as f32) * 0.25;
          for (let n: u32 = 0 as u32; n < (8 as u32); n = n + (1 as u32)) {
            const dt = (n as f32) * 0.01;
            t = clamp(t - dt, 0.0, 1.0);
          }
          const d = (t - input.v_uv.x) * (t - input.v_uv.x);
          best = min(best, d);
        }
        return { outColor: new V4f(best, 0.0, 0.0, 1.0) };
      });
    `);
  });

  it("for-loop variable referenced in helper called from marker body", async () => {
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      function processCand(seed: u32, x: f32): f32 {
        let t: f32 = (seed as f32) * 0.25;
        for (let n: u32 = 0 as u32; n < (4 as u32); n = n + (1 as u32)) {
          t = t - (n as f32) * 0.01;
        }
        return t * x;
      }
      const fs = fragment((input: { v_uv: V2f }) => {
        let acc: f32 = 0.0;
        for (let s: u32 = 0 as u32; s < (3 as u32); s = s + (1 as u32)) {
          acc = acc + processCand(s, input.v_uv.x);
        }
        return { outColor: new V4f(acc, 0.0, 0.0, 1.0) };
      });
    `);
  });
});

// ─── Binding-slot collision (regression: text-sdf storage + uniform) ─

describe("inline plugin: uniform / storage / sampler bindings don't collide", () => {
  it("uniform block + storage buffer share @group(0) with distinct bindings", async () => {
    // Pre-fix this emitted both at @group(0) @binding(0) →
    // "binding index (0) was specified by a previous entry".
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      declare const Color: V4f;
      declare const colors: Storage<V4f[], "read">;
      const fs = fragment((input: { v_idx: f32 }) => {
        const i = (input.v_idx as u32);
        const ZERO: u32 = 0 as u32;
        const c = colors[i + ZERO] as V4f;
        return { outColor: new V4f(Color.x + c.x, 0.0, 0.0, 1.0) };
      });
    `, { bindings: { colors: () => null } });
  });

  it("two uniform-blocks (named buffers) get distinct binding slots", async () => {
    // Each `declare const X: T` becomes its own UniformDecl; without a
    // shared slot counter both end up at @binding(0).
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      declare const Color: V4f;
      declare const Tint: V4f;
      const fs = fragment(() => ({
        outColor: new V4f(Color.x * Tint.x, Color.y * Tint.y, Color.z * Tint.z, 1.0),
      }));
    `);
  });

  it("ordering: emitter walks uniforms → samplers → storage", async () => {
    // Module-order placement of StorageBuffer / Uniform in the IR
    // shouldn't matter — the emitter pins uniforms first so the
    // runtime's interface walk lines up with the WGSL @binding(...).
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      declare const data: Storage<f32[], "read">;
      declare const Scale: f32;
      const fs = fragment((input: { v_idx: f32 }) => {
        const i = (input.v_idx as u32);
        const v = data[i] as f32;
        return { outColor: new V4f(v * Scale, 0.0, 0.0, 1.0) };
      });
    `, { bindings: { data: () => null } });
  });
});

// ─── Compute / atomics / storage textures ────────────────────────────

describe("inline plugin: compute markers", () => {
  it("compute with atomic counter validates", async () => {
    await emitAndValidate(`
      import { compute } from "@aardworx/wombat.shader";
      declare const counter: Storage<u32, "read_write">;
      /** @workgroupSize 64 */
      const cs = compute((b: ComputeBuiltins) => {
        atomicAdd(counter, 1 as u32);
      });
    `, { bindings: { counter: () => null } });
  });

  it("compute writes to a storage texture", async () => {
    await emitAndValidate(`
      import { compute } from "@aardworx/wombat.shader";
      declare const out: StorageTexture2D<"rgba8unorm", "write">;
      /** @workgroupSize 8, 8 */
      const cs = compute((b: ComputeBuiltins) => {
        const x = b.globalInvocationId.x as i32;
        const y = b.globalInvocationId.y as i32;
        textureStore(out, new V2i(x, y), new V4f((x as f32) / 256.0, (y as f32) / 256.0, 0.0, 1.0));
      });
    `, { bindings: { out: () => null } });
  });

  it("compute with for-loop accumulator into storage buffer", async () => {
    // Combines compute + storage + nested for-loop bound variables —
    // the stack of regressions in one test.
    await emitAndValidate(`
      import { compute } from "@aardworx/wombat.shader";
      declare const buf: Storage<f32[], "read_write">;
      /** @workgroupSize 32 */
      const cs = compute((b: ComputeBuiltins) => {
        const idx = b.globalInvocationId.x as u32;
        let s: f32 = 0.0;
        for (let i: u32 = 0 as u32; i < (8 as u32); i = i + (1 as u32)) {
          for (let j: u32 = 0 as u32; j < (4 as u32); j = j + (1 as u32)) {
            s = s + (i as f32) * (j as f32);
          }
        }
        buf[idx] = s;
      });
    `, { bindings: { buf: () => null } });
  });
});

// ─── Cross-stage compose (vertex + fragment via effect()) ────────────

describe("inline plugin: vertex + fragment composition", () => {
  it("VS+FS markers compose, varyings flow through linkCrossStage", async () => {
    await emitAndValidate(`
      import { vertex, fragment } from "@aardworx/wombat.shader";
      const vs = vertex((input: { Positions: V4f; UV: V2f }) => ({
        gl_Position: input.Positions,
        v_uv: input.UV,
      }));
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0),
      }));
    `);
  });

  it("VS+FS with helper called from both stages — single Function ValueDef per helper", async () => {
    await emitAndValidate(`
      import { vertex, fragment } from "@aardworx/wombat.shader";
      function smoothFalloff(t: f32): f32 {
        return t * t * (3.0 - 2.0 * t);
      }
      const vs = vertex((input: { Positions: V4f; UV: V2f }) => ({
        gl_Position: input.Positions,
        v_uv: input.UV,
      }));
      const fs = fragment((input: { v_uv: V2f }) => {
        const a = smoothFalloff(input.v_uv.x);
        const b = smoothFalloff(input.v_uv.y);
        return { outColor: new V4f(a, b, 0.5, 1.0) };
      });
    `);
  });
});

// ─── Larger control-flow shapes ──────────────────────────────────────

describe("inline plugin: control flow", () => {
  it("for + break + continue inside a marker body", async () => {
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => {
        let acc: f32 = 0.0;
        for (let i: i32 = 0 as i32; i < (32 as i32); i = i + (1 as i32)) {
          if ((i % (2 as i32)) == (0 as i32)) { continue; }
          acc = acc + input.v_uv.x;
          if (acc > 0.5) { break; }
        }
        return { outColor: new V4f(acc, 0.0, 0.0, 1.0) };
      });
    `);
  });

  it("early-return from inside a for-loop", async () => {
    await emitAndValidate(`
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => {
        for (let i: i32 = 0 as i32; i < (8 as i32); i = i + (1 as i32)) {
          if (input.v_uv.x > (i as f32) * 0.1) {
            return { outColor: new V4f((i as f32) / 8.0, 0.0, 0.0, 1.0) };
          }
        }
        return { outColor: new V4f(0.0, 1.0, 0.0, 1.0) };
      });
    `);
  });
});
