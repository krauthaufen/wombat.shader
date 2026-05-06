// End-to-end: the inline-marker plugin consumes Semantic / Builtin
// brand annotations on an effect's input/output records and lowers
// them to the right `EntryParameter` shape (semantic name + builtin
// decoration where applicable). User picks any field name they
// like; the brand carries the meaning.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";

let workdir: string;
let resolver: TypeResolver;

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-brand-test-"));
  fs.writeFileSync(path.join(workdir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022", module: "ESNext", moduleResolution: "Bundler",
      strict: false, esModuleInterop: true, skipLibCheck: true, types: [],
    },
    include: ["**/*.ts", "**/*.d.ts"],
  }));
  // Shim type defs — the plugin only needs to RECOGNISE the brand
  // structure; the actual `@aardworx/wombat.shader/types` shape is
  // mirrored in this in-test ambient declaration. The brands use the
  // same `__wombat_semantic` / `__wombat_builtin` plain-string keys
  // the plugin's walker matches.
  fs.writeFileSync(path.join(workdir, "types.d.ts"), `
    declare class V2f { x: number; y: number; constructor(x: number, y: number); }
    declare class V3f { x: number; y: number; z: number; constructor(x: number, y: number, z: number); }
    declare class V4f {
      x: number; y: number; z: number; w: number;
      constructor(x: number, y: number, z: number, w: number);
    }

    declare type Semantic<T, N extends string> = T & { readonly __wombat_semantic: N };
    declare type Builtin<T, K extends string> = T & { readonly __wombat_builtin: K };

    declare type Position<T = V4f> = Semantic<T, "Positions">;
    declare type Color<T = V4f> = Semantic<T, "Colors">;
    declare type Normal<T = V3f> = Semantic<T, "Normals">;

    declare type ClipPosition<T = V4f> = Builtin<T, "position">;
    declare type FragCoord<T = V4f> = Builtin<T, "position">;
    declare type FragDepth<T = number> = Builtin<T, "frag_depth">;
  `);
  resolver = new TypeResolver({ rootDir: workdir });
});

afterAll(() => {
  fs.rmSync(workdir, { recursive: true, force: true });
});

function transform(src: string): string {
  const file = path.join(workdir, `t${Math.random().toString(36).slice(2, 8)}.ts`);
  fs.writeFileSync(file, src);
  const r = transformInlineShaders(src, file, resolver);
  if (!r) throw new Error("transform returned null");
  return r.code;
}

function templateJson(code: string): string {
  const idx = code.indexOf("__wombat_stage(");
  if (idx < 0) throw new Error("no __wombat_stage call");
  const start = code.indexOf("{", idx);
  let depth = 0;
  for (let i = start; i < code.length; i++) {
    if (code[i] === "{") depth++;
    else if (code[i] === "}") {
      depth--;
      if (depth === 0) return code.slice(start, i + 1);
    }
  }
  throw new Error("unterminated JSON template");
}

describe("inline plugin: Semantic / Builtin brand consumption", () => {
  it("`Position<V3f>` on an input field sets semantic regardless of field name", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { pos: Position<V3f>; col: Color }) => ({
        gl_Position: new V4f(v.pos.x, v.pos.y, v.pos.z, 1.0),
        col: v.col,
      }));
    `;
    const code = transform(src);
    const json = templateJson(code);
    expect(json).toMatch(/"name":"pos"[\s\S]*?"semantic":"Positions"/);
    expect(json).toMatch(/"name":"pos"[\s\S]*?"kind":"Vector"[\s\S]*?"dim":3/);
    expect(json).toMatch(/"name":"col"[\s\S]*?"semantic":"Colors"/);
    expect(json).toMatch(/"name":"col"[\s\S]*?"kind":"Vector"[\s\S]*?"dim":4/);
  });

  it("`ClipPosition` on a vertex output decorates as @builtin(position)", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { pos: Position<V3f> }): {
        clip: ClipPosition;
        tint: Color;
      } => ({
        clip: new V4f(v.pos.x, v.pos.y, v.pos.z, 1.0),
        tint: new V4f(1.0, 0.0, 0.0, 1.0),
      }));
    `;
    const code = transform(src);
    const json = templateJson(code);
    expect(json).toMatch(/"name":"clip"[\s\S]*?"decorations":\[\{"kind":"Builtin","value":"position"\}\]/);
    expect(json).toMatch(/"name":"tint"[\s\S]*?"semantic":"Colors"/);
  });

  it("`FragCoord` on a fragment input → @builtin(position)", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { fc: FragCoord; tint: Color }) => ({
        outColor: new V4f(input.fc.x, input.fc.y, input.tint.x, 1.0),
      }));
    `;
    const code = transform(src);
    const json = templateJson(code);
    expect(json).toMatch(/"name":"fc"[\s\S]*?"decorations":\[\{"kind":"Builtin","value":"position"\}\]/);
  });

  it("`FragDepth` on a fragment output → @builtin(frag_depth)", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { col: Color }): {
        outColor: Color;
        d: FragDepth;
      } => ({ outColor: input.col, d: 0.5 }));
    `;
    const code = transform(src);
    const json = templateJson(code);
    expect(json).toMatch(/"name":"d"[\s\S]*?"decorations":\[\{"kind":"Builtin","value":"frag_depth"\}\]/);
  });

  it("rejects FragCoord on a vertex input (wrong stage)", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { pos: Position<V3f>; bad: FragCoord }) => ({
        gl_Position: new V4f(0.0, 0.0, 0.0, 1.0),
      }));
    `;
    expect(() => transform(src)).toThrowError(/not allowed on vertex-in/);
  });

  it("rejects FragDepth on a vertex output (wrong direction)", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { pos: Position<V3f> }): {
        gl_Position: V4f;
        d: FragDepth;
      } => ({
        gl_Position: new V4f(0.0, 0.0, 0.0, 1.0),
        d: 0.5,
      }));
    `;
    expect(() => transform(src)).toThrowError(/not allowed on vertex-out/);
  });
});
