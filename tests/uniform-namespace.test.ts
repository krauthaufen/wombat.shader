// Standard `uniform` namespace from `@aardworx/wombat.shader/uniforms`
// — Aardvark UniformScope mirror. Plugin classifies `uniform.X`
// access as a uniform binding via the existing ambient-namespace
// path; consumer code stops writing per-file
// `declare const ModelTrafo: M44f;` boilerplate.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { transformInlineShaders, TypeResolver } from "@aardworx/wombat.shader-vite";

let workdir: string;
let resolver: TypeResolver;

beforeAll(() => {
  workdir = fs.mkdtempSync(path.join(os.tmpdir(), "wombat-uniform-test-"));
  fs.writeFileSync(path.join(workdir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022", module: "ESNext", moduleResolution: "Bundler",
      strict: false, esModuleInterop: true, skipLibCheck: true, types: [],
    },
    include: ["**/*.ts", "**/*.d.ts"],
  }));
  // Mirror `@aardworx/wombat.shader/uniforms` ambient declarations.
  fs.writeFileSync(path.join(workdir, "types.d.ts"), `
    declare class V3f { x: number; y: number; z: number; constructor(x: number, y: number, z: number); }
    declare class V4f {
      x: number; y: number; z: number; w: number;
      constructor(x: number, y: number, z: number, w: number);
      mul(other: V4f | M44f): V4f;
    }
    declare class M44f {
      mul(other: V4f | M44f): V4f;
    }

    interface UniformScope {
      readonly ModelTrafo: M44f;
      readonly ViewTrafo: M44f;
      readonly ProjTrafo: M44f;
      readonly ViewProjTrafo: M44f;
      readonly NormalMatrix: M44f;
      readonly CameraLocation: V3f;
      readonly LightLocation: V3f;
      readonly ViewportSize: V4f;
    }
    declare const uniform: UniformScope;
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

describe("uniform namespace — plugin lowers `uniform.X` to ReadInput('Uniform', X)", () => {
  it("`uniform.ModelTrafo` is treated as a uniform binding (no closure capture)", () => {
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { Positions: V3f }) => ({
        gl_Position: uniform.ViewProjTrafo.mul(uniform.ModelTrafo.mul(
          new V4f(v.Positions.x, v.Positions.y, v.Positions.z, 1.0),
        )),
      }));
    `;
    const code = transform(src);
    // Both uniforms are registered as Uniform decls in the IR
    // template, with the right type (mat4x4).
    expect(code).toMatch(/"kind":"Uniform"[\s\S]*?"name":"ModelTrafo"[\s\S]*?"kind":"Matrix"/);
    expect(code).toMatch(/"kind":"Uniform"[\s\S]*?"name":"ViewProjTrafo"[\s\S]*?"kind":"Matrix"/);
    // `uniform.ModelTrafo` lowers to a ReadInput("Uniform") — there
    // are no closure captures (the `holes` object is empty).
    expect(code).toMatch(/__wombat_stage\([\s\S]+, \{\}, "[0-9a-f]+", \{\}\)/);
    // The uniform doesn't get the `uniform.` prefix in IR — it's
    // just `ModelTrafo`. The runtime matches by bare name.
    expect(code).toMatch(/"scope":"Uniform"[\s\S]*?"name":"ModelTrafo"/);
  });

  it("only the uniforms actually accessed end up in the Uniform decl list", () => {
    // The shader reads ModelTrafo and CameraLocation only. Other
    // members of UniformScope (ViewTrafo, ProjTrafo, …) are NOT
    // emitted — `reduceUniforms` would drop them anyway, but the
    // plugin shouldn't include them in the first place because the
    // body never references them.
    const src = `
      import { vertex } from "@aardworx/wombat.shader";
      const vs = vertex((v: { Positions: V3f }) => ({
        gl_Position: uniform.ModelTrafo.mul(new V4f(
          v.Positions.x - uniform.CameraLocation.x,
          v.Positions.y - uniform.CameraLocation.y,
          v.Positions.z - uniform.CameraLocation.z,
          1.0,
        )),
      }));
    `;
    const code = transform(src);
    expect(code).toContain("ModelTrafo");
    expect(code).toContain("CameraLocation");
    // Untouched members not pulled into the Uniform block.
    expect(code).not.toMatch(/"name":"ViewTrafo"\b/);
    expect(code).not.toMatch(/"name":"ProjTrafo"\b/);
    expect(code).not.toMatch(/"name":"NormalMatrix"\b/);
  });

  it("V3f-typed uniform (CameraLocation) emits as Vector dim 3", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment(() => ({
        outColor: new V4f(uniform.CameraLocation.x, uniform.CameraLocation.y, uniform.CameraLocation.z, 1.0),
      }));
    `;
    const code = transform(src);
    expect(code).toMatch(/"name":"CameraLocation"[\s\S]*?"kind":"Vector"[\s\S]*?"dim":3/);
  });
});
