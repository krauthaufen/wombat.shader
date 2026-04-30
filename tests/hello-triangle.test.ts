// End-to-end test for the IR → GLSL ES 3.00 / WGSL emitters.
//
// Hand-builds a small "hello triangle" shader pair (one vertex, one
// fragment) using the IR types directly, then snapshots both backends'
// outputs. If you change emitter formatting intentionally, re-run with
// `vitest run -u` to refresh the snapshots.

import { describe, expect, it } from "vitest";
import {
  Mat,
  Tf32,
  Tvoid,
  Vec,
  type EntryDef,
  type Module,
  type Stmt,
  type Type,
  type Var,
} from "@aardworx/wombat.shader-ir";
import { emitGlsl } from "@aardworx/wombat.shader-glsl";
import { emitWgsl } from "@aardworx/wombat.shader-wgsl";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4f: Type = Mat(Tf32, 4, 4);

// ─── helpers to build IR more compactly ───────────────────────────────

function variable(name: string, type: Type, mutable = false): Var {
  return { name, type, mutable };
}

function readInput(name: string, type: Type): import("@aardworx/wombat.shader-ir").Expr {
  return { kind: "ReadInput", scope: "Input", name, type };
}

function readUniform(name: string, type: Type): import("@aardworx/wombat.shader-ir").Expr {
  return { kind: "ReadInput", scope: "Uniform", name, type };
}

function v(va: Var): import("@aardworx/wombat.shader-ir").Expr {
  return { kind: "Var", var: va, type: va.type };
}

function newVec(type: Type, components: import("@aardworx/wombat.shader-ir").Expr[]): import("@aardworx/wombat.shader-ir").Expr {
  return { kind: "NewVector", components, type };
}

function constF(value: number): import("@aardworx/wombat.shader-ir").Expr {
  return { kind: "Const", value: { kind: "Float", value }, type: Tf32 };
}

// ─── vertex shader ───────────────────────────────────────────────────
//
//   layout(location=0) in vec3 a_position;
//   layout(location=1) in vec3 a_color;
//   uniform mat4 u_mvp;
//   out vec3 v_color;
//   void main() {
//     gl_Position = u_mvp * vec4(a_position, 1.0);
//     v_color = a_color;
//   }

function vertexEntry(): EntryDef {
  const body: Stmt = {
    kind: "Sequential",
    body: [
      // gl_Position = u_mvp * vec4(a_position, 1.0)
      {
        kind: "WriteOutput",
        name: "gl_Position",
        value: {
          kind: "Expr",
          value: {
            kind: "MulMatVec",
            lhs: readUniform("u_mvp", Tmat4f),
            rhs: newVec(Tvec4f, [readInput("a_position", Tvec3f), constF(1)]),
            type: Tvec4f,
          },
        },
      },
      // v_color = a_color
      {
        kind: "WriteOutput",
        name: "v_color",
        value: { kind: "Expr", value: readInput("a_color", Tvec3f) },
      },
    ],
  };
  return {
    name: "vsMain",
    stage: "vertex",
    inputs: [
      { name: "a_position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
      { name: "a_color",    type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 1 }] },
    ],
    outputs: [
      { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
      { name: "v_color",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
    ],
    arguments: [],
    returnType: Tvoid,
    body,
    decorations: [],
  };
}

// ─── fragment shader ──────────────────────────────────────────────────
//
//   in vec3 v_color;
//   out vec4 outColor;
//   void main() { outColor = vec4(v_color, 1.0); }

function fragmentEntry(): EntryDef {
  const body: Stmt = {
    kind: "Sequential",
    body: [
      {
        kind: "WriteOutput",
        name: "outColor",
        value: {
          kind: "Expr",
          value: newVec(Tvec4f, [readInput("v_color", Tvec3f), constF(1)]),
        },
      },
    ],
  };
  return {
    name: "fsMain",
    stage: "fragment",
    inputs: [
      { name: "v_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
    ],
    outputs: [
      { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
    ],
    arguments: [],
    returnType: Tvoid,
    body,
    decorations: [],
  };
}

function helloTriangleModule(): Module {
  return {
    types: [],
    values: [
      { kind: "Uniform", uniforms: [{ name: "u_mvp", type: Tmat4f }] },
      { kind: "Entry", entry: vertexEntry() },
      { kind: "Entry", entry: fragmentEntry() },
    ],
  };
}

describe("hello triangle — GLSL ES 3.00", () => {
  it("vertex stage", () => {
    const m = helloTriangleModule();
    const r = emitGlsl(m, "vsMain");
    expect(r.source).toMatchInlineSnapshot(`
      "#version 300 es
      precision highp float;

      uniform mat4 u_mvp;

      layout(location = 0) in vec3 a_position;
      layout(location = 1) in vec3 a_color;
      layout(location = 0) out vec3 v_color;

      void main() {
          gl_Position = (u_mvp * vec4(a_position, 1.0));
          v_color = a_color;
      }
      "
    `);
    expect(r.bindings.uniforms).toEqual([{ name: "u_mvp", type: Tmat4f }]);
    expect(r.bindings.inputs.map((b) => b.location)).toEqual([0, 1]);
  });

  it("fragment stage", () => {
    const r = emitGlsl(helloTriangleModule(), "fsMain");
    expect(r.source).toMatchInlineSnapshot(`
      "#version 300 es
      precision highp float;

      uniform mat4 u_mvp;

      layout(location = 0) in vec3 v_color;
      layout(location = 0) out vec4 outColor;

      void main() {
          outColor = vec4(v_color, 1.0);
      }
      "
    `);
  });
});

describe("hello triangle — WGSL", () => {
  it("vertex stage", () => {
    const r = emitWgsl(helloTriangleModule(), "vsMain");
    expect(r.source).toMatchInlineSnapshot(`
      "@group(0) @binding(0) var<uniform> u_mvp: mat4x4<f32>;
      struct VsMainInput {
          @location(0) a_position: vec3<f32>,
          @location(1) a_color: vec3<f32>,
      };
      struct VsMainOutput {
          @builtin(position) gl_Position: vec4<f32>,
          @location(0) v_color: vec3<f32>,
      };

      @vertex
      fn vsMain(in: VsMainInput) -> VsMainOutput {
          var out: VsMainOutput;
          out.gl_Position = (u_mvp * vec4<f32>(in.a_position, 1.0f));
          out.v_color = in.a_color;
          return out;
      }
      "
    `);
    expect(r.bindings.uniforms[0]?.name).toBe("u_mvp");
  });

  it("fragment stage", () => {
    const r = emitWgsl(helloTriangleModule(), "fsMain");
    expect(r.source).toMatchInlineSnapshot(`
      "@group(0) @binding(0) var<uniform> u_mvp: mat4x4<f32>;
      struct FsMainInput {
          @location(0) v_color: vec3<f32>,
      };
      struct FsMainOutput {
          @location(0) outColor: vec4<f32>,
      };

      @fragment
      fn fsMain(in: FsMainInput) -> FsMainOutput {
          var out: FsMainOutput;
          out.outColor = vec4<f32>(in.v_color, 1.0f);
          return out;
      }
      "
    `);
  });
});
