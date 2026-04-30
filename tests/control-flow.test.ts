// Coverage tests: swizzles, control flow, intrinsics, conversions.

import { describe, expect, it } from "vitest";
import {
  Tbool,
  Tf32,
  Ti32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
  type IntrinsicRef,
  type Module,
  type Stmt,
  type Type,
  type Var,
} from "@aardworx/wombat.shader-ir";
import { emitGlsl } from "@aardworx/wombat.shader-glsl";
import { emitWgsl } from "@aardworx/wombat.shader-wgsl";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

const SinIntrinsic: IntrinsicRef = {
  name: "sin",
  pure: true,
  emit: { glsl: "sin", wgsl: "sin" },
  returnTypeOf: ([t]) => t!,
};

function v(va: Var): Expr { return { kind: "Var", var: va, type: va.type }; }
function constI(value: number): Expr { return { kind: "Const", value: { kind: "Int", signed: true, value }, type: Ti32 }; }
function constF(value: number): Expr { return { kind: "Const", value: { kind: "Float", value }, type: Tf32 }; }

function testEntry(body: Stmt): EntryDef {
  return {
    name: "fsMain",
    stage: "fragment",
    inputs: [],
    outputs: [
      { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
    ],
    arguments: [],
    returnType: Tvoid,
    body,
    decorations: [],
  };
}

describe("control flow + intrinsics — GLSL", () => {
  it("if/else, sin call, swizzle, ReadInput uniform", () => {
    const t: Var = { name: "t", type: Tf32, mutable: true };
    const colour: Var = { name: "colour", type: Tvec3f, mutable: true };
    const body: Stmt = {
      kind: "Sequential",
      body: [
        { kind: "Declare", var: t, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 } } },
        { kind: "Declare", var: colour, init: { kind: "Expr", value: { kind: "NewVector", components: [constF(1), constF(0), constF(0)], type: Tvec3f } } },
        {
          kind: "If",
          cond: { kind: "Gt", lhs: { kind: "CallIntrinsic", op: SinIntrinsic, args: [v(t)], type: Tf32 }, rhs: constF(0), type: Tbool },
          then: {
            kind: "Write",
            target: { kind: "LSwizzle", target: { kind: "LVar", var: colour, type: Tvec3f }, comps: ["x"], type: Tf32 },
            value: constF(1),
          },
          else: {
            kind: "Write",
            target: { kind: "LSwizzle", target: { kind: "LVar", var: colour, type: Tvec3f }, comps: ["y"], type: Tf32 },
            value: constF(1),
          },
        },
        {
          kind: "WriteOutput",
          name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [v(colour), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "u_time", type: Tf32 }] },
        { kind: "Entry", entry: testEntry(body) },
      ],
    };
    const out = emitGlsl(m).source;
    expect(out).toContain("uniform float u_time;");
    expect(out).toContain("if ((sin(t) > 0.0))");
    expect(out).toContain("colour.x = 1.0;");
    expect(out).toContain("colour.y = 1.0;");
    expect(out).toContain("outColor = vec4(colour, 1.0);");
  });

  it("for loop with declare-init", () => {
    const i: Var = { name: "i", type: Ti32, mutable: true };
    const accum: Var = { name: "accum", type: Tf32, mutable: true };
    const body: Stmt = {
      kind: "Sequential",
      body: [
        { kind: "Declare", var: accum, init: { kind: "Expr", value: constF(0) } },
        {
          kind: "For",
          init: { kind: "Declare", var: i, init: { kind: "Expr", value: constI(0) } },
          cond: { kind: "Lt", lhs: v(i), rhs: constI(10), type: Tbool },
          step: { kind: "Increment", target: { kind: "LVar", var: i, type: Ti32 }, prefix: false },
          body: {
            kind: "Write",
            target: { kind: "LVar", var: accum, type: Tf32 },
            value: { kind: "Add", lhs: v(accum), rhs: constF(1), type: Tf32 },
          },
        },
        {
          kind: "WriteOutput",
          name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [v(accum), v(accum), v(accum), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry: testEntry(body) }] };
    const out = emitGlsl(m).source;
    expect(out).toContain("for (int i = 0; (i < 10); i++)");
    expect(out).toContain("accum = (accum + 1.0);");
  });
});

describe("control flow + intrinsics — WGSL", () => {
  it("if/else lowers correctly", () => {
    const colour: Var = { name: "colour", type: Tvec3f, mutable: true };
    const body: Stmt = {
      kind: "Sequential",
      body: [
        { kind: "Declare", var: colour, init: { kind: "Expr", value: { kind: "NewVector", components: [constF(1), constF(0), constF(0)], type: Tvec3f } } },
        {
          kind: "If",
          cond: { kind: "Gt", lhs: { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 }, rhs: constF(0), type: Tbool },
          then: {
            kind: "Write",
            target: { kind: "LSwizzle", target: { kind: "LVar", var: colour, type: Tvec3f }, comps: ["x"], type: Tf32 },
            value: constF(1),
          },
        },
        {
          kind: "WriteOutput",
          name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [v(colour), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "u_time", type: Tf32 }] },
        { kind: "Entry", entry: testEntry(body) },
      ],
    };
    const out = emitWgsl(m).source;
    expect(out).toContain("@group(0) @binding(0) var<uniform> u_time: f32;");
    expect(out).toContain("if ((u_time > 0.0f)) {");
    expect(out).toContain("colour.x = 1.0f;");
    expect(out).toContain("out.outColor = vec4<f32>(colour, 1.0f);");
  });
});
