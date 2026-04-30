// Coverage tests for IR node kinds we didn't exercise in the
// hello-triangle / control-flow tests.

import { describe, expect, it } from "vitest";
import {
  Mat,
  Tbool,
  Tf32,
  Ti32,
  Tu32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
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

const constI = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const constU = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: false, value: n }, type: Tu32 });
const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const v = (va: Var): Expr => ({ kind: "Var", var: va, type: va.type });

function frag(body: Stmt): EntryDef {
  return {
    name: "fsMain", stage: "fragment", inputs: [],
    outputs: [
      { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
    ],
    arguments: [], returnType: Tvoid, body, decorations: [],
  };
}

function moduleWith(values: Module["values"]): Module {
  return { types: [], values };
}

// ─── matrix nodes ────────────────────────────────────────────────────

describe("matrix emit", () => {
  it("MulMatVec, MulMatMat, Transpose", () => {
    const m: Var = { name: "m", type: Tmat4f, mutable: false };
    const n: Var = { name: "n", type: Tmat4f, mutable: false };
    const p: Var = { name: "p", type: Tvec4f, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: m, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Uniform", name: "u_m", type: Tmat4f } } },
        { kind: "Declare", var: n, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Uniform", name: "u_n", type: Tmat4f } } },
        {
          kind: "Declare", var: p, init: {
            kind: "Expr",
            value: {
              kind: "MulMatVec",
              lhs: { kind: "MulMatMat", lhs: v(m), rhs: { kind: "Transpose", value: v(n), type: Tmat4f }, type: Tmat4f },
              rhs: { kind: "ReadInput", scope: "Input", name: "a_pos", type: Tvec4f },
              type: Tvec4f,
            },
          },
        },
        {
          kind: "WriteOutput", name: "outColor",
          value: { kind: "Expr", value: v(p) },
        },
      ],
    };
    const mod = moduleWith([
      { kind: "Uniform", uniforms: [{ name: "u_m", type: Tmat4f }, { name: "u_n", type: Tmat4f }] },
      { kind: "Entry", entry: frag(body) },
    ]);
    const glsl = emitGlsl(mod).source;
    expect(glsl).toContain("(m * transpose(n))");
    expect(glsl).toContain("((m * transpose(n)) * a_pos)");
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("(m * transpose(n))");
  });
});

// ─── vector nodes ────────────────────────────────────────────────────

describe("vector emit", () => {
  it("Dot, Cross, Length, swizzle", () => {
    const a: Var = { name: "a", type: Tvec3f, mutable: false };
    const b: Var = { name: "b", type: Tvec3f, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: a, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Input", name: "v_a", type: Tvec3f } } },
        { kind: "Declare", var: b, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Input", name: "v_b", type: Tvec3f } } },
        {
          kind: "WriteOutput", name: "outColor",
          value: {
            kind: "Expr",
            value: {
              kind: "NewVector",
              components: [
                { kind: "Dot", lhs: v(a), rhs: v(b), type: Tf32 },
                { kind: "Length", value: { kind: "Cross", lhs: v(a), rhs: v(b), type: Tvec3f }, type: Tf32 },
                { kind: "VecSwizzle", value: v(a), comps: ["x"], type: Tf32 },
                constF(1),
              ],
              type: Tvec4f,
            },
          },
        },
      ],
    };
    const mod = moduleWith([{ kind: "Entry", entry: frag(body) }]);
    const glsl = emitGlsl(mod).source;
    expect(glsl).toContain("dot(a, b)");
    expect(glsl).toContain("length(cross(a, b))");
    expect(glsl).toContain("a.x");
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("dot(a, b)");
    expect(wgsl).toContain("length(cross(a, b))");
  });
});

// ─── unsigned integers / mixed types ─────────────────────────────────

describe("integer types", () => {
  it("u32 literal suffixes differ between backends", () => {
    const x: Var = { name: "x", type: Tu32, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constU(7) } },
        {
          kind: "WriteOutput", name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [constF(0), constF(0), constF(0), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const mod = moduleWith([{ kind: "Entry", entry: frag(body) }]);
    expect(emitGlsl(mod).source).toContain("uint x = 7u;");
    // x is immutable in our IR (mutable: false) → WGSL `let`.
    expect(emitWgsl(mod).source).toContain("let x: u32 = 7u;");
  });
});

// ─── control flow extras ─────────────────────────────────────────────

describe("control flow extras", () => {
  it("Discard in fragment", () => {
    const body: Stmt = {
      kind: "Sequential", body: [
        {
          kind: "If",
          cond: { kind: "Lt", lhs: { kind: "ReadInput", scope: "Uniform", name: "u_alpha", type: Tf32 }, rhs: constF(0.5), type: Tbool },
          then: { kind: "Discard" },
        },
        {
          kind: "WriteOutput", name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [constF(1), constF(1), constF(1), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const mod = moduleWith([
      { kind: "Uniform", uniforms: [{ name: "u_alpha", type: Tf32 }] },
      { kind: "Entry", entry: frag(body) },
    ]);
    expect(emitGlsl(mod).source).toContain("discard;");
    expect(emitWgsl(mod).source).toContain("discard;");
  });

  it("DoWhile lowers correctly per backend", () => {
    const i: Var = { name: "i", type: Ti32, mutable: true };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: i, init: { kind: "Expr", value: constI(0) } },
        {
          kind: "DoWhile",
          cond: { kind: "Lt", lhs: v(i), rhs: constI(10), type: Tbool },
          body: { kind: "Increment", target: { kind: "LVar", var: i, type: Ti32 }, prefix: false },
        },
        {
          kind: "WriteOutput", name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector", components: [constF(0), constF(0), constF(0), constF(1)], type: Tvec4f } },
        },
      ],
    };
    const mod = moduleWith([{ kind: "Entry", entry: frag(body) }]);
    // GLSL keeps do-while.
    expect(emitGlsl(mod).source).toMatch(/do\s*\{[\s\S]*?\}\s*while/);
    // WGSL lowers to loop + break. Note WGSL int literal suffix "i".
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("loop {");
    expect(wgsl).toContain("if (!((i < 10i))) { break; }");
  });

  it("Switch with cases", () => {
    const x: Var = { name: "x", type: Ti32, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constI(2) } },
        {
          kind: "Switch", value: v(x),
          cases: [
            { literal: { kind: "Int", signed: true, value: 1 }, body: { kind: "Discard" } },
            { literal: { kind: "Int", signed: true, value: 2 }, body: { kind: "Break" } },
          ],
          default: {
            kind: "WriteOutput", name: "outColor",
            value: { kind: "Expr", value: { kind: "NewVector", components: [constF(0.5), constF(0.5), constF(0.5), constF(1)], type: Tvec4f } },
          },
        },
      ],
    };
    const mod = moduleWith([{ kind: "Entry", entry: frag(body) }]);
    const glsl = emitGlsl(mod).source;
    expect(glsl).toContain("switch (x)");
    expect(glsl).toContain("case 1:");
    expect(glsl).toContain("default:");
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("switch (x)");
    expect(wgsl).toContain("case 1i: {");
    expect(wgsl).toContain("default: {");
  });
});

// ─── samplers ────────────────────────────────────────────────────────

describe("samplers", () => {
  it("GLSL emits combined sampler2D, WGSL emits separate sampler+texture", () => {
    const TextureIntrinsic = {
      name: "texture", pure: true,
      emit: { glsl: "texture", wgsl: "textureSample" },
      returnTypeOf: () => Tvec4f,
      samplerBinding: true,
    };
    const samplerType: Type = { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false };
    const textureType: Type = { kind: "Texture", target: "2D", sampled: { kind: "Float" }, arrayed: false, multisampled: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        {
          kind: "WriteOutput", name: "outColor",
          value: {
            kind: "Expr",
            value: {
              kind: "CallIntrinsic", op: TextureIntrinsic, args: [
                { kind: "ReadInput", scope: "Uniform", name: "u_tex", type: samplerType },
                { kind: "ReadInput", scope: "Input", name: "v_uv", type: Vec(Tf32, 2) },
              ], type: Tvec4f,
            },
          },
        },
      ],
    };
    const mod: Module = {
      types: [], values: [
        // GLSL: sampler2D directly. WGSL: separate.
        { kind: "Sampler", binding: { group: 0, slot: 0 }, name: "u_tex", type: samplerType },
        { kind: "Sampler", binding: { group: 0, slot: 1 }, name: "u_tex_view", type: textureType },
        { kind: "Entry", entry: frag(body) },
      ],
    };
    const glsl = emitGlsl(mod).source;
    expect(glsl).toContain("uniform sampler2D u_tex;");
    expect(glsl).toContain("texture(u_tex, v_uv)");
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("@group(0) @binding(0) var u_tex: sampler;");
    expect(wgsl).toContain("@group(0) @binding(1) var u_tex_view: texture_2d<f32>;");
  });
});

// ─── compute stage ───────────────────────────────────────────────────

describe("compute stage", () => {
  it("emits @compute @workgroup_size in WGSL", () => {
    const body: Stmt = { kind: "Nop" };
    const entry: EntryDef = {
      name: "csMain", stage: "compute",
      inputs: [], outputs: [], arguments: [
        { name: "gid", type: Vec(Tu32, 3), semantic: "Position", decorations: [{ kind: "Builtin", value: "global_invocation_id" }] },
      ],
      returnType: Tvoid, body,
      decorations: [{ kind: "WorkgroupSize", x: 8, y: 8, z: 1 }],
    };
    const mod: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const wgsl = emitWgsl(mod).source;
    expect(wgsl).toContain("@compute @workgroup_size(8, 8, 1)");
    expect(wgsl).toContain("@builtin(global_invocation_id) gid: vec3<u32>");
    expect(emitWgsl(mod).meta.workgroupSize).toEqual([8, 8, 1]);
  });
});
