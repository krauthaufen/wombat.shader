// Regressions found porting the volume viewer's shaders to F# (the Fable
// shader plugin emits helper functions, structs, constants and loops that
// the TS frontend rarely produces).

import { describe, expect, it } from "vitest";
import { Tbool, Tf32, Ti32, Vec, type Expr, type Module, type Stmt, type Type, type Var } from "@aardworx/wombat.shader/ir";
import { cse, inlinePass, legaliseTypes } from "@aardworx/wombat.shader/passes";
import { emitWgsl } from "@aardworx/wombat.shader/wgsl";
import { compileModule } from "@aardworx/wombat.shader";

const v = (x: Var): Expr => ({ kind: "Var", var: x, type: x.type });
const cf = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const ci = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const lv = (x: Var) => ({ kind: "LVar" as const, var: x, type: x.type });
const seq = (...body: Stmt[]): Stmt => ({ kind: "Sequential", body });
const compute = (body: Stmt, extra: Module["values"] = []): Module => ({
  types: [],
  values: [...extra, {
    kind: "Entry",
    entry: { name: "main", stage: "compute", inputs: [], outputs: [], arguments: [], returnType: { kind: "Void" }, body,
             decorations: [{ kind: "WorkgroupSize", x: 1 }] },
  }],
});
const entryBody = (m: Module): Stmt => {
  for (const x of m.values) if (x.kind === "Entry") return x.entry.body;
  throw new Error("no entry");
};
const text = (s: Stmt) => JSON.stringify(s);

describe("cse / copy propagation with mutable variables", () => {
  it("does not hoist a mutable read out of a loop condition", () => {
    const accS: Var = { name: "acc", type: Tf32, mutable: true };
    const lt = (): Expr => ({ kind: "Lt", lhs: v(accS), rhs: cf(4), type: Tbool });
    const body = seq(
      { kind: "Declare", var: accS, init: { kind: "Expr", value: cf(0) } },
      { kind: "While", cond: lt(), body: { kind: "Write", target: lv(accS), value: { kind: "Add", lhs: v(accS), rhs: cf(1), type: Tf32 } } },
      { kind: "If", cond: lt(), then: { kind: "Write", target: lv(accS), value: cf(-1) } },
    );
    const out = text(entryBody(cse(compute(body))));
    expect(out).not.toContain("_cse");
  });

  it("keeps a temp holding a mutable variable's old value (swap)", () => {
    const x: Var = { name: "x", type: Ti32, mutable: true };
    const y: Var = { name: "y", type: Ti32, mutable: true };
    const t: Var = { name: "t", type: Ti32, mutable: false };
    const body = seq(
      { kind: "Declare", var: x, init: { kind: "Expr", value: ci(3) } },
      { kind: "Declare", var: y, init: { kind: "Expr", value: ci(5) } },
      { kind: "Declare", var: t, init: { kind: "Expr", value: v(x) } },
      { kind: "Write", target: lv(x), value: v(y) },
      { kind: "Write", target: lv(y), value: v(t) },
    );
    const out = entryBody(inlinePass(compute(body)));
    expect(text(out)).toContain('"name":"t"');
  });
});

describe("wgsl emission", () => {
  const Tvec3: Type = Vec(Tf32, 3);
  const cell: Type = { kind: "Struct", name: "Cell", fields: [
    { name: "lo", type: Vec(Tf32, 4) },
    { name: "faces", type: { kind: "Array", element: Vec(Tf32, 4), length: 6 } },
  ] };
  const table: Type = { kind: "Array", element: Tvec3, length: 2 };
  const cells: Var = { name: "cells", type: { kind: "Array", element: cell, length: "runtime" }, mutable: true };
  const out: Var = { name: "out", type: { kind: "Array", element: Tf32, length: "runtime" }, mutable: true };
  const TABLE: Var = { name: "TABLE", type: table, mutable: false };
  const vec = (a: number, b: number, c: number): Expr => ({ kind: "NewVector", components: [cf(a), cf(b), cf(c)], type: Tvec3 });

  const m: Module = compute(
    { kind: "Write", target: { kind: "LItem", target: lv(out), index: ci(0), type: Tf32 },
      value: { kind: "Add",
        lhs: { kind: "VecSwizzle", value: { kind: "Field", target: { kind: "Item", target: v(cells), index: ci(0), type: cell }, name: "lo", type: Vec(Tf32, 4) }, comps: ["x"], type: Tf32 },
        rhs: { kind: "VecSwizzle", value: { kind: "Item", target: v(TABLE), index: ci(1), type: Tvec3 }, comps: ["y"], type: Tf32 },
        type: Tf32 } },
    [
      { kind: "StorageBuffer", binding: { group: 0, slot: 0 }, name: "cells", layout: cells.type, access: "read" },
      { kind: "StorageBuffer", binding: { group: 0, slot: 0 }, name: "out", layout: out.type, access: "read_write" },
      { kind: "Constant", varType: table, name: "TABLE", init: { kind: "ArrayLiteral", arrayType: table, values: [vec(0, 1, 2), vec(1e30, 1, 1)] } },
    ],
  );

  it("declares structs used only by bindings, and module constants", () => {
    const src = emitWgsl(m).source;
    expect(src).toMatch(/struct Cell \{\s+lo: vec4<f32>,\s+faces: array<vec4<f32>, 6>,/);
    expect(src).toContain("const TABLE: array<vec3<f32>, 2> = array<vec3<f32>, 2>(");
    expect(src).toContain("1e+30f");
  });

  it("declares a binding shared by composed stages once", () => {
    const twice: Module = { ...m, values: [m.values[0]!, ...m.values] };
    const src = compileModule(twice, { target: "wgsl" }).stages[0]!.source;
    expect(src.match(/var<storage, read> cells/g)).toHaveLength(1);
  });

  it("loads texels without the sampler", () => {
    const s: Type = { kind: "Sampler", target: "3D", sampled: { kind: "Float" }, comparison: false };
    const load: Expr = {
      kind: "CallIntrinsic",
      op: { name: "texelFetch", pure: true, samplerBinding: true, emit: { glsl: "texelFetch", wgsl: "textureLoad" }, returnTypeOf: () => Vec(Tf32, 4) },
      args: [{ kind: "ReadInput", scope: "Uniform", name: "atlas", type: s },
             { kind: "NewVector", components: [ci(0), ci(0), ci(0)], type: Vec(Ti32, 3) }, ci(0)],
      type: Vec(Tf32, 4),
    };
    const mm = compute({ kind: "Write", target: { kind: "LItem", target: lv(out), index: ci(0), type: Tf32 },
                         value: { kind: "VecSwizzle", value: load, comps: ["x"], type: Tf32 } },
      [ { kind: "Sampler", binding: { group: 0, slot: 0 }, name: "atlas", type: s },
        { kind: "StorageBuffer", binding: { group: 0, slot: 0 }, name: "out", layout: out.type, access: "read_write" } ]);
    const src = emitWgsl(legaliseTypes(mm, "wgsl")).source;
    expect(src).toContain("textureLoad(atlas_view, vec3<i32>(0i, 0i, 0i), 0i)");
  });
});
