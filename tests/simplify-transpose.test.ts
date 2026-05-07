// Algebraic Transpose-propagation pass.
//
// Tests the rules in `passes/simplifyTranspose.ts` and the end-to-end
// effect when combined with `reverseMatrixOps`: the only `transpose`
// WGSL call should remain at genuine leaf reads (`uniform.M.transpose`
// where M itself comes from a UBO).

import { describe, expect, it } from "vitest";
import { simplifyTranspose } from "@aardworx/wombat.shader/passes";
import { compileModule } from "@aardworx/wombat.shader";
import type { EntryDef, Expr, Module, Stmt, Type, Var } from "@aardworx/wombat.shader/ir";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec3f: Type = { kind: "Vector", element: Tf32, dim: 3 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };
const Tmat3f: Type = { kind: "Matrix", element: Tf32, rows: 3, cols: 3 };
const Tmat4f: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };

const wrapInEntry = (body: Stmt, extraInputs: { name: string; type: Type }[] = []): Module => {
  const entry: EntryDef = {
    name: "vsMain", stage: "vertex",
    inputs: extraInputs.map((p, i) => ({
      name: p.name, type: p.type, semantic: p.name,
      decorations: [{ kind: "Location", value: i }],
    })),
    outputs: [],
    arguments: [], returnType: { kind: "Void" }, body, decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
};

describe("simplifyTranspose: algebraic rules", () => {
  it("`T(T(X))` collapses to `X`", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "m", type: Tmat4f };
    const TT: Expr = {
      kind: "Transpose",
      value: { kind: "Transpose", value: M, type: Tmat4f },
      type: Tmat4f,
    };
    const t: Var = { name: "t", type: Tmat4f, mutable: false };
    const body: Stmt = { kind: "Declare", var: t, init: { kind: "Expr", value: TT } };
    const m: Module = wrapInEntry(body);
    const after = simplifyTranspose(m);
    const json = JSON.stringify(after);
    expect(json).not.toContain('"kind":"Transpose"');
    expect(json).toContain('"name":"m"');
  });

  it("`T(MatrixFromCols([cs]))` becomes `MatrixFromRows([cs])`", () => {
    const c: Expr = { kind: "ReadInput", scope: "Input", name: "v", type: Tvec3f };
    const expr: Expr = {
      kind: "Transpose",
      value: { kind: "MatrixFromCols", cols: [c, c, c], type: Tmat3f },
      type: Tmat3f,
    };
    const t: Var = { name: "t", type: Tmat3f, mutable: false };
    const body: Stmt = { kind: "Declare", var: t, init: { kind: "Expr", value: expr } };
    const m = wrapInEntry(body, [{ name: "v", type: Tvec3f }]);
    const after = simplifyTranspose(m);
    const json = JSON.stringify(after);
    expect(json).not.toContain('"kind":"Transpose"');
    expect(json).not.toContain('"kind":"MatrixFromCols"');
    expect(json).toContain('"kind":"MatrixFromRows"');
  });

  it("`T(MatrixFromRows([rs]))` becomes `MatrixFromCols([rs])`", () => {
    const r: Expr = { kind: "ReadInput", scope: "Input", name: "v", type: Tvec3f };
    const expr: Expr = {
      kind: "Transpose",
      value: { kind: "MatrixFromRows", rows: [r, r, r], type: Tmat3f },
      type: Tmat3f,
    };
    const t: Var = { name: "t", type: Tmat3f, mutable: false };
    const body: Stmt = { kind: "Declare", var: t, init: { kind: "Expr", value: expr } };
    const m = wrapInEntry(body, [{ name: "v", type: Tvec3f }]);
    const after = simplifyTranspose(m);
    const json = JSON.stringify(after);
    expect(json).not.toContain('"kind":"Transpose"');
    expect(json).not.toContain('"kind":"MatrixFromRows"');
    expect(json).toContain('"kind":"MatrixFromCols"');
  });

  it("`T(A · B)` survives intact (does NOT distribute)", () => {
    // `simplifyTranspose` does NOT distribute Transpose through
    // MulMatMat. Distribution would force two leaf `transpose(...)`
    // calls in WGSL where `reverseMatrixOps`'s MulMatVec absorb +
    // chain-lowering would otherwise have eliminated the transpose
    // entirely — see `passes/reverseMatrixOps.ts` for why. The
    // `Transpose(MulMatMat(...))` IR survives this pass; downstream
    // `reverseMatrixOps` flips operands and absorbs into the
    // surrounding mat-vec mul.
    const A: Expr = { kind: "ReadInput", scope: "Uniform", name: "A", type: Tmat4f };
    const B: Expr = { kind: "ReadInput", scope: "Uniform", name: "B", type: Tmat4f };
    const AB: Expr = { kind: "MulMatMat", lhs: A, rhs: B, type: Tmat4f };
    const T: Expr = { kind: "Transpose", value: AB, type: Tmat4f };
    const t: Var = { name: "t", type: Tmat4f, mutable: false };
    const body: Stmt = { kind: "Declare", var: t, init: { kind: "Expr", value: T } };
    const m: Module = wrapInEntry(body);
    const after = simplifyTranspose(m);
    const json = JSON.stringify(after);
    // The outer Transpose stays alive, wrapping the MulMatMat.
    expect(json).toMatch(/"kind":"Transpose","value":\{"kind":"MulMatMat"/);
  });
});

describe("simplifyTranspose × reverseMatrixOps end-to-end (WGSL emit)", () => {
  it("`T(uniform M).mul(v)` emits as `M * v` — column-vec form, no `transpose(...)` call", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: {
        kind: "Expr",
        value: {
          kind: "MulMatVec",
          lhs: { kind: "Transpose", value: M, type: Tmat4f },
          rhs: v, type: Tvec4f,
        },
      },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "pos", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "mvp", type: Tmat4f }] },
        { kind: "Entry", entry },
      ],
    };
    const out = compileModule(m, { target: "wgsl" });
    const vs = out.stages.find(s => s.stage === "vertex")!.source;
    expect(vs).toMatch(/mvp\s*\*\s*in\.pos/);
    expect(vs).not.toContain("transpose(");
  });

  it("`T(A · B).mul(v)` collapses to `(B * A) * v` — zero `transpose(...)` calls (full FShade ideal)", () => {
    // DSL: `transpose(A · B) · v`. The full pipeline (reverseMatrixOps
    // operand flips + Transpose absorb on MulMatVec) eliminates every
    // explicit `transpose(...)` call:
    //
    //   IR        : MulMatVec(T(MulMatMat(A, B)), v)
    //   reverse   : inner MulMatMat → MulMatMat(B, A); outer
    //               MulMatVec(T(...), v) absorbs the Transpose →
    //               MulMatVec(MulMatMat(B, A), v)
    //   emit      : `(B * A) * v`
    //
    // Math: WGSL `(B_wgsl · A_wgsl) · v` = `transpose(B_cpu) ·
    // transpose(A_cpu) · v` = `transpose(A_cpu · B_cpu) · v`, which
    // equals the DSL `T(A · B) · v`. ✓
    const A: Expr = { kind: "ReadInput", scope: "Uniform", name: "A", type: Tmat4f };
    const B: Expr = { kind: "ReadInput", scope: "Uniform", name: "B", type: Tmat4f };
    const AB: Expr = { kind: "MulMatMat", lhs: A, rhs: B, type: Tmat4f };
    const T: Expr = { kind: "Transpose", value: AB, type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: {
        kind: "Expr",
        value: { kind: "MulMatVec", lhs: T, rhs: v, type: Tvec4f },
      },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "pos", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "A", type: Tmat4f }, { name: "B", type: Tmat4f }] },
        { kind: "Entry", entry },
      ],
    };
    const out = compileModule(m, { target: "wgsl" });
    const vs = out.stages.find(s => s.stage === "vertex")!.source;
    expect(vs).not.toContain("transpose(");
    expect(vs).toMatch(/\(\s*B\s*\*\s*A\s*\)\s*\*\s*in\.pos/);
  });
});
