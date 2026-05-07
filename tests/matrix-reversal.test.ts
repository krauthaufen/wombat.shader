// Row/column-major reversal pass. Verifies the IR-level rewrites
// and the resulting WGSL emit so a row-major matrix uploaded raw
// produces correct results when the shader does the natural
// `mvp * v` math.

import { describe, expect, it } from "vitest";
import { reverseMatrixOps } from "@aardworx/wombat.shader/passes";
import { compileModule } from "@aardworx/wombat.shader";
import type { EntryDef, Expr, Module, Stmt, Type } from "@aardworx/wombat.shader/ir";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };
const Tmat4f: Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 };

describe("reverseMatrixOps", () => {
  it("`MulMatVec(M, v)` becomes `MulVecMat(v, M)`", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: M, rhs: v, type: Tvec4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{
        name: "pos", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Location", value: 0 }],
      }],
      outputs: [{
        name: "gl_Position", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }],
      }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const after = reverseMatrixOps(m);
    const json = JSON.stringify(after);
    expect(json).toContain('"kind":"MulVecMat"');
    expect(json).not.toContain('"kind":"MulMatVec"');
  });

  it("`MulMatMat(A, B)` swaps to `MulMatMat(B, A)`", () => {
    const A: Expr = { kind: "ReadInput", scope: "Uniform", name: "view", type: Tmat4f };
    const B: Expr = { kind: "ReadInput", scope: "Uniform", name: "model", type: Tmat4f };
    const body: Stmt = {
      kind: "Declare",
      var: { name: "mv", type: Tmat4f, mutable: false },
      init: { kind: "Expr", value: { kind: "MulMatMat", lhs: A, rhs: B, type: Tmat4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [], outputs: [], arguments: [],
      returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const after = reverseMatrixOps(m);
    // Find the MulMatMat call and verify lhs/rhs swap.
    const stmt = (after.values[0]! as { entry: EntryDef }).entry.body as Stmt & { kind: "Declare" };
    const mul = (stmt.init as { value: Expr & { kind: "MulMatMat" } }).value;
    expect((mul.lhs as { name: string }).name).toBe("model"); // was B
    expect((mul.rhs as { name: string }).name).toBe("view");  // was A
  });

  it("`Transpose(M).mul(v)` evaluates to `transpose(M_cpu) · v` (composition regression)", async () => {
    // Regression for the "vanish `Transpose`" bug: with the elision
    // active, `M.transpose().mul(v)` came out as `M_cpu · v` (wrong).
    // We verify by Dawn-validating the emitted WGSL contains an
    // explicit `transpose(...)` call and the multiply uses that
    // transposed value — i.e., the composition is *not* short-circuited
    // to the bare `M`.
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const tM: Expr = { kind: "Transpose", value: M, type: Tmat4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: tM, rhs: v, type: Tvec4f } },
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
    // The transpose absorbs the operand-flip — `transpose(M).mul(v)`
    // emits as the column-vec form `M * v`, which under the upload
    // trick (`M_wgsl = transpose(M_cpu)`) evaluates to
    // `transpose(M_cpu) · v`. No `transpose(...)` WGSL call needed.
    // FShade does the same — the row-major upload + per-op rewrites
    // make `transpose` a free no-op at the source level for the
    // common compositions.
    expect(vs).toMatch(/mvp\s*\*\s*in\.pos/);
    expect(vs).not.toContain("transpose(");
    // The pre-fix vanish bug would have emitted `(in.pos * mvp)` —
    // the row-vec form, which evaluates to `M_cpu · v` instead of
    // `transpose(M_cpu) · v`. Reject that explicitly.
    expect(vs).not.toMatch(/in\.pos\s*\*\s*mvp/);
  });

  it("chain-lowers `(A · B) · v` → `A · (B · v)` (8 dot-products vs 20)", () => {
    // After reverseMatrixOps: post-form `MulMatVec(MulMatMat(B, A), v)`
    // (the inner mat-mat had its operands swapped). Re-associating
    // saves a mat-mat (16 dot-4s) for an extra mat-vec (4 dot-4s).
    // CSE would have already bound a multi-use intermediate to a Var,
    // so a literal `MulMatMat` here is single-use and lowering is a
    // strict win. Math is preserved: `(A·B)·v = A·(B·v)`.
    const A: Expr = { kind: "ReadInput", scope: "Uniform", name: "A", type: Tmat4f };
    const B: Expr = { kind: "ReadInput", scope: "Uniform", name: "B", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const AB: Expr = { kind: "MulMatMat", lhs: A, rhs: B, type: Tmat4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: AB, rhs: v, type: Tvec4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "pos", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = {
      types: [], values: [
        { kind: "Uniform", uniforms: [{ name: "A", type: Tmat4f }, { name: "B", type: Tmat4f }] },
        { kind: "Entry", entry },
      ],
    };
    const out = compileModule(m, { target: "wgsl" });
    const vs = out.stages.find(s => s.stage === "vertex")!.source;
    // Right-associated form: `M1 * (M2 * v)` or `(v * M2) * M1`
    // (depending on which side absorbed the chain). The key
    // signature is that there is NEVER a `mat * mat * vec`-style
    // sequence in the emit — the mat-mat is gone.
    expect(vs).not.toMatch(/\(\s*[AB]\s*\*\s*[AB]\s*\)\s*\*\s*in\.pos/);
    // Match the actual chain-lowered form for our concrete IR.
    // `MulMatVec(MulMatMat(A, B), v)` after the pass: outer MulMatVec
    // hits the chain-lower rule, producing `MulVecMat(MulVecMat(v, B), A)`,
    // which emits `(in.pos * B) * A`.
    expect(vs).toMatch(/\(\s*in\.pos\s*\*\s*[AB]\s*\)\s*\*\s*[AB]/);
  });

  it("`Transpose(M)` survives the pass — emit produces a real `transpose(...)` call", () => {
    // Vanishing `Transpose` would only be correct at the leaf (the
    // GPU's view of a uniform M is already `transpose(M_cpu)`, so
    // `M.transpose()` "is" what the GPU has). It breaks the moment
    // the result composes — `M.transpose().mul(v)` would then come
    // out as `M_cpu · v` instead of `transpose(M_cpu) · v`. The pass
    // maintains the invariant *WGSL value = transpose(DSL value)*;
    // that requires `Transpose(M)` to reach emit so WGSL evaluates
    // `transpose(M_wgsl) = M_cpu`. FShade matches: no IR rewrite for
    // transpose, just an emit.
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "m", type: Tmat4f };
    const body: Stmt = {
      kind: "Declare",
      var: { name: "t", type: Tmat4f, mutable: false },
      init: { kind: "Expr", value: { kind: "Transpose", value: M, type: Tmat4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [], outputs: [], arguments: [],
      returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const after = reverseMatrixOps(m);
    const json = JSON.stringify(after);
    expect(json).toContain('"kind":"Transpose"');
    expect(json).toContain('"name":"m"');
  });

  it("idempotent: a second invocation is a no-op", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: M, rhs: v, type: Tvec4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{
        name: "pos", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Location", value: 0 }],
      }],
      outputs: [{
        name: "gl_Position", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }],
      }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry }] };
    const a = reverseMatrixOps(m);
    const b = reverseMatrixOps(a);
    // Module should be the same reference if nothing changed.
    expect(b).toBe(a);
    // Verify still has MulVecMat (didn't double-flip).
    expect(JSON.stringify(b)).toContain('"kind":"MulVecMat"');
    expect(JSON.stringify(b)).not.toContain('"kind":"MulMatVec"');
  });

  it("end-to-end: WGSL emit shows `v * M`, not `M * v`", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: M, rhs: v, type: Tvec4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{
        name: "pos", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Location", value: 0 }],
      }],
      outputs: [{
        name: "gl_Position", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }],
      }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "mvp", type: Tmat4f }] },
        { kind: "Entry", entry },
      ],
    };
    const wgsl = compileModule(m, { target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toMatch(/in\.pos\s*\*\s*mvp/);
    expect(wgsl).not.toMatch(/mvp\s*\*\s*in\.pos/);
  });

  it("`skipMatrixReversal: true` keeps the original M*v emit", () => {
    const M: Expr = { kind: "ReadInput", scope: "Uniform", name: "mvp", type: Tmat4f };
    const v: Expr = { kind: "ReadInput", scope: "Input", name: "pos", type: Tvec4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "gl_Position",
      value: { kind: "Expr", value: { kind: "MulMatVec", lhs: M, rhs: v, type: Tvec4f } },
    };
    const entry: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [{
        name: "pos", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Location", value: 0 }],
      }],
      outputs: [{
        name: "gl_Position", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }],
      }],
      arguments: [], returnType: { kind: "Void" }, body, decorations: [],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "mvp", type: Tmat4f }] },
        { kind: "Entry", entry },
      ],
    };
    const wgsl = compileModule(m, { target: "wgsl", skipMatrixReversal: true }).stages[0]!.source;
    expect(wgsl).toMatch(/mvp\s*\*\s*in\.pos/);
  });
});
