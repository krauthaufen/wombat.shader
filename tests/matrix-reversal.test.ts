// Row/column-major reversal pass. Verifies the IR-level rewrites
// and the resulting WGSL emit so a row-major matrix uploaded raw
// produces correct results when the shader does the natural
// `mvp * v` math.

import { describe, expect, it } from "vitest";
import {
  reverseMatrixOps,
  type Holes,
} from "@aardworx/wombat.shader-passes";
import { compileModule } from "@aardworx/wombat.shader-runtime";
import type { EntryDef, Expr, Module, Stmt, Type } from "@aardworx/wombat.shader-ir";

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

  it("`Transpose(M)` becomes the underlying expression (a no-op)", () => {
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
    expect(json).not.toContain('"kind":"Transpose"');
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
