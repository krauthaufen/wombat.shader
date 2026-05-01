// Tests for the optimisation passes: foldConstants, dce, cse, inlinePass.
// Each test hand-builds a small IR, runs the pass, and asserts on the
// resulting structure.

import { describe, expect, it } from "vitest";
import {
  Tbool,
  Tf32,
  Ti32,
  Tvoid,
  type Expr,
  type Module,
  type Stmt,
  type Var,
} from "@aardworx/wombat.shader/ir";
import {
  cse,
  dce,
  foldConstants,
  foldExpr,
  inlinePass,
  isPure,
} from "@aardworx/wombat.shader/passes";

const constI = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const v = (va: Var): Expr => ({ kind: "Var", var: va, type: va.type });

function moduleWith(body: Stmt): Module {
  return {
    types: [],
    values: [{
      kind: "Entry",
      entry: {
        name: "main",
        stage: "fragment",
        inputs: [],
        outputs: [],
        arguments: [],
        returnType: Tvoid,
        body,
        decorations: [],
      },
    }],
  };
}

function bodyOf(m: Module): Stmt {
  for (const v of m.values) if (v.kind === "Entry") return v.entry.body;
  throw new Error("no entry");
}

// ─── foldConstants ───────────────────────────────────────────────────

describe("foldConstants", () => {
  it("integer arithmetic", () => {
    const e = foldExpr({
      kind: "Add",
      lhs: { kind: "Mul", lhs: constI(2), rhs: constI(3), type: Ti32 },
      rhs: constI(4),
      type: Ti32,
    });
    expect(e.kind).toBe("Const");
    expect(e.kind === "Const" && e.value.kind === "Int" && e.value.value).toBe(10);
  });

  it("float division", () => {
    const e = foldExpr({ kind: "Div", lhs: constF(7), rhs: constF(2), type: Tf32 });
    expect(e.kind === "Const" && e.value.kind === "Float" && e.value.value).toBe(3.5);
  });

  it("Conditional with constant cond", () => {
    const e = foldExpr({
      kind: "Conditional",
      cond: { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool },
      ifTrue: constI(1),
      ifFalse: constI(2),
      type: Ti32,
    });
    expect(e.kind === "Const" && e.value.kind === "Int" && e.value.value).toBe(1);
  });

  it("And short-circuits on false", () => {
    const e = foldExpr({
      kind: "And",
      lhs: { kind: "Const", value: { kind: "Bool", value: false }, type: Tbool },
      rhs: { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool },
      type: Tbool,
    });
    expect(e.kind === "Const" && e.value.kind === "Bool" && e.value.value).toBe(false);
  });

  it("comparison", () => {
    const e = foldExpr({ kind: "Lt", lhs: constI(2), rhs: constI(3), type: Tbool });
    expect(e.kind === "Const" && e.value.kind === "Bool" && e.value.value).toBe(true);
  });

  it("module-level fold", () => {
    const x: Var = { name: "x", type: Ti32, mutable: false };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: { kind: "Add", lhs: constI(1), rhs: constI(2), type: Ti32 } } },
      ],
    });
    const out = foldConstants(m);
    const decl = (bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>).body[0]!;
    expect(decl.kind).toBe("Declare");
    if (decl.kind === "Declare") {
      const init = decl.init!;
      expect(init.kind).toBe("Expr");
      if (init.kind === "Expr") {
        expect(init.value.kind).toBe("Const");
      }
    }
  });
});

// ─── dce ──────────────────────────────────────────────────────────────

describe("dce", () => {
  it("drops dead immutable declarations with pure init", () => {
    const x: Var = { name: "x", type: Ti32, mutable: false };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constI(42) } },
      ],
    });
    const out = dce(m);
    expect(bodyOf(out).kind).toBe("Nop");
  });

  it("keeps used declarations", () => {
    const x: Var = { name: "x", type: Ti32, mutable: false };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constI(42) } },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = dce(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    expect(seq.kind).toBe("Sequential");
    expect(seq.body.length).toBe(2);
  });

  it("constant-true if collapses to then branch", () => {
    const m = moduleWith({
      kind: "If",
      cond: { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool },
      then: {
        kind: "WriteOutput", name: "outVal",
        value: { kind: "Expr", value: constI(1) },
      },
      else: {
        kind: "WriteOutput", name: "outVal",
        value: { kind: "Expr", value: constI(2) },
      },
    });
    const out = dce(m);
    const body = bodyOf(out);
    expect(body.kind).toBe("WriteOutput");
    if (body.kind === "WriteOutput") {
      expect(body.value.kind).toBe("Expr");
      if (body.value.kind === "Expr") {
        expect(body.value.value.kind === "Const" && body.value.value.value.kind === "Int" && body.value.value.value.value).toBe(1);
      }
    }
  });
});

// ─── cse ──────────────────────────────────────────────────────────────

describe("cse", () => {
  it("hoists a repeated subexpression into a temp", () => {
    const x: Var = { name: "x", type: Tf32, mutable: true };
    // r = (a*b + c) + (a*b + c)  → declare _cse0 = a*b+c; r = _cse0 + _cse0
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    const b: Expr = { kind: "ReadInput", scope: "Uniform", name: "b", type: Tf32 };
    const c: Expr = { kind: "ReadInput", scope: "Uniform", name: "c", type: Tf32 };
    const ab: Expr = { kind: "Mul", lhs: a, rhs: b, type: Tf32 };
    const abc: Expr = { kind: "Add", lhs: ab, rhs: c, type: Tf32 };
    const sum: Expr = { kind: "Add", lhs: abc, rhs: abc, type: Tf32 };

    const m = moduleWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: sum } },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = cse(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    expect(seq.kind).toBe("Sequential");
    // First stmt should be a Declare of a `_cse*` Var.
    const first = seq.body[0]!;
    expect(first.kind).toBe("Declare");
    if (first.kind === "Declare") {
      expect(first.var.name).toMatch(/^_cse/);
    }
  });

  it("doesn't touch single-use expressions", () => {
    const x: Var = { name: "x", type: Tf32, mutable: false };
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    const b: Expr = { kind: "ReadInput", scope: "Uniform", name: "b", type: Tf32 };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: { kind: "Add", lhs: a, rhs: b, type: Tf32 } } },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = cse(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    // No _cse declarations introduced — still 2 statements.
    expect(seq.body.length).toBe(2);
    expect(seq.body[0]!.kind).toBe("Declare");
    if (seq.body[0]!.kind === "Declare") {
      expect(seq.body[0]!.var.name).toBe("x");
    }
  });
});

// ─── inline ───────────────────────────────────────────────────────────

describe("inline", () => {
  it("inlines a single-return user function", () => {
    const a: Var = { name: "a", type: Tf32, mutable: false };
    const b: Var = { name: "b", type: Tf32, mutable: false };
    // fn add(a: f32, b: f32): f32 { return a + b; }
    const fn = {
      kind: "Function" as const,
      signature: {
        name: "addFn",
        returnType: Tf32,
        parameters: [
          { name: "a", type: Tf32, modifier: "in" as const },
          { name: "b", type: Tf32, modifier: "in" as const },
        ],
      },
      body: {
        kind: "ReturnValue" as const,
        value: { kind: "Add" as const, lhs: v(a), rhs: v(b), type: Tf32 },
      },
    };

    const r: Var = { name: "r", type: Tf32, mutable: false };
    const inA: Expr = { kind: "ReadInput", scope: "Input", name: "in_a", type: Tf32 };
    const inB: Expr = { kind: "ReadInput", scope: "Input", name: "in_b", type: Tf32 };
    const m: Module = {
      types: [],
      values: [
        fn,
        {
          kind: "Entry",
          entry: {
            name: "main",
            stage: "fragment",
            inputs: [],
            outputs: [],
            arguments: [],
            returnType: Tvoid,
            body: {
              kind: "Sequential",
              body: [
                {
                  kind: "Declare",
                  var: r,
                  init: {
                    kind: "Expr",
                    value: {
                      kind: "Call",
                      fn: { id: "addFn", signature: fn.signature, pure: true },
                      args: [inA, inB],
                      type: Tf32,
                    },
                  },
                },
                {
                  kind: "WriteOutput", name: "outVal",
                  value: { kind: "Expr", value: v(r) },
                },
              ],
            },
            decorations: [],
          },
        },
      ],
    };
    const out = inlinePass(m);
    const entry = out.values.find((x) => x.kind === "Entry")!;
    if (entry.kind !== "Entry") throw new Error();
    const seq = entry.entry.body as Extract<Stmt, { kind: "Sequential" }>;
    // After inlining + copy-prop, the Declare for r would be propagated
    // (init is now an Add, which isn't trivially copyable, so it stays).
    const decl = seq.body.find((s) => s.kind === "Declare");
    expect(decl).toBeTruthy();
    if (decl?.kind === "Declare" && decl.init?.kind === "Expr") {
      expect(decl.init.value.kind).toBe("Add");
    }
  });

  it("copy-propagation drops trivial bindings", () => {
    const x: Var = { name: "x", type: Tf32, mutable: false };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        // let x = u_time
        {
          kind: "Declare",
          var: x,
          init: { kind: "Expr", value: { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 } },
        },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = inlinePass(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    // Declare of x should have been dropped; the WriteOutput now references u_time directly.
    expect(seq.body.length).toBe(1);
    expect(seq.body[0]!.kind).toBe("WriteOutput");
    if (seq.body[0]!.kind === "WriteOutput" && seq.body[0]!.value.kind === "Expr") {
      expect(seq.body[0]!.value.value.kind).toBe("ReadInput");
    }
  });
});

// ─── integration ──────────────────────────────────────────────────────

describe("pipeline: fold → dce", () => {
  it("dead Declare with constant init disappears after both passes", () => {
    const x: Var = { name: "x", type: Ti32, mutable: false };
    const m = moduleWith({
      kind: "Sequential",
      body: [
        {
          kind: "Declare",
          var: x,
          init: {
            kind: "Expr",
            value: { kind: "Add", lhs: constI(1), rhs: constI(2), type: Ti32 },
          },
        },
      ],
    });
    const folded = foldConstants(m);
    const cleaned = dce(folded);
    expect(bodyOf(cleaned).kind).toBe("Nop");
  });
});

// ─── isPure smoke ─────────────────────────────────────────────────────

describe("isPure", () => {
  it("Var/Const/Add/Read are pure", () => {
    expect(isPure(constI(1))).toBe(true);
    expect(isPure({ kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 })).toBe(true);
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    expect(isPure({ kind: "Add", lhs: a, rhs: a, type: Tf32 })).toBe(true);
  });
  it("CallIntrinsic with pure=false is not pure", () => {
    const impure: Expr = {
      kind: "CallIntrinsic",
      op: { name: "imageStore", pure: false, emit: { glsl: "imageStore", wgsl: "textureStore" }, returnTypeOf: () => Tvoid },
      args: [],
      type: Tvoid,
    };
    expect(isPure(impure)).toBe(false);
  });
});
