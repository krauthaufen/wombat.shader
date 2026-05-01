// Edge-case tests for the optimisation passes.

import { describe, expect, it } from "vitest";
import {
  Tbool,
  Tf32,
  Ti32,
  Tvoid,
  Vec,
  type Expr,
  type IntrinsicRef,
  type Module,
  type Stmt,
  type Type,
  type Var,
} from "@aardworx/wombat.shader/ir";
import { cse, dce, foldExpr, hasSideEffects, inlinePass, isPure } from "@aardworx/wombat.shader/passes";

const Tvec3f: Type = Vec(Tf32, 3);

const constI = (n: number, signed = true): Expr => ({
  kind: "Const", value: { kind: "Int", signed, value: n }, type: Ti32,
});
const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const constB = (b: boolean): Expr => ({ kind: "Const", value: { kind: "Bool", value: b }, type: Tbool });
const v = (va: Var): Expr => ({ kind: "Var", var: va, type: va.type });

const ImageStore: IntrinsicRef = {
  name: "imageStore", pure: false,
  emit: { glsl: "imageStore", wgsl: "textureStore" },
  returnTypeOf: () => Tvoid,
};
const Sin: IntrinsicRef = {
  name: "sin", pure: true,
  emit: { glsl: "sin", wgsl: "sin" },
  returnTypeOf: ([t]) => t!,
};

function bodyOf(m: Module): Stmt {
  for (const v of m.values) if (v.kind === "Entry") return v.entry.body;
  throw new Error("no entry");
}
function entryWith(body: Stmt): Module {
  return {
    types: [], values: [{
      kind: "Entry", entry: {
        name: "main", stage: "fragment", inputs: [], outputs: [],
        arguments: [], returnType: Tvoid, body, decorations: [],
      },
    }],
  };
}

// ─── foldConstants edge cases ─────────────────────────────────────────

describe("foldConstants edge cases", () => {
  it("Or short-circuits on true", () => {
    const r = foldExpr({ kind: "Or", lhs: constB(true), rhs: constB(false), type: Tbool });
    expect(r.kind === "Const" && r.value.kind === "Bool" && r.value.value).toBe(true);
  });

  it("integer Mod", () => {
    const r = foldExpr({ kind: "Mod", lhs: constI(10), rhs: constI(3), type: Ti32 });
    expect(r.kind === "Const" && r.value.kind === "Int" && r.value.value).toBe(1);
  });

  it("integer Div by zero is non-trapping (returns 0)", () => {
    const r = foldExpr({ kind: "Div", lhs: constI(10), rhs: constI(0), type: Ti32 });
    expect(r.kind === "Const" && r.value.kind === "Int" && r.value.value).toBe(0);
  });

  it("unary Neg on int", () => {
    const r = foldExpr({ kind: "Neg", value: constI(7), type: Ti32 });
    expect(r.kind === "Const" && r.value.kind === "Int" && r.value.value).toBe(-7);
  });

  it("unary Not on bool", () => {
    const r = foldExpr({ kind: "Not", value: constB(true), type: Tbool });
    expect(r.kind === "Const" && r.value.kind === "Bool" && r.value.value).toBe(false);
  });

  it("VecSwizzle of NewVector projects components", () => {
    // NewVector(a, b, c).yx → NewVector(b, a)
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    const b: Expr = { kind: "ReadInput", scope: "Uniform", name: "b", type: Tf32 };
    const c: Expr = { kind: "ReadInput", scope: "Uniform", name: "c", type: Tf32 };
    const sw = foldExpr({
      kind: "VecSwizzle",
      value: { kind: "NewVector", components: [a, b, c], type: Tvec3f },
      comps: ["y", "x"],
      type: Vec(Tf32, 2),
    });
    expect(sw.kind).toBe("NewVector");
    if (sw.kind === "NewVector") {
      expect(sw.components.length).toBe(2);
      expect(sw.components[0]).toBe(b);
      expect(sw.components[1]).toBe(a);
    }
  });

  it("nested Add with all constants folds fully", () => {
    // ((1 + 2) + (3 + 4)) → 10
    const r = foldExpr({
      kind: "Add",
      lhs: { kind: "Add", lhs: constI(1), rhs: constI(2), type: Ti32 },
      rhs: { kind: "Add", lhs: constI(3), rhs: constI(4), type: Ti32 },
      type: Ti32,
    });
    expect(r.kind === "Const" && r.value.kind === "Int" && r.value.value).toBe(10);
  });

  it("partial fold leaves non-const operand intact", () => {
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    const r = foldExpr({
      kind: "Add",
      lhs: { kind: "Add", lhs: constF(1), rhs: constF(2), type: Tf32 },
      rhs: a,
      type: Tf32,
    });
    expect(r.kind).toBe("Add");
    if (r.kind === "Add") {
      expect(r.lhs.kind === "Const" && r.lhs.value.kind === "Float" && r.lhs.value.value).toBe(3);
      expect(r.rhs).toBe(a);
    }
  });
});

// ─── dce edge cases ───────────────────────────────────────────────────

describe("dce edge cases", () => {
  it("preserves Declare with side-effecting init (impure call)", () => {
    const x: Var = { name: "x", type: Tvoid, mutable: false };
    const m = entryWith({
      kind: "Sequential",
      body: [
        {
          kind: "Declare", var: x, init: {
            kind: "Expr",
            value: { kind: "CallIntrinsic", op: ImageStore, args: [], type: Tvoid },
          },
        },
      ],
    });
    const out = dce(m);
    // Even though x is unused, the impure init must be retained.
    expect(bodyOf(out).kind).not.toBe("Nop");
  });

  it("preserves Increment statements", () => {
    const x: Var = { name: "x", type: Ti32, mutable: true };
    const m = entryWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constI(0) } },
        { kind: "Increment", target: { kind: "LVar", var: x, type: Ti32 }, prefix: false },
      ],
    });
    const out = dce(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    // Increment counts as use of x, so Declare stays. Increment itself never elided.
    expect(seq.body.some((s) => s.kind === "Increment")).toBe(true);
  });

  it("collapses empty If with both branches Nop", () => {
    const m = entryWith({
      kind: "If",
      cond: { kind: "ReadInput", scope: "Uniform", name: "u_flag", type: Tbool },
      then: { kind: "Nop" },
      else: { kind: "Nop" },
    });
    const out = dce(m);
    // Cond is pure → whole If becomes Nop.
    expect(bodyOf(out).kind).toBe("Nop");
  });

  it("flattens singleton Sequential", () => {
    const m = entryWith({
      kind: "Sequential",
      body: [
        {
          kind: "WriteOutput", name: "x",
          value: { kind: "Expr", value: constF(1) },
        },
      ],
    });
    const out = dce(m);
    expect(bodyOf(out).kind).toBe("WriteOutput");
  });

  it("nested Sequential collapse", () => {
    const m = entryWith({
      kind: "Sequential",
      body: [
        { kind: "Sequential", body: [] },
        { kind: "Nop" },
        {
          kind: "WriteOutput", name: "x",
          value: { kind: "Expr", value: constF(1) },
        },
      ],
    });
    const out = dce(m);
    expect(bodyOf(out).kind).toBe("WriteOutput");
  });
});

// ─── cse: commutativity + side-effect barrier ────────────────────────

describe("cse — commutativity", () => {
  it("recognises a+b and b+a as the same expression", () => {
    const r: Var = { name: "r", type: Tf32, mutable: true };
    const a: Expr = { kind: "ReadInput", scope: "Uniform", name: "a", type: Tf32 };
    const b: Expr = { kind: "ReadInput", scope: "Uniform", name: "b", type: Tf32 };
    const c: Expr = { kind: "ReadInput", scope: "Uniform", name: "c", type: Tf32 };
    // (a+b) + (b+a) + c — first two are commutatively equal.
    // Make each subexpression at least 3 nodes (the threshold).
    const e1: Expr = { kind: "Add", lhs: { kind: "Mul", lhs: a, rhs: b, type: Tf32 }, rhs: c, type: Tf32 };
    const e2: Expr = { kind: "Add", lhs: c, rhs: { kind: "Mul", lhs: b, rhs: a, type: Tf32 }, type: Tf32 };
    const sum: Expr = { kind: "Add", lhs: e1, rhs: e2, type: Tf32 };

    const m = entryWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: r, init: { kind: "Expr", value: sum } },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(r) },
        },
      ],
    });
    const out = cse(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    const cseDecls = seq.body.filter((s) => s.kind === "Declare" && s.var.name.startsWith("_cse"));
    // At least one CSE temp introduced (the (a*b)+c shape).
    expect(cseDecls.length).toBeGreaterThan(0);
  });
});

// ─── inline: copy-prop limits ────────────────────────────────────────

describe("inline — copy propagation limits", () => {
  it("does NOT copy-propagate non-trivial init", () => {
    const x: Var = { name: "x", type: Tf32, mutable: false };
    // let x = sin(u_time);  use(x) — sin call shouldn't be inlined twice
    // (it would be if we propagated; we only propagate trivial Var/Const/etc.)
    const init: Expr = {
      kind: "CallIntrinsic", op: Sin, args: [
        { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 },
      ], type: Tf32,
    };
    const m = entryWith({
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: init } },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = inlinePass(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    expect(seq.kind).toBe("Sequential");
    expect(seq.body[0]!.kind).toBe("Declare");
  });

  it("does NOT copy-propagate mutable Var", () => {
    const x: Var = { name: "x", type: Tf32, mutable: true };
    const m = entryWith({
      kind: "Sequential",
      body: [
        // var x = u_time
        { kind: "Declare", var: x, init: { kind: "Expr", value: { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 } } },
        // x = x * 2
        {
          kind: "Write",
          target: { kind: "LVar", var: x, type: Tf32 },
          value: { kind: "Mul", lhs: v(x), rhs: constF(2), type: Tf32 },
        },
        {
          kind: "WriteOutput", name: "outVal",
          value: { kind: "Expr", value: v(x) },
        },
      ],
    });
    const out = inlinePass(m);
    const seq = bodyOf(out) as Extract<Stmt, { kind: "Sequential" }>;
    // Declare must remain — mutable, would observe writes.
    expect(seq.body[0]!.kind).toBe("Declare");
  });
});

// ─── purity / side-effect helpers ────────────────────────────────────

describe("purity helpers", () => {
  it("CallIntrinsic with pure args but impure op → not pure", () => {
    const e: Expr = {
      kind: "Add",
      lhs: constF(1),
      rhs: { kind: "CallIntrinsic", op: ImageStore, args: [], type: Tvoid },
      type: Tf32,
    };
    expect(isPure(e)).toBe(false);
  });

  it("Sequential of pure expressions has no side effects", () => {
    const x: Var = { name: "x", type: Tf32, mutable: false };
    const s: Stmt = {
      kind: "Sequential",
      body: [
        { kind: "Declare", var: x, init: { kind: "Expr", value: constF(1) } },
      ],
    };
    expect(hasSideEffects(s)).toBe(false);
  });

  it("WriteOutput is a side effect", () => {
    const s: Stmt = {
      kind: "WriteOutput", name: "x",
      value: { kind: "Expr", value: constF(1) },
    };
    expect(hasSideEffects(s)).toBe(true);
  });

  it("Discard is a side effect", () => {
    expect(hasSideEffects({ kind: "Discard" })).toBe(true);
  });
});
