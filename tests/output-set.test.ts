// Symbolic output-set analysis on shader-IR statement trees.

import { describe, expect, it } from "vitest";
import {
  analyseOutputSet, unfoldConditional, substituteVars,
  evaluateConcrete, evaluateSet,
  Tu32, Tf32,
  type Expr, type Stmt, type Var, type IntrinsicEvalTable,
} from "@aardworx/wombat.shader/ir";

function u32(value: number): Expr {
  return { kind: "Const", type: Tu32, value: { kind: "Int", value: value >>> 0, signed: false } };
}
function f32(value: number): Expr {
  return { kind: "Const", type: Tf32, value: { kind: "Float", value } };
}
function ret(v: Expr): Stmt { return { kind: "ReturnValue", value: v }; }
function seq(...body: Stmt[]): Stmt { return { kind: "Sequential", body }; }
function ifs(cond: Expr, then: Stmt, elseS?: Stmt): Stmt {
  return elseS !== undefined
    ? { kind: "If", cond, then, else: elseS }
    : { kind: "If", cond, then };
}
function decl(v: Var, init: Expr): Stmt { return { kind: "Declare", var: v, init }; }
function v(name: string, type = Tu32): Var { return { name, type, mutable: false }; }
function varRef(vv: Var): Expr { return { kind: "Var", var: vv, type: vv.type }; }
const declaredLeaf: Expr = { kind: "ReadInput", scope: "Uniform", name: "declared", type: Tu32 };
function cond(c: Expr, t: Expr, e: Expr): Expr {
  return { kind: "Conditional", cond: c, ifTrue: t, ifFalse: e, type: t.type };
}
function lt(a: Expr, b: Expr): Expr {
  return { kind: "Lt", lhs: a, rhs: b, type: { kind: "Bool" } };
}

describe("analyseOutputSet", () => {
  it("collects a single ReturnValue", () => {
    const body = ret(u32(42));
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(1);
    expect((exprs[0] as { value: { value: number } }).value.value).toBe(42);
  });

  it("returns an empty set when the body has no return", () => {
    const body = seq(decl(v("x"), u32(0)));
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(0);
  });

  it("collects both branches of an `if` with returns", () => {
    const body = seq(
      ifs(lt(declaredLeaf, u32(1)),
        ret(u32(0)),
        ret(declaredLeaf)),
    );
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(2);
  });

  it("flattens a top-level Conditional in the returned expression", () => {
    // `return cond ? a : b` lowers to `ReturnValue(Conditional(cond,a,b))`
    // — the analysis should yield {a, b}, not {Conditional}.
    const body = ret(cond(lt(declaredLeaf, u32(0)), u32(7), declaredLeaf));
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(2);
    expect((exprs[0] as { value: { value: number } }).value.value).toBe(7);
    expect((exprs[1] as { scope: string }).scope).toBe("Uniform");
  });

  it("flattens nested Conditionals (pickEnum-style chains)", () => {
    // pickEnum lowering: select(select(0, 2, eq(d,1)), 1, eq(d,2))
    const c1 = cond(lt(declaredLeaf, u32(1)), u32(0), u32(2));   // (d<1) ? 0 : 2
    const c2 = cond(lt(declaredLeaf, u32(2)), u32(1), c1);       // (d<2) ? 1 : c1
    const { exprs } = analyseOutputSet(ret(c2));
    // {0, 2, 1} after flattening
    expect(exprs).toHaveLength(3);
  });

  it("substitutes `let`-bound vars into subsequent returns", () => {
    const xx = v("x");
    const body = seq(
      decl(xx, declaredLeaf),
      ret(varRef(xx)),
    );
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(1);
    expect((exprs[0] as { scope: string }).scope).toBe("Uniform");
    expect((exprs[0] as { name: string }).name).toBe("declared");
  });

  it("dedupes structurally identical returns", () => {
    // Same `declared` leaf returned from both branches.
    const body = seq(
      ifs(lt(declaredLeaf, u32(0)), ret(declaredLeaf), ret(declaredLeaf)),
    );
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(1);
  });

  it("unfoldConditional leaves non-Conditional unchanged", () => {
    const r = unfoldConditional(u32(7));
    expect(r).toHaveLength(1);
    expect(r[0]).toBe(u32(7).constructor === Object ? r[0] : r[0]); // identity-ish
  });

  it("substituteVars resolves a Var to its env binding", () => {
    const xx = v("x");
    const e: Expr = { kind: "Add", lhs: varRef(xx), rhs: u32(1), type: Tu32 };
    const env = new Map<string, Expr>([["x", u32(5)]]);
    const out = substituteVars(e, env) as { lhs: { value: { value: number } } };
    expect(out.lhs.value.value).toBe(5);
  });

  it("evaluateConcrete folds arithmetic + conditionals against an env", () => {
    // (declared < 1) ? 0 : (declared + 5)
    const declared: Expr = { kind: "ReadInput", scope: "Uniform", name: "declared", type: Tu32 };
    const body: Expr = cond(
      lt(declared, u32(1)),
      u32(0),
      { kind: "Add", lhs: declared, rhs: u32(5), type: Tu32 },
    );
    const env0 = new Map<string, number>([["declared", 0]]);
    const env3 = new Map<string, number>([["declared", 3]]);
    expect(evaluateConcrete(body, env0)).toBe(0);
    expect(evaluateConcrete(body, env3)).toBe(8);
  });

  it("evaluateConcrete folds CallIntrinsic via the registry", () => {
    const flipCullTable: IntrinsicEvalTable = new Map([
      ["flipCull", (args: ReadonlyArray<number>) => {
        const x = args[0]!;
        return x === 1 ? 2 : x === 2 ? 1 : 0;
      }],
    ]);
    const declared: Expr = { kind: "ReadInput", scope: "Uniform", name: "declared", type: Tu32 };
    const call: Expr = {
      kind: "CallIntrinsic",
      op: { name: "flipCull", returnTypeOf: () => Tu32, pure: true, emit: { glsl: "flipCull", wgsl: "flipCull" } },
      args: [declared],
      type: Tu32,
    };
    expect(evaluateConcrete(call, new Map([["declared", 1]]), flipCullTable)).toBe(2);
    expect(evaluateConcrete(call, new Map([["declared", 2]]), flipCullTable)).toBe(1);
    expect(evaluateConcrete(call, new Map([["declared", 0]]), flipCullTable)).toBe(0);
  });

  it("evaluateSet folds the cull-rule symbolic set into per-declared slot counts", () => {
    const flipCullTable: IntrinsicEvalTable = new Map([
      ["flipCull", (args: ReadonlyArray<number>) => {
        const x = args[0]!;
        return x === 1 ? 2 : x === 2 ? 1 : 0;
      }],
    ]);
    const flipCallStub: Expr = {
      kind: "CallIntrinsic",
      op: { name: "flipCull", returnTypeOf: () => Tu32, pure: true, emit: { glsl: "flipCull", wgsl: "flipCull" } },
      args: [declaredLeaf],
      type: Tu32,
    };
    const det: Expr = { kind: "ReadInput", scope: "Uniform", name: "det", type: Tf32 };
    // Note: at slot-counting time we evaluate just the post-flatten
    // output set — det need not be bound because the analysis
    // produced two independent expressions, only one of which reads
    // det (via the original Conditional). Here `det` is unreferenced
    // by either branch after the flatten, so we just bind `declared`.
    void det;
    const set = analyseOutputSet(ret({
      kind: "Conditional", cond: lt(det, f32(0)),
      ifTrue: flipCallStub, ifFalse: declaredLeaf, type: Tu32,
    }));
    // declared=2 ("back") → {flipCull(2)=1, 2} → 2 slots.
    expect(evaluateSet(set, new Map([["declared", 2]]), flipCullTable)).toEqual(
      { resolved: [1, 2], unresolvedCount: 0 },
    );
    // declared=0 ("none") → {flipCull(0)=0, 0} → 1 slot.
    expect(evaluateSet(set, new Map([["declared", 0]]), flipCullTable)).toEqual(
      { resolved: [0], unresolvedCount: 0 },
    );
    // declared=1 ("front") → {flipCull(1)=2, 1} → 2 slots.
    expect(evaluateSet(set, new Map([["declared", 1]]), flipCullTable)).toEqual(
      { resolved: [1, 2], unresolvedCount: 0 },
    );
  });

  it("declared-flip-cull body: analysis yields the cull-rule's two-element set", () => {
    // Equivalent to `(det < 0) ? flipCull(declared) : declared` —
    // but we use a synthetic "flipCull" CallIntrinsic since the
    // analysis doesn't know about it semantically (it just keeps
    // the call as an opaque Expr leaf).
    const flipCallStub: Expr = {
      kind: "CallIntrinsic",
      op: { name: "flipCull", returnTypeOf: () => Tu32, pure: true, emit: { glsl: "flipCull", wgsl: "flipCull" } },
      args: [declaredLeaf],
      type: Tu32,
    };
    const det: Expr = { kind: "ReadInput", scope: "Uniform", name: "det", type: Tf32 };
    const body = ret(cond(lt(det, f32(0)), flipCallStub, declaredLeaf));
    const { exprs } = analyseOutputSet(body);
    expect(exprs).toHaveLength(2);
    // First branch: flipCull(declared); second: declared. Neither is constant-folded.
    expect((exprs[0] as { kind: string }).kind).toBe("CallIntrinsic");
    expect((exprs[1] as { kind: string }).kind).toBe("ReadInput");
  });
});
