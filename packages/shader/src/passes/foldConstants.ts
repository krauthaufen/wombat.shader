// Constant folding. Evaluates pure constant subtrees at compile time.
//
// Conservative: we fold only on shapes the IR itself describes — Add/Sub/
// Mul/Div/Mod on Int/Float, Neg, Not, comparisons, And/Or short-circuit,
// Conditional with constant cond, simple swizzles of NewVector. Leaves
// matrix folding to a more careful pass.

import type { Expr, Literal, Module, Stmt } from "../ir/index.js";
import { mapStmt, mapExpr } from "./transform.js";

export function foldConstants(module: Module): Module {
  const values = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: foldStmt(v.body) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: foldStmt(v.entry.body) } };
    return v;
  });
  return { ...module, values };
}

export function foldStmt(s: Stmt): Stmt {
  return mapStmt(s, { expr: foldExpr });
}

export function foldExpr(e: Expr): Expr {
  // Fold post-order: children first, then the node.
  return foldOnce(mapExpr(e, foldOnce));
}

function foldOnce(e: Expr): Expr {
  switch (e.kind) {
    case "Neg":
      return foldUnaryNeg(e);
    case "Not":
      return foldUnaryNot(e);
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Mod":
      return foldArith(e);
    case "And":
    case "Or":
      return foldLogic(e);
    case "Eq":
    case "Neq":
    case "Lt":
    case "Le":
    case "Gt":
    case "Ge":
      return foldCompare(e);
    case "Conditional":
      return foldConditional(e);
    case "VecSwizzle":
      return foldSwizzle(e);
    default:
      return e;
  }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function asConst(e: Expr): Literal | undefined {
  return e.kind === "Const" ? e.value : undefined;
}

function constInt(value: number, signed: boolean): Literal {
  return { kind: "Int", signed, value };
}
function constFloat(value: number): Literal {
  return { kind: "Float", value };
}
function constBool(value: boolean): Literal {
  return { kind: "Bool", value };
}

function withValue(e: Expr, value: Literal): Expr {
  return { ...e, kind: "Const", value };
}

// ─────────────────────────────────────────────────────────────────────
// Arithmetic
// ─────────────────────────────────────────────────────────────────────

function foldUnaryNeg(e: Expr & { kind: "Neg" }): Expr {
  const c = asConst(e.value);
  if (!c) return e;
  if (c.kind === "Int") return withValue(e, constInt(-c.value | 0, c.signed));
  if (c.kind === "Float") return withValue(e, constFloat(-c.value));
  return e;
}

function foldUnaryNot(e: Expr & { kind: "Not" }): Expr {
  const c = asConst(e.value);
  if (c?.kind === "Bool") return withValue(e, constBool(!c.value));
  return e;
}

function foldArith(
  e: Expr & { kind: "Add" | "Sub" | "Mul" | "Div" | "Mod" },
): Expr {
  const a = asConst(e.lhs), b = asConst(e.rhs);
  if (!a || !b) return e;
  if (a.kind === "Int" && b.kind === "Int") {
    const av = a.value, bv = b.value;
    const r = e.kind === "Add" ? av + bv
      : e.kind === "Sub" ? av - bv
      : e.kind === "Mul" ? av * bv
      : e.kind === "Div" ? (bv !== 0 ? Math.trunc(av / bv) : 0)
      : (bv !== 0 ? av % bv : 0);
    return withValue(e, constInt(r | 0, a.signed));
  }
  if (a.kind === "Float" && b.kind === "Float") {
    const av = a.value, bv = b.value;
    const r = e.kind === "Add" ? av + bv
      : e.kind === "Sub" ? av - bv
      : e.kind === "Mul" ? av * bv
      : e.kind === "Div" ? av / bv
      : av % bv;
    return withValue(e, constFloat(r));
  }
  return e;
}

// ─────────────────────────────────────────────────────────────────────
// Logic and comparison
// ─────────────────────────────────────────────────────────────────────

function foldLogic(e: Expr & { kind: "And" | "Or" }): Expr {
  const a = asConst(e.lhs);
  if (a?.kind === "Bool") {
    if (e.kind === "And") return a.value ? e.rhs : withValue(e, constBool(false));
    return a.value ? withValue(e, constBool(true)) : e.rhs;
  }
  const b = asConst(e.rhs);
  if (b?.kind === "Bool") {
    // We can fold `x && true → x` and `x || false → x`, but only if x is pure.
    // Without isPure here we play it safe and only handle the absorbing side.
    if (e.kind === "And" && b.value === false) return withValue(e, constBool(false));
    if (e.kind === "Or" && b.value === true) return withValue(e, constBool(true));
  }
  return e;
}

function foldCompare(
  e: Expr & { kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge" },
): Expr {
  const a = asConst(e.lhs), b = asConst(e.rhs);
  if (!a || !b) return e;
  if (a.kind !== b.kind) return e;
  if (a.kind === "Int" && b.kind === "Int") {
    return withValue(e, constBool(cmp(e.kind, a.value, b.value)));
  }
  if (a.kind === "Float" && b.kind === "Float") {
    return withValue(e, constBool(cmp(e.kind, a.value, b.value)));
  }
  if (a.kind === "Bool" && b.kind === "Bool") {
    if (e.kind === "Eq") return withValue(e, constBool(a.value === b.value));
    if (e.kind === "Neq") return withValue(e, constBool(a.value !== b.value));
  }
  return e;
}

function cmp(kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge", a: number, b: number): boolean {
  switch (kind) {
    case "Eq": return a === b;
    case "Neq": return a !== b;
    case "Lt": return a < b;
    case "Le": return a <= b;
    case "Gt": return a > b;
    case "Ge": return a >= b;
  }
}

// ─────────────────────────────────────────────────────────────────────
// Conditional
// ─────────────────────────────────────────────────────────────────────

function foldConditional(e: Expr & { kind: "Conditional" }): Expr {
  const c = asConst(e.cond);
  if (c?.kind === "Bool") return c.value ? e.ifTrue : e.ifFalse;
  return e;
}

// ─────────────────────────────────────────────────────────────────────
// Swizzle of NewVector
// ─────────────────────────────────────────────────────────────────────

function foldSwizzle(e: Expr & { kind: "VecSwizzle" }): Expr {
  if (e.value.kind !== "NewVector") return e;
  const idx = (c: "x" | "y" | "z" | "w"): number =>
    c === "x" ? 0 : c === "y" ? 1 : c === "z" ? 2 : 3;
  // If single component → that component's expression.
  if (e.comps.length === 1) {
    const i = idx(e.comps[0]!);
    const c = e.value.components[i];
    if (c) return c;
    return e;
  }
  // If a permutation/projection of components → a new NewVector built from them.
  const picked: Expr[] = [];
  for (const k of e.comps) {
    const i = idx(k);
    const c = e.value.components[i];
    if (!c) return e;
    picked.push(c);
  }
  return { ...e, kind: "NewVector", components: picked };
}
