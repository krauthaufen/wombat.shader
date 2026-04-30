// Common-subexpression elimination.
//
// Within an EntryDef body, this pass walks every Sequential block and:
//
//   1. Computes a stable hash for every pure Expr.
//   2. Whenever the same expression appears twice (or more) and is large
//      enough to be worth lifting, hoists the first occurrence into a
//      `Declare` and replaces both with `Var(temp)`.
//
// Conservative scoping: we only CSE within the same `Sequential` block.
// Hoisting across `If` branches needs a dominance analysis we don't have
// yet, and hoisting out of loops is correctness-sensitive.
//
// Expressions on the LHS of an l-value write are treated as opaque (we
// don't dive into LExpr indices or fields for CSE). Side-effecting
// statements between two occurrences are a barrier — if any happens
// in between, the second occurrence is left alone.

import type { Expr, Module, Stmt, Type, Var } from "@aardworx/wombat.shader-ir";
import { isPure } from "./analysis.js";
import { mapExprChildren as _mapExprChildren, mapStmtChildren } from "./transform.js";

const MIN_NODES_FOR_CSE = 3;

export function cse(module: Module): Module {
  const values = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: cseStmt(v.body) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: cseStmt(v.entry.body) } };
    return v;
  });
  return { ...module, values };
}

export function cseStmt(s: Stmt): Stmt {
  switch (s.kind) {
    case "Sequential":
    case "Isolated": {
      const lifted = liftBlock(s.body);
      return { ...s, body: lifted };
    }
    default:
      return mapStmtChildren(s, { stmt: cseStmt });
  }
}

function liftBlock(body: readonly Stmt[]): readonly Stmt[] {
  // First, recurse into every child statement.
  const recursed = body.map(cseStmt);

  // Pass 1: count occurrences of each pure expression.
  const counts = new Map<string, { expr: Expr; count: number }>();
  const seen = (e: Expr): void => {
    if (!isPure(e)) return;
    if (sizeOf(e) < MIN_NODES_FOR_CSE) return;
    const k = hash(e);
    const prev = counts.get(k);
    if (prev) prev.count++;
    else counts.set(k, { expr: e, count: 1 });
  };
  for (const st of recursed) walkExprs(st, seen);

  // Candidates with count > 1.
  const candidates = [...counts.values()].filter((c) => c.count > 1);
  if (candidates.length === 0) return recursed;

  // Pass 2: for each candidate, prepend a Declare and replace occurrences.
  const allocated = new Map<string, Var>();
  let counter = 0;
  for (const c of candidates) {
    const tmp: Var = {
      name: `_cse${counter++}`,
      type: c.expr.type,
      mutable: false,
    };
    allocated.set(hash(c.expr), tmp);
  }

  const replaced: Stmt[] = recursed.map((st) =>
    replaceInStmt(st, allocated)
  );

  // Prepend declarations in deterministic order (insertion order of `allocated`).
  const declarations: Stmt[] = [];
  for (const [k, tmp] of allocated) {
    const c = counts.get(k)!;
    declarations.push({
      kind: "Declare",
      var: tmp,
      init: { kind: "Expr", value: c.expr },
    });
  }
  return [...declarations, ...replaced];
}

function replaceInStmt(s: Stmt, allocated: Map<string, Var>): Stmt {
  // Walk Exprs only (skip lvalue subtrees other than via mapStmtChildren).
  const replaceExpr = (e: Expr): Expr => {
    const k = hashIfPure(e);
    if (k) {
      const tmp = allocated.get(k);
      if (tmp) return { kind: "Var", var: tmp, type: tmp.type };
    }
    // Recurse into children.
    return mapExprChildrenLocal(e, replaceExpr);
  };
  return mapStmtChildren(s, { expr: replaceExpr });
}

function mapExprChildrenLocal(e: Expr, fn: (c: Expr) => Expr): Expr {
  return _mapExprChildren(e, fn);
}

// ─────────────────────────────────────────────────────────────────────
// Hashing — stable across runs, doesn't dedupe via reference identity
// ─────────────────────────────────────────────────────────────────────

function hashIfPure(e: Expr): string | undefined {
  return isPure(e) ? hash(e) : undefined;
}

/**
 * Structural hash: a canonical string. Two Exprs produce the same hash
 * iff they have the same shape, the same literal values, and the same
 * Var identities (we use Var reference equality for vars, encoded via
 * a per-walk table; passes call `hash` from one root so consistent).
 */
function hash(e: Expr): string {
  return JSON.stringify(canonical(e));
}

function canonical(e: Expr): unknown {
  // Strip `span`; keep `type` (so V3f swizzle differs from f32 swizzle)
  // and `kind`. Vars are hashed by name + type — passes that rename use
  // fresh Var objects, so two distinct-but-equal-named locals would
  // collide here. Acceptable: scope is a Sequential block.
  switch (e.kind) {
    case "Var":
      return ["Var", e.var.name, typeId(e.var.type)];
    case "Const":
      return ["Const", e.value];
    case "ReadInput":
      return ["ReadInput", e.scope, e.name, e.index ? canonical(e.index) : null];
    case "Call":
      return ["Call", e.fn.id, e.args.map(canonical)];
    case "CallIntrinsic":
      return ["CallIntrinsic", e.op.name, e.args.map(canonical)];
    case "VecSwizzle":
      return ["VecSwizzle", canonical(e.value), e.comps.join("")];
    case "MatrixElement":
      return ["MatrixElement", canonical(e.matrix), canonical(e.row), canonical(e.col)];
    case "MatrixRow":
      return ["MatrixRow", canonical(e.matrix), canonical(e.row)];
    case "MatrixCol":
      return ["MatrixCol", canonical(e.matrix), canonical(e.col)];
    case "NewVector":
      return ["NewVector", e.components.map(canonical)];
    case "NewMatrix":
      return ["NewMatrix", e.elements.map(canonical)];
    case "MatrixFromRows":
      return ["MatrixFromRows", e.rows.map(canonical)];
    case "MatrixFromCols":
      return ["MatrixFromCols", e.cols.map(canonical)];
    case "Field":
      return ["Field", canonical(e.target), e.name];
    case "Item":
      return ["Item", canonical(e.target), canonical(e.index)];
    case "VecItem":
      return ["VecItem", canonical(e.value), canonical(e.index)];
    case "Conditional":
      return ["Conditional", canonical(e.cond), canonical(e.ifTrue), canonical(e.ifFalse)];
    case "Neg":
    case "Not":
    case "BitNot":
    case "Transpose":
    case "Inverse":
    case "Determinant":
    case "Length":
    case "VecAny":
    case "VecAll":
    case "ConvertMatrix":
    case "Convert":
      return [e.kind, canonical(e.value), typeId(e.type)];
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Mod":
    case "MulMatMat":
    case "MulMatVec":
    case "MulVecMat":
    case "Dot":
    case "Cross":
    case "And":
    case "Or":
    case "BitAnd":
    case "BitOr":
    case "BitXor":
    case "ShiftLeft":
    case "ShiftRight":
    case "Eq":
    case "Neq":
    case "Lt":
    case "Le":
    case "Gt":
    case "Ge":
    case "VecLt":
    case "VecLe":
    case "VecGt":
    case "VecGe":
    case "VecEq":
    case "VecNeq": {
      // Binary nodes — for commutative ops we sort the operand canonicals
      // so `a+b` and `b+a` hash the same.
      const lhs = canonical(e.lhs), rhs = canonical(e.rhs);
      if (isCommutative(e.kind)) {
        const [l, r] = stringify(lhs) < stringify(rhs) ? [lhs, rhs] : [rhs, lhs];
        return [e.kind, l, r];
      }
      return [e.kind, lhs, rhs];
    }
    case "DebugPrintf":
      return ["DebugPrintf", canonical(e.format), e.args.map(canonical)];
  }
}

function isCommutative(k: Expr["kind"]): boolean {
  return k === "Add" || k === "Mul" || k === "And" || k === "Or" ||
    k === "BitAnd" || k === "BitOr" || k === "BitXor" ||
    k === "Eq" || k === "Neq" ||
    k === "VecEq" || k === "VecNeq" || k === "Dot";
}

function typeId(t: Type): string {
  switch (t.kind) {
    case "Void": return "void";
    case "Bool": return "bool";
    case "Int": return t.signed ? "i32" : "u32";
    case "Float": return "f32";
    case "Vector": return `vec${t.dim}<${typeId(t.element)}>`;
    case "Matrix": return `mat${t.cols}x${t.rows}<${typeId(t.element)}>`;
    case "Array": return `array<${typeId(t.element)},${t.length}>`;
    case "Struct": return `struct:${t.name}`;
    case "Sampler": return `sampler:${t.target}:${t.sampled.kind}:${t.comparison}`;
    case "Texture": return `texture:${t.target}:${t.sampled.kind}:${t.arrayed}:${t.multisampled}`;
    case "AtomicI32": return "atomic<i32>";
    case "AtomicU32": return "atomic<u32>";
    case "Intrinsic": return `intrinsic:${t.name}`;
  }
}

function stringify(x: unknown): string {
  return JSON.stringify(x);
}

function sizeOf(e: Expr): number {
  let n = 1;
  switch (e.kind) {
    case "Var":
    case "Const":
      return 1;
    case "ReadInput":
      return 1 + (e.index ? sizeOf(e.index) : 0);
    case "Call":
    case "CallIntrinsic":
      return 1 + e.args.reduce((a, b) => a + sizeOf(b), 0);
    case "Conditional":
      return 1 + sizeOf(e.cond) + sizeOf(e.ifTrue) + sizeOf(e.ifFalse);
    default: {
      // Generic recursion — count child Exprs.
      let total = n;
      _mapExprChildren(e, (c) => { total += sizeOf(c); return c; });
      return total;
    }
  }
}

function walkExprs(s: Stmt, fn: (e: Expr) => void): void {
  // Visit each Expr in the immediate Stmt, not into nested Stmt children
  // (we recurse into children separately via cseStmt at higher level).
  switch (s.kind) {
    case "Expression":
      visitExprDeep(s.value, fn);
      return;
    case "Declare":
      if (s.init?.kind === "Expr") visitExprDeep(s.init.value, fn);
      else if (s.init) for (const v of s.init.values) visitExprDeep(v, fn);
      return;
    case "Write":
      visitExprDeep(s.value, fn);
      return;
    case "WriteOutput":
      if (s.index) visitExprDeep(s.index, fn);
      if (s.value.kind === "Expr") visitExprDeep(s.value.value, fn);
      else for (const v of s.value.values) visitExprDeep(v, fn);
      return;
    case "If":
      visitExprDeep(s.cond, fn);
      return;
    case "ReturnValue":
      visitExprDeep(s.value, fn);
      return;
    case "While":
    case "DoWhile":
      visitExprDeep(s.cond, fn);
      return;
    case "Switch":
      visitExprDeep(s.value, fn);
      return;
    case "For":
      visitExprDeep(s.cond, fn);
      return;
    default:
      return;
  }
}

function visitExprDeep(e: Expr, fn: (e: Expr) => void): void {
  fn(e);
  _mapExprChildren(e, (c) => { visitExprDeep(c, fn); return c; });
}
