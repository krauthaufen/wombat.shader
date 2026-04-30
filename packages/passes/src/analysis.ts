// Read-only analyses on IR subtrees. Used by passes to make decisions.

import {
  visitExprChildren,
  visitLExprChildren,
  visitStmt,
  type Expr,
  type InputScope,
  type LExpr,
  type RExpr,
  type Stmt,
  type Var,
} from "@aardworx/wombat.shader-ir";

// ─────────────────────────────────────────────────────────────────────
// Purity
// ─────────────────────────────────────────────────────────────────────

/** True iff `e` has no impure intrinsic / function call anywhere. */
export function isPure(e: Expr): boolean {
  let pure = true;
  const walk = (x: Expr): void => {
    if (!pure) return;
    if (x.kind === "Call" && !x.fn.pure) { pure = false; return; }
    if (x.kind === "CallIntrinsic" && !x.op.pure) { pure = false; return; }
    visitExprChildren(x, walk);
  };
  walk(e);
  return pure;
}

/** True iff `s` (or any nested Stmt) has any side-effecting call/write. */
export function hasSideEffects(s: Stmt): boolean {
  let dirty = false;
  visitStmt(s, {
    preStmt(stmt) {
      if (dirty) return;
      switch (stmt.kind) {
        case "Write":
        case "WriteOutput":
        case "Increment":
        case "Decrement":
        case "Discard":
        case "Barrier":
          dirty = true;
          return;
        case "Expression":
          if (!isPure(stmt.value)) dirty = true;
          return;
      }
    },
  });
  return dirty;
}

// ─────────────────────────────────────────────────────────────────────
// Free variables (of an Expr or Stmt)
// ─────────────────────────────────────────────────────────────────────

/** Variables read by the expression. */
export function freeVarsExpr(e: Expr, into: Set<Var> = new Set()): ReadonlySet<Var> {
  const walk = (x: Expr): void => {
    if (x.kind === "Var") { into.add(x.var); return; }
    visitExprChildren(x, walk);
  };
  walk(e);
  return into;
}

export function freeVarsLExpr(l: LExpr, into: Set<Var> = new Set()): ReadonlySet<Var> {
  const walkE = (x: Expr): void => {
    if (x.kind === "Var") { into.add(x.var); return; }
    visitExprChildren(x, walkE);
  };
  const walkL = (x: LExpr): void => {
    if (x.kind === "LVar") { into.add(x.var); return; }
    visitLExprChildren(x, walkL, walkE);
  };
  walkL(l);
  return into;
}

/** Variables read by the statement subtree (excluding the variables it declares). */
export function freeVarsStmt(s: Stmt): ReadonlySet<Var> {
  const reads = new Set<Var>();
  const declared = new Set<Var>();
  visitStmt(s, {
    preStmt(stmt) {
      if (stmt.kind === "Declare") declared.add(stmt.var);
    },
    expr: {
      pre(e) {
        if (e.kind === "Var" && !declared.has(e.var)) reads.add(e.var);
      },
    },
  });
  return reads;
}

// ─────────────────────────────────────────────────────────────────────
// Inputs / outputs
// ─────────────────────────────────────────────────────────────────────

export interface ScopedName {
  readonly scope: InputScope;
  readonly name: string;
}

function key(s: ScopedName): string { return `${s.scope}:${s.name}`; }

/** Inputs (any scope) read by the subtree. */
export function readInputs(s: Stmt | Expr): ReadonlyMap<string, ScopedName> {
  const out = new Map<string, ScopedName>();
  const onExpr = (e: Expr): void => {
    if (e.kind === "ReadInput") {
      const sn = { scope: e.scope, name: e.name };
      out.set(key(sn), sn);
    }
    visitExprChildren(e, onExpr);
  };
  if ("kind" in s && (s as Expr).type !== undefined && expressionKinds.has((s as Expr).kind)) {
    onExpr(s as Expr);
  } else {
    visitStmt(s as Stmt, {
      expr: { pre: onExpr },
    });
  }
  return out;
}

const expressionKinds = new Set<Expr["kind"]>([
  "Var","Const","ReadInput","Call","CallIntrinsic","Conditional",
  "Neg","Not","BitNot",
  "Add","Sub","Mul","Div","Mod","MulMatMat","MulMatVec","MulVecMat",
  "Transpose","Inverse","Determinant",
  "Dot","Cross","Length","VecSwizzle","VecItem",
  "MatrixElement","MatrixRow","MatrixCol",
  "NewVector","NewMatrix","MatrixFromRows","MatrixFromCols",
  "ConvertMatrix","Convert",
  "And","Or","BitAnd","BitOr","BitXor","ShiftLeft","ShiftRight",
  "Eq","Neq","Lt","Le","Gt","Ge",
  "VecAny","VecAll","VecLt","VecLe","VecGt","VecGe","VecEq","VecNeq",
  "Field","Item","DebugPrintf",
]);

/** Output names (LInput.scope === "Output") written in the subtree. */
export function writtenOutputs(s: Stmt): ReadonlySet<string> {
  const out = new Set<string>();
  visitStmt(s, {
    preStmt(stmt) {
      if (stmt.kind === "WriteOutput") out.add(stmt.name);
      else if (stmt.kind === "Write" && stmt.target.kind === "LInput" && stmt.target.scope === "Output") {
        out.add(stmt.target.name);
      }
    },
  });
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Live variables — variables actually read by code that can run later.
// Conservative: we treat Sequential as flowing top-to-bottom.
// ─────────────────────────────────────────────────────────────────────

/**
 * The set of vars that are "live out" of `s` — read by code following `s`
 * lexically. For a `Sequential` of [a; b; c], after a the set is
 * `freeVars(b) ∪ freeVars(c)`. We compute this approximately by
 * scanning bottom-up.
 */
export function liveAfter(s: readonly Stmt[]): Var[] {
  const live = new Set<Var>();
  for (let i = s.length - 1; i >= 0; i--) {
    // Read: every freeVar inside the current Stmt is live before it.
    for (const v of freeVarsStmt(s[i]!)) live.add(v);
    // Kill: variables declared in the Stmt are no longer live before it.
    visitStmt(s[i]!, {
      preStmt(stmt) {
        if (stmt.kind === "Declare") live.delete(stmt.var);
      },
    });
  }
  return [...live];
}

// ─────────────────────────────────────────────────────────────────────
// RExpr helpers
// ─────────────────────────────────────────────────────────────────────

export function isRExprPure(r: RExpr): boolean {
  if (r.kind === "Expr") return isPure(r.value);
  return r.values.every(isPure);
}
