// Variable / input substitution. Used by inlining and constant propagation.

import type { Expr, InputScope, LExpr, Stmt, Var } from "../ir/index.js";
import { mapExpr, mapStmt } from "./transform.js";

/** Replace every read of `oldVar` with `newExpr`. Writes (LVar) are untouched. */
export function substVar(s: Stmt, oldVar: Var, newExpr: Expr): Stmt {
  const exprFn = (e: Expr): Expr =>
    e.kind === "Var" && e.var === oldVar ? newExpr : e;
  // Walk the tree; mapExpr is post-order so substitution happens leaf-first.
  return mapStmt(s, { expr: exprFn });
}

/** Replace `ReadInput(scope, name)` with `expr`. */
export function substInput(
  s: Stmt,
  scope: InputScope,
  name: string,
  expr: Expr,
): Stmt {
  const exprFn = (e: Expr): Expr =>
    e.kind === "ReadInput" && e.scope === scope && e.name === name ? expr : e;
  return mapStmt(s, { expr: exprFn });
}

/** Simultaneous substitution: many `Var → Expr` mappings. */
export function substVars(s: Stmt, mapping: ReadonlyMap<Var, Expr>): Stmt {
  if (mapping.size === 0) return s;
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "Var") {
      const r = mapping.get(e.var);
      return r ?? e;
    }
    return e;
  };
  return mapStmt(s, { expr: exprFn });
}

/** Same but on a single Expr. */
export function substVarsExpr(e: Expr, mapping: ReadonlyMap<Var, Expr>): Expr {
  if (mapping.size === 0) return e;
  return mapExpr(e, (x) => {
    if (x.kind === "Var") {
      const r = mapping.get(x.var);
      return r ?? x;
    }
    return x;
  });
}

/** Used by the inliner for parameter binding. */
export function substVarsLExpr(
  l: LExpr,
  _mapping: ReadonlyMap<Var, Expr>,
): LExpr {
  // L-values can't be replaced by Exprs (they need to remain l-values).
  // The inliner should reject inout parameters with non-LVar arguments.
  // Returning unchanged is the safe default; the inliner is responsible
  // for upstream legalisation.
  return l;
}
