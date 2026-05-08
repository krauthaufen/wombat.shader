// Variable / input substitution. Used by inlining and constant propagation.

import type { Expr, InputScope, LExpr, Module, Stmt, Var } from "../ir/index.js";
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

// ─── Module-level substitution helpers ─────────────────────────────────
//
// `substInput` operates on a single Stmt. The helpers below walk a whole
// Module (every Function body, every Entry body) and apply the same
// transformation. Useful when a binding-shape transform — e.g. swapping
// a uniform read for a heap-buffer load — needs to apply to all entries
// without callers iterating ValueDefs themselves.

/**
 * For every read of `ReadInput(scope, name)` in the Module's
 * Function and Entry bodies, replace it with `mapping(name)`.
 * If `mapping(name)` returns `undefined`, the read is left intact.
 *
 * Pure wrt the Module's identity-preservation contract: only bodies
 * change; ValueDef structure (uniforms, storage buffers, samplers,
 * etc.) is untouched.
 */
export function substituteInputs(
  m: Module,
  scope: InputScope,
  mapping: (name: string) => Expr | undefined,
): Module {
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "ReadInput" && e.scope === scope) {
      const r = mapping(e.name);
      if (r !== undefined) return r;
    }
    return e;
  };
  const stmtMap = (s: Stmt): Stmt => mapStmt(s, { expr: exprFn });

  const values = m.values.map((v): typeof v => {
    if (v.kind === "Function") return { ...v, body: stmtMap(v.body) };
    if (v.kind === "Entry")    return { ...v, entry: { ...v.entry, body: stmtMap(v.entry.body) } };
    return v;
  });
  return { ...m, values };
}

/**
 * Convenience: substitute every `ReadInput("Uniform", name)` whose
 * name is in the mapping. Equivalent to
 * `substituteInputs(m, "Uniform", n => mapping.get(n))`.
 */
export function substituteUniforms(
  m: Module,
  mapping: ReadonlyMap<string, Expr> | ((name: string) => Expr | undefined),
): Module {
  const fn = typeof mapping === "function" ? mapping : (n: string) => mapping.get(n);
  return substituteInputs(m, "Uniform", fn);
}

/**
 * Convenience: substitute every `ReadInput("Input", name)` whose
 * name is in the mapping. Use for vertex-attribute substitutions in
 * the vertex stage (heap-buffer pulls, instance-array fan-out, …).
 *
 * NOTE: applies to ALL stages' Input reads. If the substitution is
 * vertex-specific, restrict the mapping accordingly (e.g. via the
 * `name` keys; or pre-filter the entries before calling).
 */
export function substituteAttributes(
  m: Module,
  mapping: ReadonlyMap<string, Expr> | ((name: string) => Expr | undefined),
): Module {
  const fn = typeof mapping === "function" ? mapping : (n: string) => mapping.get(n);
  return substituteInputs(m, "Input", fn);
}
