// Dead-code elimination.
//
// Drops:
//  - `Declare` of a Var that's never read AND whose initialiser is pure
//  - `Expression` of a pure expression (its value is discarded; with no
//    side effects, the whole statement is dead)
//  - `Sequential` flattening: empty bodies become Nop, single-stmt bodies
//    surface the inner statement
//  - `If` with constant cond → the matching branch
//
// Doesn't touch outputs, writes, increments, decrements, function calls
// with side effects, or barriers.

import type {
  Module,
  Stmt,
  Var,
} from "../ir/index.js";
import { freeVarsStmt, isPure, isRExprPure } from "./analysis.js";

export function dce(module: Module): Module {
  const values = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: dceStmt(v.body) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: dceStmt(v.entry.body) } };
    return v;
  });
  return { ...module, values };
}

export function dceStmt(s: Stmt): Stmt {
  // Recurse into nested statements first.
  const recursed = recurse(s);
  return collapse(recursed);
}

function recurse(s: Stmt): Stmt {
  switch (s.kind) {
    case "Sequential":
    case "Isolated":
      return { ...s, body: s.body.map(dceStmt) };
    case "If": {
      const then_ = dceStmt(s.then);
      const else_ = s.else ? dceStmt(s.else) : undefined;
      return {
        ...s,
        then: then_,
        ...(else_ !== undefined ? { else: else_ } : {}),
      };
    }
    case "For":
      return { ...s, init: dceStmt(s.init), step: dceStmt(s.step), body: dceStmt(s.body) };
    case "While":
    case "DoWhile":
      return { ...s, body: dceStmt(s.body) };
    case "Loop":
      return { ...s, body: dceStmt(s.body) };
    case "Switch": {
      const cases = s.cases.map((c) => ({ ...c, body: dceStmt(c.body) }));
      const def = s.default ? dceStmt(s.default) : undefined;
      return { ...s, cases, ...(def !== undefined ? { default: def } : {}) };
    }
    default:
      return s;
  }
}

function collapse(s: Stmt): Stmt {
  switch (s.kind) {
    case "Sequential":
    case "Isolated": {
      // Drop dead Declare and pure Expression statements.
      const used = collectUsed(s.body);
      const kept: Stmt[] = [];
      for (const child of s.body) {
        if (child.kind === "Nop") continue;
        if (child.kind === "Declare") {
          const initPure = !child.init || isRExprPure(child.init);
          if (!used.has(child.var) && initPure) continue;
        }
        if (child.kind === "Expression" && isPure(child.value)) continue;
        if ((child.kind === "Sequential" || child.kind === "Isolated") && child.body.length === 0) continue;
        kept.push(child);
      }
      if (kept.length === 0) return { kind: "Nop" };
      if (kept.length === 1 && s.kind === "Sequential") return kept[0]!;
      return { ...s, body: kept };
    }
    case "If": {
      // Constant-cond elimination, after constant folding.
      if (s.cond.kind === "Const" && s.cond.value.kind === "Bool") {
        return s.cond.value.value ? s.then : (s.else ?? { kind: "Nop" });
      }
      // Both branches Nop?
      if (isNop(s.then) && (!s.else || isNop(s.else))) {
        // Cond may be impure — emit it as an expression statement.
        if (isPure(s.cond)) return { kind: "Nop" };
        return { kind: "Expression", value: s.cond };
      }
      return s;
    }
    default:
      return s;
  }
}

function isNop(s: Stmt): boolean {
  if (s.kind === "Nop") return true;
  if ((s.kind === "Sequential" || s.kind === "Isolated") && s.body.length === 0) return true;
  return false;
}

/**
 * Sweep a sequential block top-to-bottom and collect which Vars are
 * actually read by later code. Only looks within this block (callers
 * that want global liveness should pre-flatten or run the pass to a
 * fixed point).
 */
function collectUsed(body: readonly Stmt[]): Set<Var> {
  const used = new Set<Var>();
  // Collect from every statement; a Var is used iff any later (or any)
  // statement reads it.
  for (const st of body) {
    for (const v of freeVarsStmt(st)) used.add(v);
  }
  return used;
}

export const _internal = { collapse, isNop };
