// relinkVars — restore Var object identity after JSON deserialisation.
//
// The shader-vite plugin emits each `vertex(...)` / `fragment(...)`
// marker as `__wombat_stage(<jsonModule>, ...)`, where `<jsonModule>`
// is `JSON.stringify(module)`. JSON has no notion of object identity,
// so every reference to a single `Var` round-trips as a *separate*
// object with the same `name`/`type` fields. Downstream passes (DCE,
// CSE, inline) match Vars by reference equality and would otherwise
// see e.g. `let wp = …` as one Var and `out.gl_Position = wp * …` as
// a different one — DCE then drops the declaration as unused while
// the use still references the (now undeclared) name.
//
// Fix: walk every Entry / Function body and intern Vars by `name`,
// preferring the Var attached to the first `Declare` we encounter.
// Subsequent `Var` / `LVar` references with the same `name` get
// retargeted to that canonical Var object.
//
// Limitation: this assumes a single Var name binds to one Var per
// body. Shadowing in nested blocks would collide; the TS frontend
// currently produces flat name spaces inside an entry, so this is
// safe in practice. We could refine to lexical scopes later.

import type { Module, Stmt, Expr, LExpr, Var } from "../ir/index.js";
import { mapStmt, mapStmtChildren } from "./transform.js";

export function relinkVars(module: Module): Module {
  const values = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: relinkStmt(v.body) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: relinkStmt(v.entry.body) } };
    return v;
  });
  return { ...module, values };
}

function relinkStmt(body: Stmt): Stmt {
  // Pass 1: collect canonical Var per name from Declare sites.
  const canonical = new Map<string, Var>();
  const collect = (s: Stmt): Stmt => {
    if (s.kind === "Declare") {
      if (!canonical.has(s.var.name)) canonical.set(s.var.name, s.var);
    }
    return mapStmtChildren(s, { stmt: collect });
  };
  collect(body);
  if (canonical.size === 0) return body;

  // Pass 2: retarget every Var / LVar / Declare.var to the canonical
  // object for that name.
  const rewriteVar = (v: Var): Var => canonical.get(v.name) ?? v;
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "Var") {
      const c = canonical.get(e.var.name);
      if (c && c !== e.var) return { ...e, var: c };
    }
    return e;
  };
  const lexprFn = (l: LExpr): LExpr => {
    if (l.kind === "LVar") {
      const c = canonical.get(l.var.name);
      if (c && c !== l.var) return { ...l, var: c };
    }
    return l;
  };
  const stmtFn = (s: Stmt): Stmt => {
    if (s.kind === "Declare") {
      const c = rewriteVar(s.var);
      if (c !== s.var) return { ...s, var: c };
    }
    return s;
  };
  return mapStmt(body, { expr: exprFn, lexpr: lexprFn, stmt: stmtFn });
}
