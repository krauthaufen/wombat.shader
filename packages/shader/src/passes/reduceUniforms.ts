// reduceUniforms — drops Uniform / Sampler / StorageBuffer ValueDefs that
// no surviving Entry actually references.
//
// The IR has two kinds of "module-level binding":
//
//  1. `Uniform` declarations, accessed via `ReadInput(scope: "Uniform", name)`.
//     We track which names appear in any Entry's body. Decls whose name is
//     never read get dropped. Multi-decl `Uniform` blocks are kept if any
//     of their members survive.
//
//  2. `Sampler` / `StorageBuffer` declarations, accessed by name through
//     `ReadInput(scope: "Uniform", name)` (samplers, treated as a uniform
//     binding by the backends) or as identifiers in the body for storage
//     buffers (we use `Var.name` matching since the frontend interns the
//     storage buffer as a Var). To remain backend-agnostic, we collect:
//       - every ReadInput in any scope, by name
//       - every Var read by name (for storage-buffer-as-Var pattern)
//
// Conservative: if a definition's name is mentioned anywhere in the
// surviving entries / functions, we keep it. False positives leave a
// dead binding; false negatives would break linking — we err on
// retention.

import type {
  Expr,
  LExpr,
  Module,
  Stmt,
  UniformDecl,
  ValueDef,
} from "../ir/index.js";
import { visitExprChildren, visitLExprChildren, visitStmt } from "../ir/index.js";

export function reduceUniforms(module: Module): Module {
  const names = collectReferencedNames(module);

  const newValues: ValueDef[] = [];
  for (const v of module.values) {
    if (v.kind === "Uniform") {
      const kept = v.uniforms.filter((u) => names.has(u.name));
      if (kept.length > 0) {
        // Keep the same buffer-grouping shape; if everyone in a buffer
        // survives we don't change anything.
        if (kept.length === v.uniforms.length) newValues.push(v);
        else newValues.push({ ...v, uniforms: kept as readonly UniformDecl[] });
      }
      continue;
    }
    if (v.kind === "Sampler" || v.kind === "StorageBuffer") {
      if (names.has(v.name)) newValues.push(v);
      continue;
    }
    newValues.push(v);
  }
  return { ...module, values: newValues };
}

function collectReferencedNames(module: Module): Set<string> {
  const names = new Set<string>();

  const onExpr = (e: Expr): void => {
    if (e.kind === "ReadInput") names.add(e.name);
    if (e.kind === "Var") names.add(e.var.name);
    visitExprChildren(e, onExpr);
  };
  const onLExpr = (l: LExpr): void => {
    // LVar names also reference module-level decls (storage buffers
    // accessed as `name[i] = ...`).
    if (l.kind === "LVar") names.add(l.var.name);
    if (l.kind === "LInput") names.add(l.name);
    visitLExprChildren(l, onLExpr, onExpr);
  };

  for (const v of module.values) {
    if (v.kind === "Function") {
      walkStmt(v.body, onExpr, onLExpr);
    } else if (v.kind === "Entry") {
      walkStmt(v.entry.body, onExpr, onLExpr);
    } else if (v.kind === "Constant") {
      if (v.init.kind === "Expr") onExpr(v.init.value);
      else for (const e of v.init.values) onExpr(e);
    }
  }

  return names;
}

function walkStmt(s: Stmt, onExpr: (e: Expr) => void, onLExpr: (l: LExpr) => void): void {
  visitStmt(s, {
    expr: { pre: onExpr },
    preStmt(stmt) {
      if (stmt.kind === "Write" || stmt.kind === "Increment" || stmt.kind === "Decrement") {
        onLExpr(stmt.target);
      }
    },
  });
}
