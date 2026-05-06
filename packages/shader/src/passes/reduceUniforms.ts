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

  // Pass 1: filter members not referenced anywhere; drop empty
  // ValueDefs.
  const filtered: ValueDef[] = [];
  for (const v of module.values) {
    if (v.kind === "Uniform") {
      const kept = v.uniforms.filter((u) => names.has(u.name));
      if (kept.length === 0) continue;
      filtered.push(kept.length === v.uniforms.length
        ? v : { ...v, uniforms: kept as readonly UniformDecl[] });
      continue;
    }
    if (v.kind === "Sampler" || v.kind === "StorageBuffer") {
      if (names.has(v.name)) filtered.push(v);
      continue;
    }
    filtered.push(v);
  }

  // Pass 2: merge `Uniform` ValueDefs that share a buffer name —
  // without this, two effects that both pull
  // `uniform.ModelTrafo` (etc.) through the namespace import each
  // emit their own `Uniform` ValueDef. The WGSL emitter then
  // produces `struct _UB_uniform { … };` twice with identical
  // names and pipeline creation fails with "redeclaration of
  // '_UB_uniform'". Loose uniforms (no `buffer`) stay split — they
  // emit as individual `var<uniform>` bindings and don't share a
  // struct.
  const bufferGroups = new Map<string, UniformDecl[]>();
  const bufferOrder: string[] = []; // first-appearance order
  const out: ValueDef[] = [];
  for (const v of filtered) {
    if (v.kind === "Uniform") {
      const loose: UniformDecl[] = [];
      for (const u of v.uniforms) {
        if (u.buffer) {
          let bucket = bufferGroups.get(u.buffer);
          if (!bucket) {
            bucket = [];
            bufferGroups.set(u.buffer, bucket);
            bufferOrder.push(u.buffer);
          }
          // Dedupe by member name within the buffer.
          const existing = bucket.findIndex((x) => x.name === u.name);
          if (existing >= 0) bucket[existing] = u;
          else bucket.push(u);
        } else {
          loose.push(u);
        }
      }
      if (loose.length > 0) out.push({ ...v, uniforms: loose });
      continue;
    }
    out.push(v);
  }
  for (const buffer of bufferOrder) {
    out.push({ kind: "Uniform", uniforms: bufferGroups.get(buffer)! });
  }
  return { ...module, values: out };
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
