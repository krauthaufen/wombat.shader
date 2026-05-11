// uniformsToInputs — rename a set of Uniform-scope reads to Input-scope
// reads and drop the matching Uniform declarations from the module.
//
// Used by the heap renderer's decoder-composition path: every per-RO
// uniform becomes a "decoder output" that the prepended heap-decoder VS
// writes into the inter-stage carrier. From the user effect's
// perspective, all per-RO state is uniform; from the post-rename
// module's perspective, the only distinction left between attributes
// and former-uniforms is which heap arena the decoder loads from —
// neither shows up in the body's read kind anymore.
//
// The pass is name-scoped on purpose: callers can opt-in for a
// specific subset (e.g. per-RO uniforms only, leaving ambient runtime-
// owned uniforms untouched).
//
// Mechanically:
//   1. ReadInput("Uniform", X) → ReadInput("Input", X) for every X in
//      `names`, walking every Entry and Function body.
//   2. Same on the LExpr side (LInput) for completeness — though
//      writes to uniforms are illegal upstream, this keeps the pass
//      symmetric with `substituteInputsInStage`.
//   3. Filter UniformDecls so any decl whose name appears in `names`
//      is removed; if a ValueDef of kind "Uniform" ends up empty, drop
//      the whole ValueDef so emit doesn't produce a dangling block.

import type { Expr, LExpr, Module, Stmt, ValueDef } from "../ir/index.js";
import { mapStmt } from "./transform.js";

export function uniformsToInputs(m: Module, names: ReadonlySet<string>): Module {
  if (names.size === 0) return m;

  const renameExpr = (e: Expr): Expr =>
    e.kind === "ReadInput" && e.scope === "Uniform" && names.has(e.name)
      ? { ...e, scope: "Input" }
      : e;
  const renameLExpr = (l: LExpr): LExpr =>
    l.kind === "LInput" && l.scope === "Uniform" && names.has(l.name)
      ? { ...l, scope: "Input" }
      : l;
  const renameBody = (s: Stmt): Stmt =>
    mapStmt(s, { expr: renameExpr, lexpr: renameLExpr });

  const values: ValueDef[] = [];
  for (const v of m.values) {
    switch (v.kind) {
      case "Uniform": {
        const kept = v.uniforms.filter((u) => !names.has(u.name));
        if (kept.length === 0) continue; // drop the whole decl block
        values.push(kept.length === v.uniforms.length ? v : { ...v, uniforms: kept });
        continue;
      }
      case "Entry":
        values.push({ ...v, entry: { ...v.entry, body: renameBody(v.entry.body) } });
        continue;
      case "Function":
        values.push({ ...v, body: renameBody(v.body) });
        continue;
      default:
        values.push(v);
    }
  }
  return { ...m, values };
}
