// uniformsToInputs — rename a set of Uniform-scope reads to Input-scope
// reads, drop the matching Uniform declarations, AND surface each
// renamed name as an `EntryParameter` on every Entry whose body reads
// it.
//
// Used by the heap renderer's decoder-composition path: every per-RO
// uniform becomes a "decoder output" that the prepended heap-decoder VS
// writes into the inter-stage carrier. From the user effect's
// perspective, all per-RO state is uniform; from the post-rename
// module's perspective, the only distinction left between attributes
// and former-uniforms is which heap arena the decoder loads from —
// neither shows up in the body's read kind anymore.
//
// The EntryParameter surfacing is load-bearing: `extractFusedEntry`
// builds the State struct (the carrier between fused-stage helpers)
// from each operand's `inputs` + `outputs`. Without an EntryParameter
// declaring the renamed name, the helper body's
// `ReadInput("Input", X)` rewrite would target a State field that
// doesn't exist. With it, the field is in State; the
// `writtenSoFar.has(...)` filter in `extractFusedEntry` then skips
// the declaration when an upstream operand wrote it (decoder VS), so
// the fused @vertex's input struct doesn't grow phantom location
// attributes.
//
// The pass is name-scoped on purpose: callers opt in for a specific
// subset (per-RO uniforms only, leaving ambient runtime-owned uniforms
// untouched).
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
//   4. For each Entry whose body now reads `Input.X` for a renamed X
//      that the entry didn't already declare as an input, add an
//      `EntryParameter` for it. Type is pulled from the dropped
//      Uniform decl; location is auto-assigned starting after the
//      entry's existing per-location inputs (collisions during
//      composition get resolved by `composeStages` /
//      `assignLocations`).

import type {
  EntryDef, EntryParameter, Expr, LExpr, Module, ParamDecoration,
  Stmt, Type, ValueDef,
} from "../ir/index.js";
import { mapStmt } from "./transform.js";
import { readInputs } from "./analysis.js";


export function uniformsToInputs(m: Module, names: ReadonlySet<string>): Module {
  if (names.size === 0) return m;

  // Snapshot (name → Type) from the Uniform decls we're about to drop
  // so step 4 can synthesise EntryParameters with the right type.
  const renamedTypes = new Map<string, Type>();
  for (const v of m.values) {
    if (v.kind !== "Uniform") continue;
    for (const u of v.uniforms) {
      if (names.has(u.name)) renamedTypes.set(u.name, u.type);
    }
  }

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

  const surfaceInputsOnEntry = (entry: EntryDef): EntryDef => {
    // What renamed names does the entry's body actually read as Input?
    const bodyInputs = new Set<string>();
    for (const sn of readInputs(entry.body).values()) {
      if (sn.scope === "Input" && names.has(sn.name)) bodyInputs.add(sn.name);
    }
    if (bodyInputs.size === 0) return entry;

    // Existing input parameter names so we don't double-declare.
    const haveInput = new Set(entry.inputs.map((p) => p.name));
    // Next free @location — one above the max already in use.
    let nextLoc = 0;
    for (const p of entry.inputs) {
      for (const d of p.decorations) {
        if (d.kind === "Location" && d.value >= nextLoc) nextLoc = d.value + 1;
      }
    }
    const additions: EntryParameter[] = [];
    for (const name of bodyInputs) {
      if (haveInput.has(name)) continue;
      const ty = renamedTypes.get(name);
      if (ty === undefined) {
        // The body reads a name that's in `names` but had no Uniform
        // decl carrying a type. Bail out hard — silently skipping
        // would leave the State struct missing a field and surface as
        // a downstream emit error far from the cause.
        throw new Error(
          `uniformsToInputs: body reads "Input.${name}" but no Uniform decl ` +
          `provided its IR Type. Either include "${name}" in the module's ` +
          `Uniform decls, or remove it from the rename set.`,
        );
      }
      const decorations: ParamDecoration[] = [
        { kind: "Location", value: nextLoc++ },
        // Former-uniforms are constant per primitive — `flat` is the
        // semantically correct interpolation mode. Also REQUIRED for
        // integer types (WGSL disallows perspective interpolation of
        // u32/i32). Using `flat` uniformly across all renamed names
        // keeps the VS-output / FS-input interpolation modes in sync
        // (the decoder writes its uniform outputs as `flat` too).
        { kind: "Interpolation", mode: "flat" },
      ];
      // Empty semantic: keeps `canonicaliseBySemantic` from treating this
      // synthesised input as the canonical port for the (former-)uniform
      // name. If another operand declares an output by a semantic that
      // happens to match, we don't want our type-narrow Input to
      // hijack it during fusion — leaving semantic="" lets that
      // operand's port keep its name unchanged.
      additions.push({ name, type: ty, semantic: "", decorations });
    }
    if (additions.length === 0) return entry;
    return { ...entry, inputs: [...entry.inputs, ...additions] };
  };

  const values: ValueDef[] = [];
  for (const v of m.values) {
    switch (v.kind) {
      case "Uniform": {
        const kept = v.uniforms.filter((u) => !names.has(u.name));
        if (kept.length === 0) continue; // drop the whole decl block
        values.push(kept.length === v.uniforms.length ? v : { ...v, uniforms: kept });
        continue;
      }
      case "Entry": {
        const renamed: EntryDef = { ...v.entry, body: renameBody(v.entry.body) };
        const surfaced = surfaceInputsOnEntry(renamed);
        values.push({ ...v, entry: surfaced });
        continue;
      }
      case "Function":
        values.push({ ...v, body: renameBody(v.body) });
        continue;
      default:
        values.push(v);
    }
  }
  return { ...m, values };
}
