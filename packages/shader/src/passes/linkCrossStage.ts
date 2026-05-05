// linkCrossStage — FShade-style cross-stage linker.
//
// Runs AFTER composeStages and BEFORE linkFragmentOutputs. Two jobs:
//
//   1. Match VS outputs to FS inputs by SEMANTIC (with name fallback).
//      When a VS output's semantic matches an FS input's semantic but
//      the names differ, the FS input is renamed to the VS output's
//      name (FShade policy: rename the input). Every
//      `ReadInput("Input", oldName)` in the FS body becomes
//      `ReadInput("Input", vsOutputName)`.
//
//   2. Auto pass-through. If an FS input has no matching VS output
//      but a vertex attribute (VS input) with the same semantic
//      exists, synthesise a carrier in the VS:
//
//          var _pt_<name>: T;
//          _pt_<name> = ReadInput("Input", attr);
//          WriteOutput(<name>, _pt_<name>);
//
//      and add `<name>` to the VS outputs list. A console.warn
//      diagnostic announces the injection.
//
// Auto-passthrough trigger condition: the FS unconditionally references
// the input via `ReadInput("Input", name)` somewhere in its body
// (static-presence check). We don't try to pattern-match conditional
// reads — if the FS has the call site at all, we feed it.
//
// Builtin FS inputs (e.g. fragCoord) are skipped — those are handled by
// the runtime / emitter.

import type {
  EntryDef,
  EntryParameter,
  Expr,
  LExpr,
  Module,
  Stmt,
  Var,
} from "../ir/index.js";
import { mapStmt } from "./transform.js";
import { readInputs } from "./analysis.js";

export function linkCrossStage(module: Module): Module {
  const entries = module.values.flatMap((v) => v.kind === "Entry" ? [v.entry] : []);
  const vs = entries.find((e) => e.stage === "vertex");
  const fs = entries.find((e) => e.stage === "fragment");
  if (!fs) return module;

  // Build VS-output and VS-input lookups by paramKey (semantic ?? name).
  const vsOutputsBySem = new Map<string, EntryParameter>();
  const vsOutputsByName = new Map<string, EntryParameter>();
  const vsInputsBySem = new Map<string, EntryParameter>();
  if (vs) {
    for (const o of vs.outputs) {
      if (o.decorations.some((d) => d.kind === "Builtin")) continue;
      const k = paramKey(o);
      if (!vsOutputsBySem.has(k)) vsOutputsBySem.set(k, o);
      vsOutputsByName.set(o.name, o);
    }
    for (const i of vs.inputs) {
      if (i.decorations.some((d) => d.kind === "Builtin")) continue;
      const k = paramKey(i);
      if (!vsInputsBySem.has(k)) vsInputsBySem.set(k, i);
    }
  }

  // Which FS-input names actually appear as ReadInput("Input", name)
  // in the FS body. The static-presence check that gates auto-passthrough.
  const fsReads = new Set<string>();
  for (const sn of readInputs(fs.body).values()) {
    if (sn.scope === "Input") fsReads.add(sn.name);
  }

  // Plan: per FS input, decide whether to rename it to a VS-output's name
  // and/or synthesise a VS pass-through.
  const renames = new Map<string, string>(); // fsInputName → newName
  const passthroughs: Array<{
    name: string;          // cross-stage output name (= FS-input name after rename)
    type: import("../ir/index.js").Type;
    fromAttr: string;      // VS-input name to read
    semantic: string;
  }> = [];

  const newFsInputs: EntryParameter[] = [];
  for (const fi of fs.inputs) {
    if (fi.decorations.some((d) => d.kind === "Builtin")) {
      newFsInputs.push(fi);
      continue;
    }
    const k = paramKey(fi);
    // 1. Match by semantic (when set), else by name.
    let target = vsOutputsBySem.get(k);
    if (!target && fi.semantic === fi.name) {
      // No semantic key collision possible; already covered.
    }
    // Fallback to name-only match if semantic match failed.
    if (!target) target = vsOutputsByName.get(fi.name);

    if (target) {
      // Adopt the target VS-output's Location for this FS input. The
      // FS-input's original Location was inferred independently (by
      // declaration order in the parameter type) and need not match
      // the VS output's; without this rewrite the linker would emit
      // VS and FS halves that disagree on which location each varying
      // lives at — silently producing crossed-wires varyings.
      const targetLoc = target.decorations.find((d) => d.kind === "Location");
      const decorations = targetLoc
        ? [
            ...fi.decorations.filter((d) => d.kind !== "Location"),
            targetLoc,
          ]
        : fi.decorations;
      const renamed = target.name !== fi.name;
      if (renamed) renames.set(fi.name, target.name);
      newFsInputs.push({
        ...fi,
        ...(renamed ? { name: target.name } : {}),
        decorations,
      });
      continue;
    }

    // 2. No VS output — try VS attribute (only if FS unconditionally reads).
    if (vs && fsReads.has(fi.name)) {
      const attr = vsInputsBySem.get(k);
      if (attr) {
        // Inject pass-through. Cross-stage name = FS input's current name.
        passthroughs.push({
          name: fi.name, type: fi.type, fromAttr: attr.name, semantic: fi.semantic,
        });
        // eslint-disable-next-line no-console
        console.warn(
          `linkCrossStage: auto pass-through inserted: VS attribute \`${attr.name}\` -> cross-stage \`${fi.name}\``,
        );
        newFsInputs.push(fi);
        continue;
      }
    }

    // 3. Genuine miss — leave as-is for downstream diagnostics.
    newFsInputs.push(fi);
  }

  // Apply FS-input renames to its body.
  const newFsBody = renames.size > 0 ? renameInputReads(fs.body, renames) : fs.body;

  // Build the new VS with appended pass-throughs.
  let newVs = vs;
  if (vs && passthroughs.length > 0) {
    newVs = appendPassThroughs(vs, passthroughs);
  }

  // Reassemble the module.
  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    if (v.entry === fs) {
      return {
        ...v,
        entry: { ...fs, inputs: newFsInputs, body: newFsBody },
      };
    }
    if (newVs && v.entry === vs) {
      return { ...v, entry: newVs };
    }
    return v;
  });
  return { ...module, values: newValues };
}

/**
 * Canonical matching key. Semantic wins when set and non-empty;
 * otherwise fall back to the parameter's `name`. This is what couples
 * VS outputs to FS inputs across the cross-stage boundary.
 */
export function paramKey(p: EntryParameter): string {
  if (p.semantic && p.semantic.length > 0) return p.semantic;
  return p.name;
}

function renameInputReads(s: Stmt, renames: ReadonlyMap<string, string>): Stmt {
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "ReadInput" && e.scope === "Input") {
      const r = renames.get(e.name);
      if (r !== undefined) return { ...e, name: r };
    }
    return e;
  };
  const lexprFn = (l: LExpr): LExpr => {
    if (l.kind === "LInput" && l.scope === "Input") {
      const r = renames.get(l.name);
      if (r !== undefined) return { ...l, name: r };
    }
    return l;
  };
  return mapStmt(s, { expr: exprFn, lexpr: lexprFn });
}

function appendPassThroughs(
  vs: EntryDef,
  passes: ReadonlyArray<{ name: string; type: import("../ir/index.js").Type; fromAttr: string; semantic: string }>,
): EntryDef {
  const stmts: Stmt[] = [];
  const newOutputs: EntryParameter[] = [...vs.outputs];
  const existingOut = new Set(vs.outputs.map((o) => o.name));

  // Determine the next free Location for the new cross-stage outputs.
  let nextLoc = 0;
  for (const o of vs.outputs) {
    for (const d of o.decorations) {
      if (d.kind === "Location" && d.value >= nextLoc) nextLoc = d.value + 1;
    }
  }

  for (const p of passes) {
    if (existingOut.has(p.name)) continue;
    const carrier: Var = { name: `_pt_${p.name}`, type: p.type, mutable: true };
    stmts.push({ kind: "Declare", var: carrier });
    stmts.push({
      kind: "Write",
      target: { kind: "LVar", var: carrier, type: p.type },
      value: { kind: "ReadInput", scope: "Input", name: p.fromAttr, type: p.type },
    });
    stmts.push({
      kind: "WriteOutput", name: p.name,
      value: { kind: "Expr", value: { kind: "Var", var: carrier, type: p.type } },
    });
    newOutputs.push({
      name: p.name,
      type: p.type,
      semantic: p.semantic,
      decorations: [{ kind: "Location", value: nextLoc++ }],
    });
    existingOut.add(p.name);
  }

  const body: Stmt = vs.body.kind === "Sequential"
    ? { kind: "Sequential", body: [...vs.body.body, ...stmts] }
    : { kind: "Sequential", body: [vs.body, ...stmts] };

  return { ...vs, outputs: newOutputs, body };
}
