// linkFragmentOutputs — re-pin fragment outputs to a target framebuffer
// signature.
//
// Given a `FragmentOutputLayout` (name → location), each fragment entry's
// non-builtin outputs are matched by name:
//   - in layout: the output's `Location` decoration is replaced with the
//     layout's location.
//   - not in layout: the output is dropped, and any `WriteOutput("name", _)`
//     in the entry body is rewritten to `Nop`. `dceStmt` then cleans up
//     orphaned variable assignments / dead declarations.
// Builtins (e.g. `frag_depth`) pass through untouched.
//
// Surviving outputs are sorted by location for deterministic emission.

import type {
  EntryDef,
  EntryParameter,
  Module,
  ParamDecoration,
  Stmt,
} from "../ir/index.js";
import { mapStmt } from "./transform.js";
import { dceStmt } from "./dce.js";

export interface FragmentOutputLayout {
  /** name → location. Outputs not present here are dropped. */
  readonly locations: ReadonlyMap<string, number>;
}

export function linkFragmentOutputs(module: Module, layout: FragmentOutputLayout): Module {
  const values = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    if (v.entry.stage !== "fragment") return v;
    return { ...v, entry: linkEntry(v.entry, layout) };
  });
  return { ...module, values };
}

function linkEntry(entry: EntryDef, layout: FragmentOutputLayout): EntryDef {
  const dropped = new Set<string>();
  const newOutputs: EntryParameter[] = [];

  for (const out of entry.outputs) {
    const isBuiltin = out.decorations.some((d) => d.kind === "Builtin");
    if (isBuiltin) {
      newOutputs.push(out);
      continue;
    }
    const loc = layout.locations.get(out.name);
    if (loc === undefined) {
      dropped.add(out.name);
      continue;
    }
    const decorations: ParamDecoration[] = [
      ...out.decorations.filter((d) => d.kind !== "Location"),
      { kind: "Location", value: loc },
    ];
    newOutputs.push({ ...out, decorations });
  }

  // Sort surviving outputs deterministically: builtins last (they have no
  // location), non-builtins by ascending location.
  newOutputs.sort((a, b) => {
    const la = locOf(a);
    const lb = locOf(b);
    if (la === undefined && lb === undefined) return a.name.localeCompare(b.name);
    if (la === undefined) return 1;
    if (lb === undefined) return -1;
    return la - lb;
  });

  let body = entry.body;
  if (dropped.size > 0) {
    body = stripDroppedWrites(body, dropped);
    body = dceStmt(body);
  }

  return { ...entry, outputs: newOutputs, body };
}

function locOf(p: EntryParameter): number | undefined {
  for (const d of p.decorations) if (d.kind === "Location") return d.value;
  return undefined;
}

function stripDroppedWrites(s: Stmt, dropped: ReadonlySet<string>): Stmt {
  const transform = (child: Stmt): Stmt => {
    if (child.kind === "WriteOutput" && dropped.has(child.name)) {
      return { kind: "Nop" };
    }
    if (
      child.kind === "Write"
      && child.target.kind === "LInput"
      && child.target.scope === "Output"
      && dropped.has(child.target.name)
    ) {
      return { kind: "Nop" };
    }
    return child;
  };
  return transform(mapStmt(s, { stmt: transform }));
}
