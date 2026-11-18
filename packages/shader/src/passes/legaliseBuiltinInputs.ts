// legaliseBuiltinInputs — a vertex input can never be `@builtin(position)`.
//
// Background. `@builtin(position)` is two different slots depending on
// which side of the rasteriser you are on: on a VERTEX OUTPUT it is the
// clip-space position the rasteriser consumes, and on a FRAGMENT INPUT
// it is `gl_FragCoord`, the screen-space pixel centre. On a VERTEX INPUT
// it does not exist at all — the vertex position arrives as an ordinary
// attribute, which FShade (and `SEMANTIC_BUILTIN_MAP`, whose only vertex
// entry is `Positions|vertex|out`) spells `Positions`.
//
// The TS frontends get this right because the Vite plugin resolves the
// builtin through `semanticToBuiltin`, which is keyed by
// `semantic|stage|direction`. A hand-built IR producer has no such
// guard: the F# Fable plugin's `fshadeAttrLookup` maps the `[<Position>]`
// attribute to `("Positions", builtin position)` for inputs and outputs
// alike, so
//
//     type SkyVIn = { [<Position>] pos : V4f }
//
// arrives here as an input named `Positions` carrying
// `Builtin "position"`, with the body reading it as
// `ReadInput("Builtin", "position")`.
//
// Two things then go wrong downstream. The WGSL emitter drops builtin
// inputs from the attribute struct, so the vertex buffer binding for the
// quad silently disappears. And `extractHelpers.canonicaliseBuiltinOutputs`
// — the pass that makes every spelling of the position output ONE port —
// skips an operand whose `Positions` input is builtin-decorated, on the
// (correct, for fragments) assumption that a builtin input is a real
// builtin and not the earlier stage's write. The stage's own position
// output therefore never folds onto the canonical `gl_Position`, and a
// fused vertex entry ends up declaring `@builtin(position)` twice, which
// WGSL rejects outright.
//
// So: strip the decoration on vertex inputs, and rewrite the body's
// builtin read into an ordinary attribute read. `assignLocations` then
// gives the attribute a location like any other, and the existing
// builtin-output canonicalisation sees the shape it expects.
//
// Run order. Before `assignLocations` (which must see the input as
// location-less rather than builtin) and therefore before fusion.
// Idempotent: an entry with no builtin-position vertex input is returned
// unchanged, by identity.

import type { Expr, Module, ParamDecoration, Stmt } from "../ir/index.js";
import { mapStmt } from "./transform.js";

export function legaliseBuiltinInputs(module: Module): Module {
  let changed = false;
  const values = module.values.map((v) => {
    if (v.kind !== "Entry" || v.entry.stage !== "vertex") return v;
    const entry = v.entry;
    // The attribute the builtin read has to be redirected to. Several
    // inputs could carry it (one record field per spelling); they all
    // denote the same attribute, so the first one wins and the rest are
    // demoted alongside it.
    let attribute: string | undefined;
    const inputs = entry.inputs.map((p) => {
      const isPos = p.decorations.some(
        (d: ParamDecoration) => d.kind === "Builtin" && d.value === "position",
      );
      if (!isPos) return p;
      if (attribute === undefined) attribute = p.name;
      return { ...p, decorations: p.decorations.filter((d) => d.kind !== "Builtin") };
    });
    if (attribute === undefined) return v;
    changed = true;
    const name = attribute;
    const body: Stmt = mapStmt(entry.body, {
      expr: (e: Expr): Expr =>
        e.kind === "ReadInput" && e.scope === "Builtin" && e.name === "position"
          ? { ...e, scope: "Input", name }
          : e,
    });
    return { ...v, entry: { ...entry, inputs, body } };
  });
  return changed ? { ...module, values } : module;
}
