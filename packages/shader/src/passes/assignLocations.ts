// assignLocations — fill in `Location` decorations on entry parameters
// that arrive without one.
//
// Background. Locations are an emit-time bookkeeping detail: WGSL/GLSL
// require `@location(N)` / `layout(location=N)` on every non-builtin
// stage I/O, and the linker passes (`linkCrossStage`,
// `linkFragmentOutputs`, `extractHelpers.renumberLocations`) all
// assume those numbers already exist. Historically the *frontends* —
// the TS Compiler-API `parseShader`, the Vite inline plugin, and any
// downstream IR producer (the F# Fable plugin) — assigned them by
// declaration order.
//
// That coupling was leaky: every new IR producer had to re-implement
// the same monotonic numbering, and any producer that *didn't* (the
// F# plugin, by design) caused WGSL emitters to drop literal
// `@location(0)` onto every undecorated varying — collisions galore.
//
// This pass moves the numbering into the pipeline. Each entry is
// processed independently; within an entry, `inputs` and `outputs` are
// numbered separately. Parameters that already carry a Location are
// left alone (so frontends that *do* assign locations, or framebuffer
// signature pinning that landed earlier, still win). Parameters that
// carry a `Builtin` decoration are skipped — they live on `@builtin(K)`
// and don't take a location slot.
//
// Run order. This pass runs at the start of the pipeline, right after
// `liftReturns`, so every downstream pass — including `composeStages`,
// `linkCrossStage`, `linkFragmentOutputs`, and the helper extractors —
// sees a fully-located IR. Idempotent: re-running on already-located
// IR is a no-op.

import type { EntryParameter, Module, ParamDecoration } from "../ir/index.js";

export function assignLocations(module: Module): Module {
  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    const inputs = numberParams(v.entry.inputs);
    const outputs = numberParams(v.entry.outputs);
    if (inputs === v.entry.inputs && outputs === v.entry.outputs) return v;
    return { ...v, entry: { ...v.entry, inputs, outputs } };
  });
  return { ...module, values: newValues };
}

function numberParams(params: readonly EntryParameter[]): readonly EntryParameter[] {
  // First pass: find the next free Location given any decorations
  // already in place, so we extend rather than collide.
  let nextLoc = 0;
  for (const p of params) {
    for (const d of p.decorations) {
      if (d.kind === "Location" && d.value >= nextLoc) nextLoc = d.value + 1;
    }
  }

  let changed = false;
  const out = params.map((p) => {
    const hasBuiltin = p.decorations.some((d) => d.kind === "Builtin");
    const hasLoc = p.decorations.some((d) => d.kind === "Location");
    if (hasBuiltin || hasLoc) return p;
    const decoration: ParamDecoration = { kind: "Location", value: nextLoc++ };
    changed = true;
    return { ...p, decorations: [...p.decorations, decoration] };
  });
  return changed ? out : params;
}
