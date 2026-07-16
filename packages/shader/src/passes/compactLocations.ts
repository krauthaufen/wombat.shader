// compactLocations — densely renumber surviving cross-stage varyings.
//
// `assignLocations` numbers vertex outputs / fragment inputs by
// declaration order, EARLY (before any pruning). `pruneCrossStage`
// then drops the interstage varyings no fragment reads — but it keeps
// the survivors' original `Location` numbers. A vertex entry that
// declared, say, 22 outputs "just in case" and had all but 5 pruned
// can leave a live varying pinned at `@location(20)`.
//
// That's fine for GLSL, but WebGPU caps inter-stage locations at
// `maxInterStageShaderVariables` (16 on many devices, so the highest
// legal index is 15). A live varying at location 20 makes pipeline
// creation fail — "location (20) that is too large. It should be less
// than (16)" — even though only a handful of varyings actually cross
// the stage boundary.
//
// This pass runs AFTER pruning and compacts the survivors: it builds a
// dense `name → location` map from the surviving vertex outputs (in
// first-appearance order) and rewrites both the vertex outputs and the
// matching fragment inputs, keeping the two sides in lockstep by name
// (the same by-name contract `pruneCrossStage` / `linkCrossStage`
// rely on). Builtins keep their `@builtin(...)` slot; fragment outputs
// (framebuffer targets) and vertex inputs (attributes, renumbered
// contiguously at emit) are left untouched.
//
// Idempotent: an already-dense module maps every location to itself
// and returns the input unchanged.

import type { EntryParameter, Module, ParamDecoration } from "../ir/index.js";

function isBuiltin(p: EntryParameter): boolean {
  return p.decorations.some((d) => d.kind === "Builtin");
}

export function compactLocations(module: Module): Module {
  // 1. Dense map from the surviving vertex-output varyings, in stable
  //    first-appearance order across all vertex entries.
  const loc = new Map<string, number>();
  let next = 0;
  for (const v of module.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "vertex") continue;
    for (const p of v.entry.outputs) {
      if (isBuiltin(p)) continue;
      if (!loc.has(p.name)) loc.set(p.name, next++);
    }
  }
  if (loc.size === 0) return module;

  const remap = (p: EntryParameter): EntryParameter => {
    if (isBuiltin(p)) return p;
    const target = loc.get(p.name);
    if (target === undefined) return p;
    let changed = false;
    const decorations = p.decorations.map((d): ParamDecoration => {
      if (d.kind === "Location") {
        if (d.value !== target) changed = true;
        return { kind: "Location", value: target };
      }
      return d;
    });
    return changed ? { ...p, decorations } : p;
  };

  let anyChange = false;
  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    // Vertex: remap outputs. Fragment: remap inputs (by the same map,
    // so the two sides stay aligned by name).
    if (v.entry.stage === "vertex") {
      const outputs = v.entry.outputs.map(remap);
      if (outputs.some((p, i) => p !== v.entry.outputs[i])) {
        anyChange = true;
        return { ...v, entry: { ...v.entry, outputs } };
      }
    } else if (v.entry.stage === "fragment") {
      const inputs = v.entry.inputs.map(remap);
      if (inputs.some((p, i) => p !== v.entry.inputs[i])) {
        anyChange = true;
        return { ...v, entry: { ...v.entry, inputs } };
      }
    }
    return v;
  });
  return anyChange ? { ...module, values: newValues } : module;
}
