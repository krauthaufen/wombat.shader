// pruneCrossStage — removes vertex outputs (and their compute chains)
// that no fragment stage reads.
//
// Problem statement. A `vertex + fragment` pipeline declares vertex
// outputs by `name` and fragment inputs that read them via
// `ReadInput(scope: "Input", name)`. Effects compose by writing too
// many vertex outputs "just in case"; the pipeline-link pass finds
// what fragment actually consumes and prunes the rest.
//
// Algorithm (fixed point):
//
//   1. Collect every `ReadInput("Input", name)` referenced by any
//      surviving fragment (or compute) entry. These are the live
//      fragment inputs.
//   2. For each vertex entry, drop output declarations whose names
//      aren't live (and aren't builtin — `position` etc. are
//      pipeline-mandatory and never pruned).
//   3. Replace every `WriteOutput(deadName, ...)` in the vertex body
//      with `Nop`. Run DCE.
//   4. Re-run from (1) until the live-input set stops changing
//      (composed pipelines may have a vertex output that *was* read
//      by an interior fragment that we just dropped).
//
// We do NOT drop fragment outputs (those are the final framebuffer
// attachments — the runtime decides which targets matter).

import type {
  EntryDef,
  Module,
  Stmt,
} from "@aardworx/wombat.shader-ir";
import { dceStmt } from "./dce.js";
import { mapStmt } from "./transform.js";
import { readInputs as readInputsAnalysis } from "./analysis.js";

export function pruneCrossStage(module: Module): Module {
  let current = module;
  // Iterate to a fixed point. Bound the iterations defensively.
  for (let iter = 0; iter < 16; iter++) {
    const next = pruneOnce(current);
    if (sameLayout(current, next)) return next;
    current = next;
  }
  return current;
}

function pruneOnce(module: Module): Module {
  const liveInputs = collectLiveInputs(module);

  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    if (v.entry.stage !== "vertex") return v;
    return { ...v, entry: pruneVertex(v.entry, liveInputs) };
  });

  return { ...module, values: newValues };
}

/**
 * The set of input *names* that any fragment / compute entry reads.
 * Vertex stages don't consume the previous-stage's output (they read
 * VBO inputs which the runtime supplies), so we exclude them.
 */
function collectLiveInputs(module: Module): Set<string> {
  const live = new Set<string>();
  for (const v of module.values) {
    if (v.kind !== "Entry") continue;
    const e = v.entry;
    if (e.stage === "vertex") continue;
    const inputs = readInputsAnalysis(e.body);
    for (const sn of inputs.values()) {
      if (sn.scope === "Input") live.add(sn.name);
    }
  }
  return live;
}

function pruneVertex(e: EntryDef, live: Set<string>): EntryDef {
  // 1. Determine which output declarations to keep:
  //    - builtin outputs (e.g. position) always
  //    - named outputs whose name is in `live`
  const keptOutputs = e.outputs.filter((p) => {
    const isBuiltin = p.decorations.some((d) => d.kind === "Builtin");
    return isBuiltin || live.has(p.name);
  });
  if (keptOutputs.length === e.outputs.length) {
    // No outputs dropped — body unchanged.
    return e;
  }

  const dropped = new Set<string>();
  for (const p of e.outputs) {
    if (!keptOutputs.includes(p)) dropped.add(p.name);
  }

  // 2. Replace WriteOutput(droppedName) with Nop.
  const stripped = stripWritesTo(e.body, dropped);

  // 3. DCE — variables / computations that fed only those writes
  //    become unused and disappear.
  const cleaned = dceStmt(stripped);

  return { ...e, outputs: keptOutputs, body: cleaned };
}

function stripWritesTo(s: Stmt, dropped: ReadonlySet<string>): Stmt {
  const transform = (child: Stmt): Stmt => {
    if (child.kind === "WriteOutput" && dropped.has(child.name)) {
      return { kind: "Nop" };
    }
    if (child.kind === "Write" && child.target.kind === "LInput"
      && child.target.scope === "Output" && dropped.has(child.target.name)) {
      return { kind: "Nop" };
    }
    return child;
  };
  return transform(mapStmt(s, { stmt: transform }));
}

/**
 * Cheap equality on the parts we care about: per-entry output names
 * and body string-shape. A more thorough fixed-point check would diff
 * IR; this is good enough for typical pipelines (a handful of stages,
 * a couple of iterations to converge).
 */
function sameLayout(a: Module, b: Module): boolean {
  if (a.values.length !== b.values.length) return false;
  for (let i = 0; i < a.values.length; i++) {
    const x = a.values[i]!, y = b.values[i]!;
    if (x.kind !== y.kind) return false;
    if (x.kind === "Entry" && y.kind === "Entry") {
      if (x.entry.outputs.length !== y.entry.outputs.length) return false;
      for (let j = 0; j < x.entry.outputs.length; j++) {
        if (x.entry.outputs[j]!.name !== y.entry.outputs[j]!.name) return false;
      }
      if (JSON.stringify(x.entry.body) !== JSON.stringify(y.entry.body)) return false;
    }
  }
  return true;
}
