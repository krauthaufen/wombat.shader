// pruneCrossStage — removes vertex outputs (and their compute chains)
// that no fragment stage reads, AND symmetrically drops the matching
// fragment input declarations so VS-output and FS-input location sets
// stay in lockstep.
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
//   4. For each fragment entry, drop input declarations whose names
//      aren't live (the body doesn't read them). Without this step
//      the FS struct still emits `@location(N) v_x` for an attribute
//      whose VS-output side just disappeared in step 2 — WebGPU
//      pipeline creation fails silently with a location-mismatch
//      validation error.
//   5. Re-run from (1) until the live-input set stops changing
//      (composed pipelines may have a vertex output that *was* read
//      by an interior fragment that we just dropped).
//
// We do NOT drop fragment outputs (those are the final framebuffer
// attachments — the runtime decides which targets matter).

import type {
  EntryDef,
  Module,
  Stmt,
} from "../ir/index.js";
import { dceStmt } from "./dce.js";
import { mapStmt } from "./transform.js";
import { readInputs as readInputsAnalysis } from "./analysis.js";

export function pruneCrossStage(module: Module): Module {
  let current = module;
  // Iterate to a fixed point. Each pass returns whether it changed
  // anything; we stop the moment a sweep is a no-op. Bounded
  // defensively at 16 iterations in case a future bug introduces
  // oscillation.
  for (let iter = 0; iter < 16; iter++) {
    const { module: next, changed } = pruneOnce(current);
    if (!changed) return next;
    current = next;
  }
  return current;
}

interface PruneResult {
  readonly module: Module;
  readonly changed: boolean;
}

function pruneOnce(module: Module): PruneResult {
  const liveInputs = collectLiveInputs(module);
  let changed = false;
  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    const e = v.entry;
    if (e.stage === "vertex") {
      const pruned = pruneVertex(e, liveInputs);
      if (pruned !== e) changed = true;
      return { ...v, entry: pruned };
    }
    if (e.stage === "fragment") {
      const pruned = pruneFragmentInputs(e, liveInputs);
      if (pruned !== e) changed = true;
      return { ...v, entry: pruned };
    }
    return v;
  });
  return { module: changed ? { ...module, values: newValues } : module, changed };
}

/**
 * Drop fragment-stage `inputs` declarations whose names the body
 * doesn't read. Without this the emitted FS struct keeps a
 * `@location(N)` declaration for a varying that the VS-output side
 * has been pruned away from in `pruneVertex`, and WebGPU pipeline
 * creation fails with a silent inter-stage location mismatch.
 * Builtin inputs (e.g. `position` / `frag_depth`) stay regardless —
 * they're supplied by the pipeline, not a previous stage.
 */
function pruneFragmentInputs(e: EntryDef, live: Set<string>): EntryDef {
  const keptInputs = e.inputs.filter((p) => {
    const isBuiltin = p.decorations.some((d) => d.kind === "Builtin");
    return isBuiltin || live.has(p.name);
  });
  if (keptInputs.length === e.inputs.length) return e;
  return { ...e, inputs: keptInputs };
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
  if (keptOutputs.length === e.outputs.length) return e;

  const dropped = new Set<string>();
  for (const p of e.outputs) {
    if (!keptOutputs.includes(p)) dropped.add(p.name);
  }
  // 2. Replace WriteOutput(droppedName) with Nop, then DCE so
  //    variables that fed only those writes disappear.
  const body = dceStmt(stripWritesTo(e.body, dropped));

  return { ...e, outputs: keptOutputs, body };
}

/**
 * Drop declared vertex INPUTS the body no longer reads. Runs as
 * a separate pass AFTER `linkHelpers` has had a chance to clean
 * up state-init writes (`s.X = in.X`) for state fields that lost
 * their consumers when `pruneCrossStage` dropped the
 * corresponding outputs. Without this step the EntryDef still
 * asks the runtime for unused vertex attributes (e.g.
 * `DiffuseColorUTangents` for an `effect(trafo, simpleLighting)`
 * pipeline that doesn't consume them) and the rendering layer
 * errors with `prepareRenderObject: missing vertex attribute
 * "X"` at draw time.
 *
 * Builtin inputs (`@builtin(vertex_index)` etc.) stay regardless
 * — they're supplied by the pipeline, not a vertex buffer.
 */
export function pruneVertexInputs(module: Module): Module {
  let changed = false;
  const newValues = module.values.map((v) => {
    if (v.kind !== "Entry" || v.entry.stage !== "vertex") return v;
    const e = v.entry;
    const reads = new Set<string>();
    for (const sn of readInputsAnalysis(e.body).values()) {
      if (sn.scope === "Input") reads.add(sn.name);
    }
    const keptInputs = e.inputs.filter((p) => {
      const isBuiltin = p.decorations.some((d) => d.kind === "Builtin");
      return isBuiltin || reads.has(p.name);
    });
    if (keptInputs.length === e.inputs.length) return v;
    changed = true;
    return { ...v, entry: { ...e, inputs: keptInputs } };
  });
  return changed ? { ...module, values: newValues } : module;
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

