// composeStages — fuse same-stage entries; package vertex+fragment as a pipeline.
//
// Composition rules (matching FShade for the Web subset):
//
//   v + v → v  (sequential: A's outputs feed B's inputs by name)
//   f + f → f  (sequential: A's outputs feed B's inputs by name)
//   v + f → vertex+fragment Module (no IR-level fusion; just bundling)
//   compute    → standalone, never composed
//
// For sequential same-stage fusion (A then B), B reads inputs that A
// writes. We rename B's `ReadInput("Input", name)` references to
// match A's `WriteOutput(name)` writes. After substitution the two
// bodies are concatenated; B's inputs that A doesn't supply remain as
// genuine inputs (and accumulate into the merged entry's input list).
//
// Following FShade: composition KEEPS every output. Carriers handle
// dataflow between adjacent stages; outputs that aren't piped survive
// in the merged entry. Same-named outputs across A and B follow
// last-wins: B's output replaces A's. The fragment-output linker
// (linkFragmentOutputs) runs at compile time against the target
// framebuffer signature and DCE's any output the FB doesn't want.
//
// We don't try to stitch multiple compute kernels — they're dispatched
// independently at runtime.

import type {
  EntryDef,
  EntryParameter,
  Expr,
  LExpr,
  Module,
  Stage,
  Stmt,
  ValueDef,
  Var,
} from "../ir/index.js";
import { mapStmt } from "./transform.js";
import { dceStmt } from "./dce.js";
import { readInputs } from "./analysis.js";

/**
 * Sequentially compose same-stage entries within a Module.
 * If multiple `vertex` entries exist they're fused into one;
 * same for `fragment`.
 */
export function composeStages(module: Module): Module {
  const groups: Record<Stage, EntryDef[]> = { vertex: [], fragment: [], compute: [] };
  const otherValues: ValueDef[] = [];

  for (const v of module.values) {
    if (v.kind === "Entry") groups[v.entry.stage].push(v.entry);
    else otherValues.push(v);
  }

  const fused: ValueDef[] = [];
  for (const stage of ["vertex", "fragment", "compute"] as const) {
    const entries = groups[stage];
    if (entries.length === 0) continue;
    if (entries.length === 1 || stage === "compute") {
      for (const e of entries) fused.push({ kind: "Entry", entry: e });
      continue;
    }
    let merged = entries[0]!;
    for (let i = 1; i < entries.length; i++) {
      merged = fusePair(merged, entries[i]!);
    }
    fused.push({ kind: "Entry", entry: merged });
  }

  return { ...module, values: [...otherValues, ...fused] };
}

/**
 * Fuse `a` then `b`. `b`'s `ReadInput("Input", name)` references that
 * match an `a` output get replaced by reads of synthetic locals that
 * carry `a`'s written value through.
 */
function fusePair(a: EntryDef, b: EntryDef): EntryDef {
  if (a.stage !== b.stage) {
    throw new Error(`composeStages: stages must match (got ${a.stage} + ${b.stage})`);
  }

  // A carrier is created for every `a` output that `b` *actually reads*
  // via a `ReadInput("Input", name)` somewhere in its body. We scan the
  // body rather than trusting `b.inputs`, because effect composition
  // sometimes leaves declared-but-unused inputs behind. Builtins are
  // never piped — they're stage outputs the runtime consumes.
  const bInputReads = new Set<string>();
  for (const sn of readInputs(b.body).values()) {
    if (sn.scope === "Input") bInputReads.add(sn.name);
  }
  const carriers = new Map<string, Var>();
  for (const out of a.outputs) {
    const isBuiltin = out.decorations.some((d) => d.kind === "Builtin");
    if (isBuiltin) continue;
    if (!bInputReads.has(out.name)) continue;
    carriers.set(out.name, {
      name: `_pipe_${out.name}`,
      type: out.type,
      mutable: true,
    });
  }
  const piped = new Set(carriers.keys());

  // 1. Rewrite `a`:
  //    - WriteOutput(piped name) → Write(carrier) (and keep, since the
  //      output is also still surfaced if not in `piped`).
  //    Other WriteOutputs stay — including ones B doesn't read; the
  //    fragment-output linker (or pruneCrossStage for vertex-side)
  //    handles dead-output elimination later.
  const aRewritten = redirectPipedWrites(a.body, carriers);

  // 2. Rewrite `b`: ReadInput("Input", piped name) → Var(carrier).
  const bRewritten = renameInputsToVars(b.body, carriers);

  // 3. Carrier declarations come first.
  const declarations: Stmt[] = [...carriers.values()].map((cv) => ({
    kind: "Declare", var: cv,
  }));

  // 4. Outputs: keep ALL of `a`'s non-piped outputs; then merge with
  //    `b`'s outputs. Iterate B FIRST so name collisions resolve as
  //    B-wins (last-wins ordering — B is downstream).
  const aOutputs = a.outputs.filter((p) => !piped.has(p.name));
  const mergedOutputs = mergeParams(b.outputs, aOutputs);
  // Inputs: `a`'s inputs always; `b`'s inputs minus piped names.
  const bInputs = b.inputs.filter((p) => !piped.has(p.name));
  const mergedInputs = mergeParams(a.inputs, bInputs);

  const fusedBody: Stmt = dceStmt({
    kind: "Sequential",
    body: [...declarations, aRewritten, bRewritten],
  });

  return {
    ...a,
    name: `${a.name}_${b.name}`,
    inputs: mergedInputs,
    outputs: mergedOutputs,
    body: fusedBody,
  };
}

function mergeParams(
  xs: readonly EntryParameter[],
  ys: readonly EntryParameter[],
): readonly EntryParameter[] {
  const seen = new Set<string>();
  const out: EntryParameter[] = [];
  for (const p of [...xs, ...ys]) {
    if (seen.has(p.name)) continue;
    seen.add(p.name);
    out.push(p);
  }
  return out;
}

function redirectPipedWrites(s: Stmt, carriers: Map<string, Var>): Stmt {
  const transform = (child: Stmt): Stmt => {
    if (child.kind === "WriteOutput") {
      const cv = carriers.get(child.name);
      if (cv) {
        return {
          kind: "Write",
          target: { kind: "LVar", var: cv, type: cv.type },
          value: rExprToExpr(child.value),
        };
      }
    }
    if (child.kind === "Write" && child.target.kind === "LInput"
      && child.target.scope === "Output") {
      const cv = carriers.get(child.target.name);
      if (cv) {
        return {
          kind: "Write",
          target: { kind: "LVar", var: cv, type: cv.type },
          value: child.value,
        };
      }
    }
    return child;
  };
  return transform(mapStmt(s, { stmt: transform }));
}

function rExprToExpr(r: { kind: "Expr"; value: Expr } | { kind: "ArrayLiteral"; arrayType: import("../ir/index.js").Type; values: readonly Expr[] }): Expr {
  if (r.kind === "Expr") return r.value;
  // ArrayLiteral RExprs only appear in array writes — array outputs are
  // unusual in pipelined I/O; we don't pipe them through carriers.
  // Fall back to first element to avoid a synthesis error; pruneCrossStage
  // will likely drop the carrier anyway.
  return r.values[0] ?? ({ kind: "Const", value: { kind: "Null" }, type: r.arrayType } as Expr);
}

function renameInputsToVars(s: Stmt, carriers: Map<string, Var>): Stmt {
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "ReadInput" && e.scope === "Input") {
      const cv = carriers.get(e.name);
      if (cv) return { kind: "Var", var: cv, type: cv.type };
    }
    return e;
  };
  const lexprFn = (l: LExpr): LExpr => {
    if (l.kind === "LInput" && l.scope === "Input") {
      const cv = carriers.get(l.name);
      if (cv) return { kind: "LVar", var: cv, type: cv.type };
    }
    return l;
  };
  return mapStmt(s, { expr: exprFn, lexpr: lexprFn });
}
