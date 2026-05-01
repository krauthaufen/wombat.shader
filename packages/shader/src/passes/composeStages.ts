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
    // Reduce by sequential fusion. We pass `futureReads`: the union
    // of `Input`-scope reads across all stages strictly downstream of
    // the current pair. Without it, intermediate fusions would drop
    // outputs that a later stage in the chain still needs (e.g.
    // `pickDepthBefore + userEff + pickFinalA`: pickDepthBefore's
    // `Depth` output is read by pickFinalA but not by userEff, so
    // pairwise fusion of A+B was dropping it before C could see it).
    let merged = entries[0]!;
    for (let i = 1; i < entries.length; i++) {
      const futureReads = new Set<string>();
      for (let j = i + 1; j < entries.length; j++) {
        for (const sn of readInputs(entries[j]!.body).values()) {
          if (sn.scope === "Input") futureReads.add(sn.name);
        }
      }
      merged = fusePair(merged, entries[i]!, futureReads);
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
function fusePair(a: EntryDef, b: EntryDef, futureReads: ReadonlySet<string> = new Set()): EntryDef {
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
  // Carriers cover both immediate (B reads) and future (later-stage
  // reads) consumption — when a downstream-but-not-immediate stage
  // wants A's output we still pipe through a local var so the
  // downstream stage's ReadInput maps cleanly without bleeding through
  // the merged entry's output list (which would clash on Location with
  // B's own outputs).
  const carriers = new Map<string, Var>();
  for (const out of a.outputs) {
    const isBuiltin = out.decorations.some((d) => d.kind === "Builtin");
    if (isBuiltin) continue;
    if (!bInputReads.has(out.name) && !futureReads.has(out.name)) continue;
    carriers.set(out.name, {
      name: `_pipe_${out.name}`,
      type: out.type,
      mutable: true,
    });
  }
  const piped = new Set(carriers.keys());

  // 1. Rewrite `a`:
  //    - WriteOutput(piped name) → Write(carrier).
  //    - WriteOutput(name not consumed by b, not builtin) → Nop. In
  //      sequential same-stage composition, `a` is producing inputs
  //      for `b`; outputs `b` doesn't read are dead by construction.
  //    - Other WriteOutputs (builtins) stay.
  const aBuiltinNames = new Set(
    a.outputs
      .filter((p) => p.decorations.some((d) => d.kind === "Builtin"))
      .map((p) => p.name),
  );
  // For redirect/output-keep decisions an A output is "needed downstream"
  // if either B reads it OR any later stage in the chain reads it.
  const consumed = new Set<string>([...bInputReads, ...futureReads]);
  const aRewritten = redirectPipedWrites(a.body, carriers, consumed, aBuiltinNames);

  // 2. Rewrite `b`: ReadInput("Input", piped name) → Var(carrier).
  const bRewritten = renameInputsToVars(b.body, carriers);

  // 3. Carrier declarations come first.
  const declarations: Stmt[] = [...carriers.values()].map((cv) => ({
    kind: "Declare", var: cv,
  }));

  // 4. Outputs: `a`'s outputs minus piped names AND minus dead-to-`b`
  //    non-builtin names; then `b`'s outputs.
  const aOutputs = a.outputs.filter((p) => {
    if (piped.has(p.name)) return false;
    const isBuiltin = aBuiltinNames.has(p.name);
    // Keep builtins, kept-by-B's-direct-read, AND anything downstream in
    // the chain reads — outputs only get dropped when nothing reads them.
    if (!isBuiltin && !bInputReads.has(p.name) && !futureReads.has(p.name)) return false;
    return true;
  });
  const mergedOutputs = mergeParams(aOutputs, b.outputs);
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

function redirectPipedWrites(
  s: Stmt,
  carriers: Map<string, Var>,
  bConsumes: ReadonlySet<string>,
  aBuiltinNames: ReadonlySet<string>,
): Stmt {
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
      // Non-piped, non-builtin write to an output that `b` doesn't read
      // is dead in a same-stage chain: the runtime will only see `b`'s
      // outputs. Drop it.
      if (!aBuiltinNames.has(child.name) && !bConsumes.has(child.name)) {
        return { kind: "Nop" };
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
      if (!aBuiltinNames.has(child.target.name) && !bConsumes.has(child.target.name)) {
        return { kind: "Nop" };
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
