// effectDependencies — per-output transitive input analysis.
//
// For each output any stage of the effect writes, compute the set of
// **Input-scope reads** (i.e. vertex/instance attributes if vertex-
// stage, cross-stage inputs if fragment-stage) it depends on.
//
// Cross-stage resolve: a fragment input matching a vertex output is
// replaced by that vertex output's deps. Inputs without an upstream
// vertex producer remain as "true required inputs" — the caller
// (e.g. wombat.dom's `chooseChain`) treats those as geometry
// requirements.
//
// Used by callers that need to ask "can this effect produce X
// given that the geometry only exposes attributes Y?". Aardvark's
// `Effect.Dependencies` / `EffectDeps.resolveTop` does the same job
// for FShade.
//
// Algorithm (forward dataflow, over-approximating control flow):
//   1. Walk the entry body in order.
//   2. For each `Declare(var, init)` and `Write(LVar(var), value)`,
//      record the union of Input-scope reads transitively reachable
//      from the value expression (resolving `Var(v)` lookups via
//      previously-computed `varDeps`).
//   3. For each `WriteOutput(name, value)`, accumulate the same
//      computation into `outputDeps[name]`.
//
// Per-Var data-flow gives us reasonable precision for straight-line
// shader code. Branchy bodies (if/while/for) just union — same
// over-approximation FShade applies. We never lose soundness.

import type {
  EntryDef, Expr, LExpr, RExpr, Stmt, Type,
} from "../ir/index.js";
import type { Effect } from "../runtime/stage.js";
import { liftReturns } from "./liftReturns.js";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface OutputDep {
  /**
   * Inputs required to produce this output. Keyed by name; value is
   * the IR type. After cross-stage resolution, the names are
   * vertex/instance attribute names if the producer chain reaches
   * the vertex stage; otherwise they are "unresolved" (the producer
   * chain reads from an upstream stage that no effect in the chain
   * provides — the caller usually treats those as geometry
   * attributes too).
   */
  readonly inputs: ReadonlyMap<string, Type>;
}

/**
 * For every output any stage of `effect` writes, return the
 * cross-stage-resolved set of Input-scope reads it depends on.
 *
 * Outputs from compute stages are reported alongside graphics
 * outputs but compute stages don't participate in cross-stage
 * resolve — their input reads are reported as-is.
 */
export function effectDependencies(effect: Effect): ReadonlyMap<string, OutputDep> {
  // 1. Per-stage, per-entry: output → set<inputName, Type>.
  //    `Effect` carries one `Stage` per `vertex/fragment/compute`
  //    template. We index by stage kind to apply the v→f resolve
  //    rule below; multiple entries within the same stage are
  //    unioned.
  const perStage = {
    vertex: new Map<string, Map<string, Type>>(),
    fragment: new Map<string, Map<string, Type>>(),
    compute: new Map<string, Map<string, Type>>(),
  };

  for (const stage of effect.stages) {
    // The frontend leaves entry bodies as `ReturnValue(<object
    // literal>)` — opaque to the dep walker. `liftReturns` is the
    // structural pass that rewrites those into explicit
    // `WriteOutput` sequences. Run it here so callers don't have
    // to remember to pre-lift the IR.
    const lifted = liftReturns(stage.template);
    for (const v of lifted.values) {
      if (v.kind !== "Entry") continue;
      const rawDeps = entryOutputDeps(v.entry);
      const bucket = perStage[v.entry.stage];
      for (const [outName, deps] of rawDeps) {
        const merged = bucket.get(outName) ?? new Map<string, Type>();
        for (const [n, t] of deps) merged.set(n, t);
        bucket.set(outName, merged);
      }
    }
  }

  // 2. Cross-stage resolve: each fragment output's input set is
  //    rewritten by replacing entries that match a vertex output
  //    name with that vertex output's input set. Recurse to allow
  //    chained vertex stages (already fused by `composeStages`,
  //    but defensively support unfused vertex chains too).
  const resolved = new Map<string, OutputDep>();

  for (const [vertOut, vertDeps] of perStage.vertex) {
    resolved.set(vertOut, { inputs: freezeMap(vertDeps) });
  }

  // Resolve each fragment output's deps through the vertex layer.
  // Inputs that match a vertex output get replaced; others pass
  // through. The recursion is bounded by the number of distinct
  // vertex outputs, so a simple inline loop with a visited-set is
  // enough.
  for (const [fragOut, fragDeps] of perStage.fragment) {
    const out = new Map<string, Type>();
    const visited = new Set<string>();
    const work: [string, Type][] = [...fragDeps.entries()];
    while (work.length > 0) {
      const [n, t] = work.pop()!;
      if (visited.has(n)) continue;
      visited.add(n);
      const upstream = perStage.vertex.get(n);
      if (upstream === undefined) {
        // No vertex producer — this is either a geometry attribute
        // requirement that flows directly into the fragment, or an
        // input the user just declared as cross-stage. Either way
        // we report it.
        out.set(n, t);
      } else {
        for (const [un, ut] of upstream) work.push([un, ut]);
      }
    }
    resolved.set(fragOut, { inputs: freezeMap(out) });
  }

  // Compute outputs (no cross-stage resolve — compute stages stand
  // alone). Reported as-is.
  for (const [cOut, cDeps] of perStage.compute) {
    resolved.set(cOut, { inputs: freezeMap(cDeps) });
  }

  return resolved;
}

// ---------------------------------------------------------------------------
// Per-entry output dependencies
// ---------------------------------------------------------------------------

interface EntryDepsScratch {
  /** Per-Var: union of Input-scope (name, type) it transitively depends on. */
  readonly varDeps: Map<string, Map<string, Type>>;
  /** Per-output-name: same. Filled as `WriteOutput` statements are visited. */
  readonly outputDeps: Map<string, Map<string, Type>>;
}

function entryOutputDeps(entry: EntryDef): Map<string, Map<string, Type>> {
  const scratch: EntryDepsScratch = {
    varDeps: new Map(),
    outputDeps: new Map(),
  };
  walkStmt(entry.body, scratch);
  // Implicit (builtin-semantic) outputs that the body never
  // explicitly writes to don't show up — that's the right
  // behaviour: a chooseChain caller asking "can the user produce
  // ViewSpaceNormal" wants a no answer if the user never writes
  // it.
  return scratch.outputDeps;
}

// ---------------------------------------------------------------------------
// Walks
// ---------------------------------------------------------------------------

function walkStmt(s: Stmt, scratch: EntryDepsScratch): void {
  switch (s.kind) {
    case "Nop":
    case "Break":
    case "Continue":
    case "Discard":
    case "Barrier":
    case "Return":
      return;
    case "Expression":
      // Side-effecting expressions (intrinsic calls etc.) — we
      // don't track per-call deps but we do still need to count
      // any Input reads they perform (relevant for compute kernels
      // calling textureStore on a pickId etc., though uncommon).
      collectExprDeps(s.value, scratch.varDeps);
      return;
    case "Declare": {
      if (s.init === undefined) {
        scratch.varDeps.set(s.var.name, new Map());
        return;
      }
      const deps = collectRExprDeps(s.init, scratch.varDeps);
      scratch.varDeps.set(s.var.name, deps);
      return;
    }
    case "Write": {
      const valueDeps = collectExprDeps(s.value, scratch.varDeps);
      writeLExprDeps(s.target, valueDeps, scratch);
      return;
    }
    case "WriteOutput": {
      const valueDeps = collectRExprDeps(s.value, scratch.varDeps);
      const acc = scratch.outputDeps.get(s.name) ?? new Map<string, Type>();
      for (const [n, t] of valueDeps) acc.set(n, t);
      scratch.outputDeps.set(s.name, acc);
      // Optional index expression also contributes.
      if (s.index !== undefined) {
        const idx = collectExprDeps(s.index, scratch.varDeps);
        for (const [n, t] of idx) acc.set(n, t);
      }
      return;
    }
    case "Increment":
    case "Decrement":
      // Left-hand side is read+written; pulls in its own deps.
      writeLExprDeps(s.target, new Map(), scratch);
      return;
    case "Sequential":
    case "Isolated":
      for (const c of s.body) walkStmt(c, scratch);
      return;
    case "ReturnValue":
      collectExprDeps(s.value, scratch.varDeps);
      return;
    case "If": {
      const condDeps = collectExprDeps(s.cond, scratch.varDeps);
      // Conditional reads flow into any output written inside the
      // branches — over-approximate by appending cond deps to all
      // outputs touched by the branches.
      const before = snapshotOutputs(scratch.outputDeps);
      walkStmt(s.then, scratch);
      if (s.else !== undefined) walkStmt(s.else, scratch);
      const touched = newlyWritten(before, scratch.outputDeps);
      for (const name of touched) {
        const acc = scratch.outputDeps.get(name)!;
        for (const [n, t] of condDeps) acc.set(n, t);
      }
      return;
    }
    case "For": {
      walkStmt(s.init, scratch);
      collectExprDeps(s.cond, scratch.varDeps);
      walkStmt(s.step, scratch);
      walkStmt(s.body, scratch);
      return;
    }
    case "While":
    case "DoWhile":
      collectExprDeps(s.cond, scratch.varDeps);
      walkStmt(s.body, scratch);
      return;
    case "Switch": {
      collectExprDeps(s.value, scratch.varDeps);
      // SwitchCase carries `literal: Literal`, a constant — no
      // expression to walk.
      for (const c of s.cases) walkStmt(c.body, scratch);
      if (s.default !== undefined) walkStmt(s.default, scratch);
      return;
    }
  }
}

function writeLExprDeps(
  target: LExpr,
  valueDeps: Map<string, Type>,
  scratch: EntryDepsScratch,
): void {
  switch (target.kind) {
    case "LVar": {
      const acc = scratch.varDeps.get(target.var.name) ?? new Map<string, Type>();
      for (const [n, t] of valueDeps) acc.set(n, t);
      scratch.varDeps.set(target.var.name, acc);
      return;
    }
    case "LInput":
      if (target.scope === "Output") {
        const acc = scratch.outputDeps.get(target.name) ?? new Map<string, Type>();
        for (const [n, t] of valueDeps) acc.set(n, t);
        scratch.outputDeps.set(target.name, acc);
        return;
      }
      // Other scopes (Closure / Uniform / ...) aren't writable in
      // practice — ignore.
      return;
    case "LField":
      writeLExprDeps(target.target, valueDeps, scratch);
      return;
    case "LItem":
      writeLExprDeps(target.target, valueDeps, scratch);
      // Index expression contributes to the var's deps too.
      for (const [n, t] of collectExprDeps(target.index, scratch.varDeps)) valueDeps.set(n, t);
      return;
    case "LSwizzle":
      writeLExprDeps(target.target, valueDeps, scratch);
      return;
    case "LMatrixElement":
      writeLExprDeps(target.matrix, valueDeps, scratch);
      // Row + col indices contribute to the underlying var's deps.
      for (const [n, t] of collectExprDeps(target.row, scratch.varDeps)) valueDeps.set(n, t);
      for (const [n, t] of collectExprDeps(target.col, scratch.varDeps)) valueDeps.set(n, t);
      return;
  }
}

function collectRExprDeps(
  r: RExpr,
  varDeps: ReadonlyMap<string, Map<string, Type>>,
): Map<string, Type> {
  if (r.kind === "Expr") return collectExprDeps(r.value, varDeps);
  // Array literals — union of element deps.
  const out = new Map<string, Type>();
  for (const v of r.values) {
    for (const [n, t] of collectExprDeps(v, varDeps)) out.set(n, t);
  }
  return out;
}

function collectExprDeps(
  e: Expr,
  varDeps: ReadonlyMap<string, Map<string, Type>>,
): Map<string, Type> {
  const out = new Map<string, Type>();
  const visit = (x: Expr): void => {
    if (x.kind === "ReadInput") {
      if (x.scope === "Input") out.set(x.name, x.type);
      return;
    }
    if (x.kind === "Var") {
      const ds = varDeps.get(x.var.name);
      if (ds !== undefined) for (const [n, t] of ds) out.set(n, t);
      return;
    }
    visitExprChildren(x, visit);
  };
  visit(e);
  return out;
}

// Local visitor — mirrors `analysis.ts`'s `visitExprChildren` but
// kept private here to avoid an export churn in the analysis module.
function visitExprChildren(e: Expr, visit: (x: Expr) => void): void {
  switch (e.kind) {
    case "Var":
    case "Const":
    case "ReadInput":
    case "DebugPrintf":
      return;
    case "Call":
      for (const a of e.args) visit(a);
      return;
    case "CallIntrinsic":
      for (const a of e.args) visit(a);
      return;
    case "Conditional":
      visit(e.cond); visit(e.ifTrue); visit(e.ifFalse);
      return;
    case "Neg": case "Not": case "BitNot":
    case "Transpose": case "Inverse": case "Determinant":
    case "Length": case "VecAny": case "VecAll":
    case "Convert": case "ConvertMatrix":
      visit(e.value);
      return;
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "MulMatMat": case "MulMatVec": case "MulVecMat":
    case "Dot": case "Cross":
    case "And": case "Or":
    case "BitAnd": case "BitOr": case "BitXor":
    case "ShiftLeft": case "ShiftRight":
    case "Eq": case "Neq": case "Lt": case "Le": case "Gt": case "Ge":
    case "VecLt": case "VecLe": case "VecGt": case "VecGe":
    case "VecEq": case "VecNeq":
      visit(e.lhs); visit(e.rhs);
      return;
    case "VecSwizzle":
      visit(e.value);
      return;
    case "VecItem":
      visit(e.value); visit(e.index);
      return;
    case "MatrixElement":
      visit(e.matrix); visit(e.row); visit(e.col);
      return;
    case "MatrixRow":
      visit(e.matrix); visit(e.row);
      return;
    case "MatrixCol":
      visit(e.matrix); visit(e.col);
      return;
    case "NewVector":
      for (const a of e.components) visit(a);
      return;
    case "NewMatrix":
      for (const a of e.elements) visit(a);
      return;
    case "MatrixFromRows":
      for (const a of e.rows) visit(a);
      return;
    case "MatrixFromCols":
      for (const a of e.cols) visit(a);
      return;
    case "Field":
      visit(e.target);
      return;
    case "Item":
      visit(e.target); visit(e.index);
      return;
  }
}

function snapshotOutputs(m: Map<string, Map<string, Type>>): Map<string, number> {
  const out = new Map<string, number>();
  for (const [k, v] of m) out.set(k, v.size);
  return out;
}

function newlyWritten(
  before: Map<string, number>,
  now: Map<string, Map<string, Type>>,
): string[] {
  const out: string[] = [];
  for (const [k, v] of now) {
    const beforeSize = before.get(k);
    if (beforeSize === undefined || v.size !== beforeSize) out.push(k);
  }
  return out;
}

function freezeMap(m: Map<string, Type>): ReadonlyMap<string, Type> {
  return m;
}
