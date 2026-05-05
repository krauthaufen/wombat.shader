// linkHelpers — backward-liveness DCE across helper-call boundaries.
//
// `composeStages` produces a fused entry as a wrapper that calls a
// sequence of `merged_state_helper` Functions threading a shared
// State struct. The standard `dce` pass only sees inside one
// function at a time, so a helper's WriteOutput targeting a State
// field that nothing downstream reads stays alive — and the
// expression on its RHS keeps every uniform / sampler / attribute
// it touches alive too. That's how `reduceUniforms` ends up
// retaining a `u_dead` whose only use is in a helper's now-dead
// write.
//
// This pass runs liveness analysis ACROSS helper calls. Algorithm
// (iterates to fixed point per wrapper):
//
//   1. seed `liveFields` from the wrapper body — every State field
//      whose value is consumed outside the helper-call chain
//      (typically: `WriteOutput(X, … s.Y …)` makes Y live).
//   2. for each helper in REVERSE call order:
//        keep only WriteOutputs to live fields;
//        scan the surviving writes' RHS for `ReadInput("Input", Z)`
//          and add Z to `liveFields` (the helper consumed state.Z);
//   3. drop dead state-init writes (`s.X = in.X` in the wrapper)
//      whose X isn't in liveFields.
//   4. re-run standard DCE on each helper body to collapse newly
//      dead local lets, and on the wrapper to collapse dead inits.
//   5. shrink the State struct TypeDef and helper signatures to
//      contain only liveFields.
//   6. iterate until liveFields stops growing AND no more writes
//      are dropped.
//
// After this pass, `reduceUniforms` sees only the uniforms still
// reachable through some live path.

import type {
  EntryDef,
  Expr,
  Module,
  Stmt,
  StructField,
  Type,
  TypeDef,
  ValueDef,
  Var,
} from "../ir/index.js";
import { mapStmt, mapExpr, mapStmtChildren } from "./transform.js";
import { dceStmt } from "./dce.js";
import { freeVarsExpr } from "./analysis.js";

export function linkHelpers(module: Module): Module {
  // Build helper lookup.
  const helpers = new Map<string, ValueDef & { kind: "Function" }>();
  for (const v of module.values) {
    if (v.kind === "Function" && (v.attributes ?? []).includes("merged_state_helper")) {
      helpers.set(v.signature.name, v);
    }
  }
  if (helpers.size === 0) return module;

  let values = [...module.values];
  let types = [...module.types];
  let changed = true;
  let iterations = 0;
  const MAX_ITERATIONS = 16;

  while (changed && iterations++ < MAX_ITERATIONS) {
    changed = false;

    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      if (v.kind !== "Entry") continue;
      const wrapper = v.entry;
      const callSeq = collectHelperCalls(wrapper.body, helpers);
      if (callSeq.length === 0) continue;

      const stateVar = findStateVar(wrapper.body, callSeq);
      if (!stateVar) continue;
      const stateTypeName =
        stateVar.type.kind === "Struct" ? stateVar.type.name : undefined;
      if (!stateTypeName) continue;

      // 1. Seed liveFields from wrapper body, EXCLUDING reads in the
      //    helper-call args (`Call(helper, [Var(s)])` "uses" s
      //    structurally but doesn't pin any specific field — that
      //    role is delegated to the helper's body reads).
      const liveFields = collectWrapperLiveFields(wrapper.body, stateVar);

      // 2. Backward propagate through helpers (reverse call order).
      for (let j = callSeq.length - 1; j >= 0; j--) {
        const helperName = callSeq[j]!;
        const fn = helpers.get(helperName);
        if (!fn) continue;
        propagateHelperReads(fn.body, liveFields);
      }

      // 3. Now we know the full liveFields set. Build a new wrapper
      //    body that drops dead state-init writes and dead surfacing.
      const newWrapperBody = pruneWrapperBody(wrapper.body, stateVar, liveFields);

      // 4. For each helper called, drop WriteOutputs to dead fields,
      //    then re-DCE its body.
      for (const helperName of callSeq) {
        const idx = values.findIndex((x) =>
          x.kind === "Function" && x.signature.name === helperName,
        );
        if (idx < 0) continue;
        const fn = values[idx]! as ValueDef & { kind: "Function" };
        const newBody = dceStmt(pruneHelperWrites(fn.body, liveFields));
        if (newBody !== fn.body) {
          const updated: ValueDef = { ...fn, body: newBody };
          values[idx] = updated;
          helpers.set(helperName, updated as ValueDef & { kind: "Function" });
          changed = true;
        }
      }

      if (newWrapperBody !== wrapper.body) {
        values[i] = {
          kind: "Entry",
          entry: { ...wrapper, body: dceStmt(newWrapperBody) },
        };
        changed = true;
      }

      // 5. Shrink the State struct + helper signatures.
      const stateIdx = types.findIndex((t) => t.kind === "Struct" && t.name === stateTypeName);
      if (stateIdx >= 0) {
        const t = types[stateIdx]!;
        if (t.kind === "Struct") {
          const keptFields = t.fields.filter((f) => liveFields.has(f.name));
          if (keptFields.length !== t.fields.length) {
            const newType: TypeDef = { kind: "Struct", name: t.name, fields: keptFields };
            types[stateIdx] = newType;
            // Rewrite every Type that points to this struct to carry
            // the new fields list (so deep equality still matches).
            const newStateType: Type = {
              kind: "Struct", name: t.name, fields: keptFields,
            };
            values = values.map((vv) => rewriteStateType(vv, t.name, newStateType));
            changed = true;
          }
        }
      }
    }
  }

  return { ...module, types, values };
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

/**
 * Walk a wrapper body and collect the names of helper calls in the
 * order they execute. Each helper call is a `Write(LVar(state),
 * Call(helperRef, [Var(state)]))` Stmt.
 */
function collectHelperCalls(
  body: Stmt,
  helpers: Map<string, unknown>,
): string[] {
  const calls: string[] = [];
  const walk = (s: Stmt): void => {
    if (s.kind === "Write" && s.value.kind === "Call") {
      const name = s.value.fn.signature.name;
      if (helpers.has(name)) calls.push(name);
    }
    if (s.kind === "Sequential" || s.kind === "Isolated") {
      for (const c of s.body) walk(c);
    }
  };
  walk(body);
  return calls;
}

/** Find the State Var declared at top level of a wrapper body. */
function findStateVar(body: Stmt, callSeq: readonly string[]): Var | undefined {
  if (callSeq.length === 0) return undefined;
  const seen = new Set<string>();
  let result: Var | undefined;
  const walk = (s: Stmt): void => {
    if (result) return;
    if (s.kind === "Declare" && s.var.type.kind === "Struct" && !seen.has(s.var.name)) {
      seen.add(s.var.name);
      result = s.var;
      return;
    }
    if (s.kind === "Sequential" || s.kind === "Isolated") {
      for (const c of s.body) walk(c);
    }
  };
  walk(body);
  return result;
}

/**
 * Collect every State field READ in the wrapper body, excluding the
 * `Var(state)` argument passed to helper Calls (which "uses" the
 * struct as a whole — the per-field liveness is delegated to the
 * helper body reads via `propagateHelperReads`). Reads include
 * `Field(Var(state), name)` in any expression position AND
 * `LField(LVar(state), name)` reads on the lhs of a Write whose
 * RHS-of-this-write somehow depends on it (rare; we just track all
 * Field reads for safety).
 */
function collectWrapperLiveFields(body: Stmt, stateVar: Var): Set<string> {
  const live = new Set<string>();
  const visitExpr = (e: Expr): void => {
    // Match by Var NAME, not identity — the State-shrinking rewrite
    // re-emits Vars per-occurrence, so JS-reference equality fails
    // across iterations of the fixed-point loop. The wrapper's
    // generated State Var has a unique name within the wrapper, so
    // name-equality is safe.
    if (e.kind === "Field"
        && e.target.kind === "Var"
        && e.target.var.name === stateVar.name) {
      live.add(e.name);
    }
    // Recurse via mapExpr-style — collect all Field-reads.
    walkExprChildren(e, visitExpr);
  };
  const visitStmt = (s: Stmt): void => {
    switch (s.kind) {
      case "WriteOutput":
        if (s.value.kind === "Expr") visitExpr(s.value.value);
        else for (const v of s.value.values) visitExpr(v);
        if (s.index) visitExpr(s.index);
        return;
      case "Write":
        if (s.value.kind === "Call") {
          // Helper call — Var(state) is the arg; don't seed liveness
          // here. The helper's own body reads do that.
          // BUT do walk OTHER args (none for our helpers).
          for (const a of s.value.args) {
            if (!(a.kind === "Var" && a.var.name === stateVar.name)) visitExpr(a);
          }
          return;
        }
        visitExpr(s.value);
        return;
      case "Expression":
        visitExpr(s.value);
        return;
      case "If":
        visitExpr(s.cond);
        visitStmt(s.then);
        if (s.else) visitStmt(s.else);
        return;
      case "Sequential":
      case "Isolated":
        for (const c of s.body) visitStmt(c);
        return;
      case "For":
        visitStmt(s.init);
        visitExpr(s.cond);
        visitStmt(s.step);
        visitStmt(s.body);
        return;
      case "While":
      case "DoWhile":
        visitExpr(s.cond);
        visitStmt(s.body);
        return;
      case "Loop":
        visitStmt(s.body);
        return;
      case "Switch":
        visitExpr(s.value);
        for (const c of s.cases) visitStmt(c.body);
        if (s.default) visitStmt(s.default);
        return;
      case "Declare":
        if (s.init) {
          if (s.init.kind === "Expr") visitExpr(s.init.value);
          else for (const v of s.init.values) visitExpr(v);
        }
        return;
      case "ReturnValue":
        visitExpr(s.value);
        return;
      default:
        return;
    }
  };
  visitStmt(body);
  return live;
}

/**
 * For a helper body, mark every `ReadInput("Input", X)` from a
 * surviving live write as live. Survivors = WriteOutputs whose
 * `name` is in `live`. The recursive walk into RHS expressions
 * picks up nested reads (e.g., `out.X = f(g(in.Y))` makes Y live).
 */
function propagateHelperReads(body: Stmt, live: Set<string>): void {
  const visitExpr = (e: Expr): void => {
    if (e.kind === "ReadInput" && e.scope === "Input") {
      live.add(e.name);
    }
    walkExprChildren(e, visitExpr);
  };
  const visitStmt = (s: Stmt): void => {
    switch (s.kind) {
      case "WriteOutput":
        if (live.has(s.name)) {
          if (s.value.kind === "Expr") visitExpr(s.value.value);
          else for (const v of s.value.values) visitExpr(v);
          if (s.index) visitExpr(s.index);
        }
        return;
      case "Write": {
        // If lhs is LField(LVar(_), name), name might be a State
        // field write — but in a `merged_state_helper` body the
        // emitter targets `out.name` only via WriteOutput(name, …),
        // so an explicit LField write on a state-typed LVar would
        // be unusual. Fall through and walk RHS regardless so we
        // don't drop reads we can't classify.
        const lname = lFieldName(s.target);
        if (lname === undefined || live.has(lname)) {
          visitExpr(s.value);
        }
        return;
      }
      case "If":
        visitExpr(s.cond);
        visitStmt(s.then);
        if (s.else) visitStmt(s.else);
        return;
      case "Sequential":
      case "Isolated":
        for (const c of s.body) visitStmt(c);
        return;
      case "For":
        visitStmt(s.init);
        visitExpr(s.cond);
        visitStmt(s.step);
        visitStmt(s.body);
        return;
      case "While":
      case "DoWhile":
        visitExpr(s.cond);
        visitStmt(s.body);
        return;
      case "Loop":
        visitStmt(s.body);
        return;
      case "Switch":
        visitExpr(s.value);
        for (const c of s.cases) visitStmt(c.body);
        if (s.default) visitStmt(s.default);
        return;
      case "Expression":
        visitExpr(s.value);
        return;
      case "Declare":
        if (s.init) {
          if (s.init.kind === "Expr") visitExpr(s.init.value);
          else for (const v of s.init.values) visitExpr(v);
        }
        return;
      case "ReturnValue":
        visitExpr(s.value);
        return;
      default:
        return;
    }
  };
  visitStmt(body);
}

function lFieldName(l: import("../ir/index.js").LExpr): string | undefined {
  if (l.kind === "LField") return l.name;
  return undefined;
}

/**
 * Drop wrapper Stmts that initialise dead State fields (`s.X = in.X`)
 * and dead surfacing WriteOutputs for fields whose entry-output
 * counterpart was removed by `linkFragmentOutputs`. We DON'T touch
 * helper-call Writes — those are kept regardless (their helper may
 * still mutate live fields).
 */
function pruneWrapperBody(
  body: Stmt,
  stateVar: Var,
  liveFields: Set<string>,
): Stmt {
  const transform = (s: Stmt): Stmt => {
    if (s.kind === "Write"
        && s.target.kind === "LField"
        && s.target.target.kind === "LVar"
        && s.target.target.var.name === stateVar.name
        && !liveFields.has(s.target.name)) {
      return { kind: "Nop" };
    }
    return s;
  };
  // mapStmt only applies the callback to *child* Stmts; we apply it
  // to the top-level result too so a body that's just a single
  // `Write` (no surrounding Sequential) gets pruned.
  return transform(mapStmt(body, { stmt: transform }));
}

/**
 * Drop helper WriteOutputs targeting fields not in the live set.
 * Keeps everything else; standard DCE then collapses any locals
 * that become dead as a result.
 */
function pruneHelperWrites(body: Stmt, live: Set<string>): Stmt {
  const transform = (s: Stmt): Stmt => {
    if (s.kind === "WriteOutput" && !live.has(s.name)) return { kind: "Nop" };
    return s;
  };
  return transform(mapStmt(body, { stmt: transform }));
}

/**
 * Update every reference to a Struct type with the given name to
 * carry the new fields list. Without this, downstream type
 * comparisons that include the fields list would diverge after
 * we shrink the State struct.
 */
function rewriteStateType(v: ValueDef, name: string, newType: Type): ValueDef {
  const rewriteT = (t: Type): Type => {
    if (t.kind === "Struct" && t.name === name) return newType;
    return t;
  };
  if (v.kind === "Function") {
    const sig = v.signature;
    const newParams = sig.parameters.map((p) => ({ ...p, type: rewriteT(p.type) }));
    const newRet = rewriteT(sig.returnType);
    if (newRet === sig.returnType
        && newParams.every((p, i) => p.type === sig.parameters[i]!.type)) {
      // Body Vars might still carry old type; rewrite Vars in body.
      const newBody = rewriteVarTypesInStmt(v.body, name, newType);
      return { ...v, body: newBody };
    }
    const newBody = rewriteVarTypesInStmt(v.body, name, newType);
    return {
      ...v,
      signature: { ...sig, returnType: newRet, parameters: newParams },
      body: newBody,
    };
  }
  if (v.kind === "Entry") {
    const newBody = rewriteVarTypesInStmt(v.entry.body, name, newType);
    return { ...v, entry: { ...v.entry, body: newBody } };
  }
  return v;
}

function rewriteVarTypesInStmt(s: Stmt, name: string, newType: Type): Stmt {
  const rewriteT = (t: Type): Type => (t.kind === "Struct" && t.name === name ? newType : t);
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "Var" && e.var.type.kind === "Struct" && e.var.type.name === name) {
      return { ...e, var: { ...e.var, type: newType }, type: newType };
    }
    if (e.kind === "Field" && e.type.kind === "Struct" && e.type.name === name) {
      return { ...e, type: newType };
    }
    if (e.kind === "Call" && e.type.kind === "Struct" && e.type.name === name) {
      return { ...e, type: newType };
    }
    return e;
  };
  return mapStmt(s, {
    stmt: (st) => {
      if (st.kind === "Declare" && st.var.type.kind === "Struct" && st.var.type.name === name) {
        return { ...st, var: { ...st.var, type: newType } };
      }
      return st;
    },
    expr: exprFn,
    lexpr: (l) => {
      if (l.kind === "LVar" && l.var.type.kind === "Struct" && l.var.type.name === name) {
        return { ...l, var: { ...l.var, type: newType }, type: newType };
      }
      return l;
    },
  });
}

/** Walk Expr children — minimal local copy to avoid pulling visit.ts. */
function walkExprChildren(e: Expr, fn: (c: Expr) => void): void {
  switch (e.kind) {
    case "Var":
    case "Const":
      return;
    case "ReadInput":
      if (e.index) fn(e.index);
      return;
    case "Call":
    case "CallIntrinsic":
      for (const a of e.args) fn(a);
      return;
    case "Conditional":
      fn(e.cond); fn(e.ifTrue); fn(e.ifFalse);
      return;
    case "Neg":
    case "Not":
    case "BitNot":
    case "Transpose":
    case "Inverse":
    case "Determinant":
    case "Length":
    case "VecAny":
    case "VecAll":
    case "ConvertMatrix":
    case "Convert":
      fn(e.value);
      return;
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Mod":
    case "MulMatMat":
    case "MulMatVec":
    case "MulVecMat":
    case "Dot":
    case "Cross":
    case "And":
    case "Or":
    case "BitAnd":
    case "BitOr":
    case "BitXor":
    case "ShiftLeft":
    case "ShiftRight":
    case "Eq":
    case "Neq":
    case "Lt":
    case "Le":
    case "Gt":
    case "Ge":
    case "VecLt":
    case "VecLe":
    case "VecGt":
    case "VecGe":
    case "VecEq":
    case "VecNeq":
      fn(e.lhs); fn(e.rhs);
      return;
    case "VecSwizzle":
    case "VecItem":
      fn(e.value);
      if (e.kind === "VecItem") fn(e.index);
      return;
    case "MatrixElement":
      fn(e.matrix); fn(e.row); fn(e.col);
      return;
    case "MatrixRow":
      fn(e.matrix); fn(e.row);
      return;
    case "MatrixCol":
      fn(e.matrix); fn(e.col);
      return;
    case "NewVector":
      for (const c of e.components) fn(c);
      return;
    case "NewMatrix":
      for (const c of e.elements) fn(c);
      return;
    case "MatrixFromRows":
      for (const r of e.rows) fn(r);
      return;
    case "MatrixFromCols":
      for (const c of e.cols) fn(c);
      return;
    case "Field":
      fn(e.target);
      return;
    case "Item":
      fn(e.target); fn(e.index);
      return;
    case "DebugPrintf":
      fn(e.format);
      for (const a of e.args) fn(a);
      return;
  }
}
