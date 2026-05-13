// Symbolic output-set analysis on a shader-IR statement tree.
//
// Walks a statement, collecting every distinct `Expr` that could be
// returned via `ReturnValue(...)`. Branches (`If`/`Switch`/loops) are
// over-approximated: both branches contribute. Local `Declare`s with
// initialisers are pre-substituted into subsequent reads (a single
// pass; this isn't full SSA — repeated assignments to the same Var
// are treated as TOP, i.e. the Var stays unresolved). Closure
// captures (`ReadInput("Uniform", …)`, `Var(...)` with no in-scope
// declaration) stay symbolic in the returned Exprs.
//
// Consumers (e.g. `@aardworx/wombat.rendering`'s `derivedMode`) use
// the set to size pipeline-slot tables: distinct *evaluated* values
// = slot count. A single-element set whose only member is e.g.
// `ReadInput("Uniform", "declared")` means "1 slot per declared
// value at any moment" → degenerates to 1 slot at runtime.
//
// What we deliberately DON'T do:
//   - Constant-fold arithmetic / comparisons / intrinsics (consumer
//     does this against a specific declared-value substitution).
//   - Track assignments through loops / re-assignments.
//   - Resolve user-defined function calls (only the top-level body
//     of one function is inspected).

import type { Expr, Stmt, Var } from "../ir/types.js";
import { stableStringify } from "../ir/hash.js";

/**
 * The conservative set of expressions a `Stmt` may return.
 *
 * Elements are deduplicated by `stableStringify`-content equality
 * (two structurally identical Exprs collapse to one entry).
 *
 * The empty set means "no `ReturnValue` is reachable" — e.g. the
 * body is a pure side-effect chain (statements only, no return).
 */
export interface SymbolicOutputSet {
  readonly exprs: ReadonlyArray<Expr>;
}

/** Walk `body` and collect every expression that flows into a
 *  `ReturnValue(...)`. */
export function analyseOutputSet(body: Stmt): SymbolicOutputSet {
  const env = new Map<string, Expr>(); // Var.name → its current init expression
  const seen = new Set<string>();      // dedupe by content
  const exprs: Expr[] = [];

  const visit = (s: Stmt): void => {
    switch (s.kind) {
      case "ReturnValue": {
        const expanded = substituteVars(s.value, env);
        // Flatten outermost Conditional into branch alternatives —
        // `return cond ? a : b` contributes BOTH `a` and `b` to the
        // set as distinct possible outputs. Without this flattening
        // the set would be `{ Conditional(cond, a, b) }` (size 1)
        // and the consumer's per-slot distinct-value count would
        // miss the branching.
        for (const e of unfoldConditional(expanded)) {
          const key = stableStringify(e);
          if (!seen.has(key)) {
            seen.add(key);
            exprs.push(e);
          }
        }
        return;
      }
      case "Declare": {
        if (s.init !== undefined) {
          // `s.init` is an RExpr — typically an Expr, but the union
          // also includes `ArrayLiteral`. We can only substitute /
          // track Exprs; ArrayLiterals leave the env untouched (the
          // Var stays unresolved and its uses surface as symbolic
          // Var refs in the analysed set).
          const init = s.init as { kind: string };
          if (init.kind !== "ArrayLiteral") {
            env.set(s.var.name, substituteVars(s.init as unknown as Expr, env));
          }
        }
        return;
      }
      case "Write": {
        // Track writes to a simple Var lvalue as re-assignment. For
        // more complex lvalues (field/index writes) we give up and
        // forget the previous binding.
        const t = s.target as { kind: string; var?: Var };
        if (t.kind === "Var" && t.var !== undefined) {
          env.set(t.var.name, substituteVars(s.value, env));
        } else {
          // Conservative: nothing.
        }
        return;
      }
      case "Sequential":
      case "Isolated":
        for (const c of s.body) visit(c);
        return;
      case "If": {
        // Branches contribute independently; their env effects
        // shouldn't leak into one another (this isn't full SSA).
        const beforeEnv = new Map(env);
        visit(s.then);
        // Restore env, then walk the else branch (if any).
        env.clear();
        for (const [k, v] of beforeEnv) env.set(k, v);
        if (s.else !== undefined) visit(s.else);
        // After the if, the env is left at whatever the else branch
        // wrote — same conservative posture as the original. For
        // analysis purposes (we only care about reachable returns)
        // this is fine.
        return;
      }
      case "Switch": {
        const beforeEnv = new Map(env);
        for (const c of s.cases) {
          env.clear();
          for (const [k, v] of beforeEnv) env.set(k, v);
          visit(c.body);
        }
        if (s.default !== undefined) {
          env.clear();
          for (const [k, v] of beforeEnv) env.set(k, v);
          visit(s.default);
        }
        return;
      }
      case "For":
      case "While":
      case "DoWhile":
      case "Loop":
        // Walk the body once; for analysis we assume the loop body
        // either runs or doesn't, contributing whatever it returns.
        // We don't propagate writes through loops (env effectively
        // becomes TOP for the body's writes — captured as
        // unresolved Vars in returned exprs).
        visit(s.body);
        return;
      case "Return":
      case "Nop":
      case "Break":
      case "Continue":
      case "Expression":
      case "Increment":
      case "Decrement":
      case "WriteOutput":
      case "Discard":
      case "Barrier":
        return;
    }
  };
  visit(body);
  return { exprs };
}

/**
 * Recursively substitute every `Var` reference in `e` with the
 * Expr from `env` (when bound). Unbound Vars and `ReadInput`s stay
 * symbolic.
 *
 * Pure / pre-order. Returns the same `e` reference (===) when no
 * substitution applies to the subtree, so consumers can cheaply
 * check for structural equality against the input.
 */
export function substituteVars(e: Expr, env: ReadonlyMap<string, Expr>): Expr {
  if (env.size === 0) return e;
  const obj = e as unknown as Record<string, unknown>;
  if (obj.kind === "Var") {
    const v = obj.var as Var;
    const bound = env.get(v.name);
    return bound !== undefined ? bound : e;
  }
  // Generic recurse: walk every property that's either an Expr or
  // an Expr array. Don't recurse into types/literals/decorations.
  let changed = false;
  const out: Record<string, unknown> = { ...obj };
  for (const k of Object.keys(out)) {
    const v = out[k];
    if (v !== null && typeof v === "object") {
      if ("kind" in (v as object)) {
        // Single Expr child.
        const child = substituteVars(v as Expr, env);
        if (child !== v) { out[k] = child; changed = true; }
      } else if (Array.isArray(v)) {
        let arrChanged = false;
        const mapped = (v as unknown[]).map((c) => {
          if (c !== null && typeof c === "object" && "kind" in (c as object)) {
            const ce = substituteVars(c as Expr, env);
            if (ce !== c) arrChanged = true;
            return ce;
          }
          return c;
        });
        if (arrChanged) { out[k] = mapped; changed = true; }
      }
    }
  }
  return changed ? (out as unknown as Expr) : e;
}

/**
 * Unfold an outermost `Conditional` (and nested `Conditional`s on
 * its branches) into a flat list of alternative result expressions.
 *
 * Why: `return cond ? a : b` lowers to `ReturnValue(Conditional(cond,
 * a, b))`. From an output-set perspective the two possible outputs
 * are `a` and `b` — the consumer evaluates each against a specific
 * `declared` substitution and counts distinct values. Leaving the
 * `Conditional` un-unfolded would give a set of size 1 and miss
 * the branch fan-out.
 *
 * Only the *outermost* expression and its `Conditional` chains are
 * unfolded — sub-expressions of arithmetic ops etc. stay intact.
 */
export function unfoldConditional(e: Expr): ReadonlyArray<Expr> {
  if ((e as { kind: string }).kind !== "Conditional") return [e];
  const c = e as { ifTrue: Expr; ifFalse: Expr };
  return [...unfoldConditional(c.ifTrue), ...unfoldConditional(c.ifFalse)];
}
