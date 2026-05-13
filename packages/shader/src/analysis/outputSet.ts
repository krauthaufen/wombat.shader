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

// ─────────────────────────────────────────────────────────────────────
// Concrete evaluation
// ─────────────────────────────────────────────────────────────────────

/**
 * Map from intrinsic name → numeric implementation, used to fold
 * named function calls in `evaluateConcrete`. Consumer-supplied so
 * each axis can declare its own helper semantics (e.g. cull has
 * `flipCull: 0→0, 1→2, 2→1`).
 *
 * Returning `undefined` from the implementation means "I can't fold
 * this here" — the caller treats the call as unresolved.
 */
export type IntrinsicEvalTable = ReadonlyMap<
  string,
  (args: ReadonlyArray<number>) => number | undefined
>;

/**
 * Concrete-value environment for `evaluateConcrete`. Each key is a
 * `ReadInput` name or a `Var.name`; the value is the numeric
 * substitution.
 */
export type EvalEnv = ReadonlyMap<string, number>;

/**
 * Try to evaluate `e` to a concrete number under the supplied env +
 * intrinsic table. Returns `undefined` (UNRESOLVED) when the
 * expression contains any leaf not bound in `env` or any node we
 * can't fold.
 *
 * Consumer pattern (mode-rule slot sizing):
 *
 *   const exprs = analyseOutputSet(body).exprs;
 *   const env = new Map<string, number>([["declared", currentDeclaredU32]]);
 *   const intrinsics = registry.forAxis("cull");
 *   const slots = new Set<number>();
 *   for (const e of exprs) {
 *     const v = evaluateConcrete(e, env, intrinsics);
 *     if (v !== undefined) slots.add(v);
 *   }
 *   // slots.size == slot count for this bucket+declared.
 */
export function evaluateConcrete(
  e: Expr,
  env: EvalEnv,
  intrinsics: IntrinsicEvalTable = new Map(),
): number | undefined {
  const k = (e as { kind: string }).kind;
  switch (k) {
    case "Const": {
      const v = (e as { value: { kind: string; value: number | boolean } }).value;
      if (v.kind === "Int" || v.kind === "Float") return v.value as number;
      if (v.kind === "Bool") return v.value ? 1 : 0;
      return undefined;
    }
    case "ReadInput": return env.get((e as { name: string }).name);
    case "Var":       return env.get((e as { var: { name: string } }).var.name);
    case "Neg": {
      const v = evaluateConcrete((e as { value: Expr }).value, env, intrinsics);
      return v === undefined ? undefined : -v;
    }
    case "Not": {
      const v = evaluateConcrete((e as { value: Expr }).value, env, intrinsics);
      return v === undefined ? undefined : (v ? 0 : 1);
    }
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "Lt": case "Le": case "Gt": case "Ge": case "Eq": case "Neq":
    case "And": case "Or": case "BitAnd": case "BitOr": case "BitXor":
    case "ShiftLeft": case "ShiftRight": {
      const bin = e as { lhs: Expr; rhs: Expr };
      const a = evaluateConcrete(bin.lhs, env, intrinsics);
      const b = evaluateConcrete(bin.rhs, env, intrinsics);
      if (a === undefined || b === undefined) return undefined;
      switch (k) {
        case "Add": return a + b;
        case "Sub": return a - b;
        case "Mul": return a * b;
        case "Div": return b === 0 ? undefined : a / b;
        case "Mod": return b === 0 ? undefined : a % b;
        case "Lt":  return a <  b ? 1 : 0;
        case "Le":  return a <= b ? 1 : 0;
        case "Gt":  return a >  b ? 1 : 0;
        case "Ge":  return a >= b ? 1 : 0;
        case "Eq":  return a === b ? 1 : 0;
        case "Neq": return a !== b ? 1 : 0;
        case "And": return (a && b) ? 1 : 0;
        case "Or":  return (a || b) ? 1 : 0;
        case "BitAnd": return (a & b) >>> 0;
        case "BitOr":  return (a | b) >>> 0;
        case "BitXor": return (a ^ b) >>> 0;
        case "ShiftLeft":  return ((a << (b & 31)) >>> 0);
        case "ShiftRight": return (a >>> (b & 31));
      }
      return undefined;
    }
    case "Conditional": {
      const cv = evaluateConcrete((e as { cond: Expr }).cond, env, intrinsics);
      if (cv === undefined) return undefined;
      return cv
        ? evaluateConcrete((e as { ifTrue: Expr }).ifTrue, env, intrinsics)
        : evaluateConcrete((e as { ifFalse: Expr }).ifFalse, env, intrinsics);
    }
    case "CallIntrinsic": {
      const ci = e as { op: { name: string }; args: ReadonlyArray<Expr> };
      // Shader-vite's `liftRuleObjectLiterals` encodes rule-body
      // object literals as `__record:<key1>|<key2>|...` intrinsics
      // — but `evaluateConcrete` returns numeric results only, so
      // a record evaluation has no useful number to produce. Return
      // `undefined`; the calling layer (wombat.rendering's heap
      // scene) interprets these structurally rather than numerically.
      if (ci.op.name.startsWith("__record:")) return undefined;
      const fn = intrinsics.get(ci.op.name);
      if (fn === undefined) return undefined;
      const args: number[] = [];
      for (const a of ci.args) {
        const v = evaluateConcrete(a, env, intrinsics);
        if (v === undefined) return undefined;
        args.push(v);
      }
      return fn(args);
    }
    case "Convert":
    case "ConvertMatrix":
      return evaluateConcrete((e as { value: Expr }).value, env, intrinsics);
    default:
      return undefined;
  }
}

/**
 * Convenience: evaluate every expression in a `SymbolicOutputSet`
 * and return the set of distinct concrete numbers we could fold to.
 * Expressions that don't fold are dropped (they remain symbolic in
 * the caller's bookkeeping — the slot count under-approximates by
 * one per unfoldable expression).
 *
 * Returns a deterministic-order array (sorted ascending) so two
 * calls with the same inputs yield the same slot ordering.
 */
export function evaluateSet(
  set: SymbolicOutputSet,
  env: EvalEnv,
  intrinsics: IntrinsicEvalTable = new Map(),
): { readonly resolved: ReadonlyArray<number>; readonly unresolvedCount: number } {
  const seen = new Set<number>();
  let unresolved = 0;
  for (const e of set.exprs) {
    const v = evaluateConcrete(e, env, intrinsics);
    if (v === undefined) unresolved++;
    else seen.add(v);
  }
  const resolved = [...seen].sort((a, b) => a - b);
  return { resolved, unresolvedCount: unresolved };
}

// ─────────────────────────────────────────────────────────────────────
// Structural value evaluation (for rules whose outputs are JS objects)
// ─────────────────────────────────────────────────────────────────────

/**
 * Evaluate an IR expression to an arbitrary JS value (number, boolean,
 * or plain object). Object-literal returns in a rule body lower to
 * `__record:<keys>` intrinsics (see `liftRuleObjectLiterals` in
 * `@aardworx/wombat.shader-vite`); we recognise that intrinsic and
 * build a JS object by recursively evaluating each field.
 *
 * Returns `undefined` for any unresolvable subexpression — the caller
 * treats those as unfoldable.
 */
export function evaluateStructural(
  e: Expr,
  env: EvalEnv,
  intrinsics: IntrinsicEvalTable = new Map(),
): unknown {
  const k = (e as { kind: string }).kind;
  if (k === "CallIntrinsic") {
    const ci = e as { op: { name: string }; args: ReadonlyArray<Expr> };
    if (ci.op.name.startsWith("__record:")) {
      const keys = ci.op.name.slice("__record:".length).split("|");
      if (keys.length !== ci.args.length) return undefined;
      const obj: Record<string, unknown> = {};
      for (let i = 0; i < keys.length; i++) {
        const v = evaluateStructural(ci.args[i]!, env, intrinsics);
        if (v === undefined) return undefined;
        obj[keys[i]!] = v;
      }
      return obj;
    }
  }
  // Fall through to numeric eval for everything else.
  return evaluateConcrete(e, env, intrinsics);
}

/**
 * Like `evaluateSet`, but for rule bodies whose outputs may be
 * structured JS objects (not just numbers). Returns a deterministic-
 * order array of distinct JS values, sorted by `stableStringify`.
 *
 * Used by `derivedMode` rules whose axis's `ModeValue` is an object
 * shape (e.g. blend — full `AttachmentBlend` struct). For enum axes
 * (cull etc.) the rule body still returns u32 indices and the
 * `resolved` array is numeric.
 */
export function evaluateStructuralSet(
  set: SymbolicOutputSet,
  env: EvalEnv,
  intrinsics: IntrinsicEvalTable = new Map(),
): { readonly resolved: ReadonlyArray<unknown>; readonly unresolvedCount: number } {
  const byKey = new Map<string, unknown>();
  let unresolved = 0;
  for (const e of set.exprs) {
    const v = evaluateStructural(e, env, intrinsics);
    if (v === undefined) { unresolved++; continue; }
    byKey.set(stableStringify(v), v);
  }
  const resolved = [...byKey.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([, v]) => v);
  return { resolved, unresolvedCount: unresolved };
}
