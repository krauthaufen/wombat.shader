// Function inlining + simple copy propagation.
//
// Two responsibilities:
//
// 1. Inline `Call` expressions whose target is a user `Function` decl
//    annotated `inline` (or always — a policy decides). The body is
//    copy-pasted with parameters substituted by argument expressions.
//    Functions returning a value get rewritten as a let-binding +
//    last-expression-as-result.
//
// 2. Copy propagation: a `Declare` of an immutable Var with a trivial
//    initialiser (Var, Const, or a small Field/VecSwizzle) is dropped
//    and every read replaced by the initialiser.
//
// We only inline single-return functions (frontend-emitted shader
// functions are typically single-return). Multi-return inlining
// would need labelled-block lowering — out of scope for v0.1.

import type {
  Expr,
  Module,
  Stmt,
  ValueDef,
  Var,
} from "../ir/index.js";
import { mapExpr, mapStmt, mapStmtChildren } from "./transform.js";
import { substVars, substVarsExpr } from "./substitute.js";

export interface InlinePolicy {
  /** Decide whether to inline a particular call. */
  shouldInline(fnId: string): boolean;
}

export const inlineAllAttributed: InlinePolicy = {
  shouldInline: () => true,
};

export function inlinePass(module: Module, policy: InlinePolicy = inlineAllAttributed): Module {
  // Index user functions by id.
  const fns = new Map<string, Extract<ValueDef, { kind: "Function" }>>();
  for (const v of module.values) {
    if (v.kind === "Function") fns.set(v.signature.name, v);
  }

  const inlineExpr = (e: Expr): Expr => {
    if (e.kind !== "Call") return e;
    const fn = fns.get(e.fn.id);
    if (!fn) return e;
    if (!policy.shouldInline(e.fn.id)) return e;
    const inlined = tryInline(fn, e.args.map(inlineExpr) /* args inline first */);
    return inlined ?? e;
  };

  const runOn = (s: Stmt): Stmt => copyPropStmt(mapStmt(s, { expr: inlineExpr, stmt: copyPropStmt }));

  const newValues = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: runOn(v.body) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: runOn(v.entry.body) } };
    return v;
  });

  return { ...module, values: newValues };
}

// ─────────────────────────────────────────────────────────────────────
// Per-call inlining
// ─────────────────────────────────────────────────────────────────────

function tryInline(
  fn: Extract<ValueDef, { kind: "Function" }>,
  args: readonly Expr[],
): Expr | undefined {
  // Only handle the canonical "function = single ReturnValue" pattern.
  // Anything more complex stays as a Call for now.
  const ret = singleReturn(fn.body);
  if (!ret) return undefined;
  // Reject if any parameter is `inout` — we don't have an LExpr arg to bind to here.
  if (fn.signature.parameters.some((p) => p.modifier === "inout")) return undefined;
  // Bind parameters to argument expressions.
  const mapping = new Map<Var, Expr>();
  for (let i = 0; i < fn.signature.parameters.length; i++) {
    const p = fn.signature.parameters[i]!;
    const a = args[i];
    if (!a) return undefined;
    // We construct a synthetic Var matching the parameter (the function
    // body will reference the param Var; if the body was authored against
    // a different Var identity for the same name, this won't work — but
    // the frontend constructs both from the same source.)
    mapping.set(syntheticParamVar(p.name, p.type), a);
  }
  return substVarsExpr(ret, mapping);
}

function singleReturn(body: Stmt): Expr | undefined {
  if (body.kind === "ReturnValue") return body.value;
  if (body.kind === "Sequential" || body.kind === "Isolated") {
    if (body.body.length === 1) return singleReturn(body.body[0]!);
    if (body.body.length === 0) return undefined;
    const last = body.body[body.body.length - 1]!;
    // All non-last statements must be pure-and-side-effect-free; for now
    // we conservatively reject anything that isn't a single ReturnValue.
    // (Improving this requires a let-binding-of-pure-decls reshape.)
    if (body.body.every((s, i) => i === body.body.length - 1 || s.kind === "Nop")) {
      return singleReturn(last);
    }
    return undefined;
  }
  return undefined;
}

function syntheticParamVar(name: string, type: Expr["type"]): Var {
  return { name, type, mutable: false };
}

// ─────────────────────────────────────────────────────────────────────
// Copy propagation
// ─────────────────────────────────────────────────────────────────────

function copyPropStmt(s: Stmt): Stmt {
  if (s.kind !== "Sequential" && s.kind !== "Isolated") return s;
  const out: Stmt[] = [];
  const subst = new Map<Var, Expr>();
  for (const child of s.body) {
    // Substitute variables we've already bound.
    const propagated = subst.size > 0
      ? mapStmtChildren(child, { expr: (e) => substVarsExpr(e, subst) })
      : child;
    if (propagated.kind === "Declare" && propagated.init?.kind === "Expr") {
      const v = propagated.var;
      const init = propagated.init.value;
      if (!v.mutable && isTriviallyCopyable(init)) {
        subst.set(v, init);
        // Drop the declaration; uses are inlined.
        continue;
      }
    }
    out.push(propagated);
  }
  return { ...s, body: out };
}

function isTriviallyCopyable(e: Expr): boolean {
  switch (e.kind) {
    case "Var":
    case "Const":
    case "ReadInput":
      return true;
    case "VecSwizzle":
      return e.value.kind === "Var" || e.value.kind === "ReadInput";
    case "Field":
      return e.target.kind === "Var" || e.target.kind === "ReadInput";
    default:
      return false;
  }
}

// Used by tests.
export const _internal = { tryInline, singleReturn, copyPropStmt };
// silence unused-import warning for substVars (kept exported by the module)
void substVars;
void mapExpr;
