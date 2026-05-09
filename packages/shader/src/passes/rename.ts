// Rename — Module-level identifier rewriting.
//
// Scope of the API:
//   * `renameInputs`   — rename `ReadInput(scope, oldName)` reads,
//                        plus the matching Entry input/output param
//                        decls (when `scope` is "Input"/"Output") or
//                        the `UniformDecl.name`s (when `scope` is
//                        "Uniform").
//   * `renameOutputs`  — rename `WriteOutput`/`LInput("Output", …)`
//                        writes plus Entry output param decls.
//   * `renameVars`     — rename a specific `Var` (by object identity);
//                        every `Var` / `LVar` / `Declare` reference
//                        gets retargeted to a single fresh `Var`
//                        object with the new name (preserves the
//                        codebase's per-body Var-identity invariant).
//   * `renameTypes`    — rename Struct types. Every `Type` node with
//                        `kind: "Struct"` and a name in the mapping
//                        is rewritten; that walks ValueDef types,
//                        Var types, Expr types, RExpr `arrayType`,
//                        function signatures, Entry params, etc.
//   * `renameEntries`  — rename `Entry.entry.name` (Entries can't be
//                        called from user code, so no Call rewrite
//                        is needed).
//   * `renameFunctions`— rename `Function.signature.name` plus every
//                        `Call.fn.signature.name` reference.
//
// Conflict policy:
//   * If `oldName` is not present, the rename is a silent no-op for
//     that key — saves callers from having to feature-detect.
//   * If `newName` would collide with an existing identifier of the
//     same kind, we throw. The throw saves consumers from
//     hard-to-debug downstream parse failures (e.g. two Entries
//     called "main").

import type {
  EntryDef,
  EntryParameter,
  Expr,
  ExprBody,
  FunctionSignature,
  InputScope,
  LExpr,
  Module,
  Parameter,
  RExpr,
  Stage,
  Stmt,
  StructField,
  Type,
  TypeDef,
  UniformDecl,
  ValueDef,
  Var,
} from "../ir/index.js";
import { mapExpr, mapLExpr, mapRExpr, mapStmt } from "./transform.js";

// ─── RenameMapping — type-aware mapping API ────────────────────────────
//
// All rename APIs that benefit from type information accept either a
// plain `ReadonlyMap` (ergonomic shorthand, name-based) OR a function
// `(name, meta) => newName | undefined`. The function form receives
// the type alongside the name so callers can disambiguate
// same-name-different-type cases without a separate inspection step.
//
//   * For input/output/uniform/varying APIs, `meta` is the `Type`.
//   * For `renameVars`, `meta` is the `Var` itself (which carries
//     `var.type`, `var.mutable`, etc.).
//
// The function returns `undefined` to leave a name alone — same
// semantics as `Map.get(name)` returning `undefined`.

export type RenameMapping<TKey = string, TKeyMeta = Type> =
  | ReadonlyMap<TKey, string>
  | ((key: TKey, meta: TKeyMeta) => string | undefined);

function isMap<K, V>(m: ReadonlyMap<K, V> | unknown): m is ReadonlyMap<K, V> {
  return typeof m !== "function";
}

/**
 * Materialise a `RenameMapping<string, Type>` against a known universe
 * of `(name, type)` pairs. Returns a plain `Map<string, string>` that
 * the existing string-keyed collision-and-rewrite machinery can
 * consume unchanged. Names whose target equals the source, or whose
 * function returns `undefined`, are dropped — same as the Map form's
 * existing self-rename short-circuit.
 *
 * `universe` is allowed to repeat the same name with different types
 * (the Builtin / Closure read-site scan does this); we call the
 * function for every (name, type) pair, but only keep one entry per
 * name. If the function returns inconsistent rewrites for the same
 * name, the *last one wins* — same-name-different-type at one call
 * site is the caller's problem to disambiguate via the function body
 * (e.g. include the type kind in the new name).
 */
function materialiseNameMapping(
  mapping: RenameMapping<string, Type>,
  universe: Iterable<readonly [string, Type]>,
): Map<string, string> {
  const out = new Map<string, string>();
  if (isMap(mapping)) {
    for (const [from, to] of mapping) {
      if (from === to) continue;
      out.set(from, to);
    }
    return out;
  }
  for (const [name, type] of universe) {
    const next = mapping(name, type);
    if (next === undefined || next === name) continue;
    out.set(name, next);
  }
  return out;
}

// ─── Type tree walker ───────────────────────────────────────────────────
//
// Recursively rewrite a `Type`. Returns the same reference when nothing
// changed (so downstream identity-preservation passes stay quiet).

function mapType(t: Type, fn: (t: Type) => Type): Type {
  switch (t.kind) {
    case "Vector": {
      const element = mapType(t.element, fn);
      return fn(element === t.element ? t : { ...t, element });
    }
    case "Matrix": {
      const element = mapType(t.element, fn);
      return fn(element === t.element ? t : { ...t, element });
    }
    case "Array": {
      const element = mapType(t.element, fn);
      return fn(element === t.element ? t : { ...t, element });
    }
    case "Struct": {
      let changed = false;
      const fields: StructField[] = t.fields.map((f) => {
        const ft = mapType(f.type, fn);
        if (ft === f.type) return f;
        changed = true;
        return { ...f, type: ft };
      });
      return fn(changed ? { ...t, fields } : t);
    }
    default:
      return fn(t);
  }
}

// ─── Helpers: rewrite Type-bearing nodes via a single Type rewriter ─────
//
// Substitute happens after the Var canonicalisation done by
// renameVars, so a Var-rewrite map is supplied separately when
// needed. For type-only rewrites, both maps are identity.

function rewriteVarTypes(v: Var, typeFn: (t: Type) => Type): Var {
  const type = typeFn(v.type);
  return type === v.type ? v : { ...v, type };
}

function rewriteExprTypes(e: Expr, typeFn: (t: Type) => Type): Expr {
  return mapExpr(e, (x) => {
    const type = typeFn(x.type);
    let next: Expr = x;
    if (type !== x.type) next = { ...x, type } as Expr;
    if (next.kind === "Var") {
      const v2 = rewriteVarTypes(next.var, typeFn);
      if (v2 !== next.var) next = { ...next, var: v2 } as Expr;
    }
    return next;
  });
}

function rewriteLExprTypes(l: LExpr, typeFn: (t: Type) => Type): LExpr {
  return mapLExpr(l,
    (x) => {
      const type = typeFn(x.type);
      let next: LExpr = x;
      if (type !== x.type) next = { ...x, type } as LExpr;
      if (next.kind === "LVar") {
        const v2 = rewriteVarTypes(next.var, typeFn);
        if (v2 !== next.var) next = { ...next, var: v2 } as LExpr;
      }
      return next;
    },
    (x) => {
      const type = typeFn(x.type);
      let next: Expr = x;
      if (type !== x.type) next = { ...x, type } as Expr;
      if (next.kind === "Var") {
        const v2 = rewriteVarTypes(next.var, typeFn);
        if (v2 !== next.var) next = { ...next, var: v2 } as Expr;
      }
      return next;
    },
  );
}

function rewriteRExprTypes(r: RExpr, typeFn: (t: Type) => Type): RExpr {
  if (r.kind === "ArrayLiteral") {
    const arrayType = typeFn(r.arrayType);
    const mapped = mapRExpr(r, (e) => rewriteExprTypes(e, typeFn));
    if (arrayType === r.arrayType && mapped === r) return r;
    return { ...mapped, arrayType } as RExpr;
  }
  return mapRExpr(r, (e) => rewriteExprTypes(e, typeFn));
}

function rewriteStmtTypes(s: Stmt, typeFn: (t: Type) => Type): Stmt {
  // Walk both Vars, Exprs, LExprs and any nested types in Declares /
  // RExprs. Note: `Declare.var.type` needs rewriting; mapStmt's
  // expr-mapper covers Expr.type but not Declare.var.type, so we
  // handle Declare ourselves and delegate the rest.
  const exprFn = (e: Expr): Expr => rewriteExprTypes(e, typeFn);
  const lexprFn = (l: LExpr): LExpr => rewriteLExprTypes(l, typeFn);
  const stmtFn = (x: Stmt): Stmt => {
    if (x.kind === "Declare") {
      const v2 = rewriteVarTypes(x.var, typeFn);
      const init = x.init ? rewriteRExprTypes(x.init, typeFn) : undefined;
      if (v2 === x.var && init === x.init) return x;
      return { ...x, var: v2, ...(init !== undefined ? { init } : {}) };
    }
    if (x.kind === "WriteOutput") {
      const value = rewriteRExprTypes(x.value, typeFn);
      if (value === x.value) return x;
      return { ...x, value };
    }
    return x;
  };
  return mapStmt(s, { expr: exprFn, lexpr: lexprFn, stmt: stmtFn });
}

function rewriteParameterTypes(p: Parameter, typeFn: (t: Type) => Type): Parameter {
  const type = typeFn(p.type);
  return type === p.type ? p : { ...p, type };
}

function rewriteSignatureTypes(
  sig: FunctionSignature,
  typeFn: (t: Type) => Type,
): FunctionSignature {
  const returnType = typeFn(sig.returnType);
  let changed = returnType !== sig.returnType;
  const parameters = sig.parameters.map((p) => {
    const np = rewriteParameterTypes(p, typeFn);
    if (np !== p) changed = true;
    return np;
  });
  return changed ? { ...sig, returnType, parameters } : sig;
}

function rewriteEntryParamTypes(
  p: EntryParameter,
  typeFn: (t: Type) => Type,
): EntryParameter {
  const type = typeFn(p.type);
  return type === p.type ? p : { ...p, type };
}

// ─── Identifier-collision helpers ───────────────────────────────────────

function checkDistinctTargets(mapping: ReadonlyMap<string, string>, kind: string): void {
  // If the user maps two distinct old names onto the same new name,
  // that's its own (rarer) collision class. Reject it eagerly.
  const seen = new Set<string>();
  for (const [from, to] of mapping) {
    if (from === to) continue;
    if (seen.has(to)) {
      throw new Error(`rename${kind}: target name "${to}" specified twice in mapping`);
    }
    seen.add(to);
  }
}

function assertNoCollision(
  existing: ReadonlySet<string>,
  mapping: ReadonlyMap<string, string>,
  kind: string,
): void {
  for (const [from, to] of mapping) {
    if (from === to) continue;
    if (!existing.has(from)) continue; // silent if oldName not present
    // The new name must not already exist (and must not be the source
    // of another rename in the same map — that's the swap case, which
    // is unsupported here because we'd need a fresh-name pivot).
    if (existing.has(to) && !mapping.has(to)) {
      throw new Error(
        `rename${kind}: cannot rename "${from}" to "${to}" — "${to}" already exists`,
      );
    }
    if (mapping.has(to) && to !== from) {
      // "swap" — both A→B and B→A in the same call. Reject; the
      // caller can do two passes with a fresh intermediate name.
      const back = mapping.get(to);
      if (back === from) {
        throw new Error(
          `rename${kind}: simultaneous swap "${from}"↔"${to}" not supported; rename via a temporary`,
        );
      }
    }
  }
}

// ─── renameInputs / renameOutputs ───────────────────────────────────────

function renameReadsInBody(
  s: Stmt,
  scope: InputScope,
  mapping: ReadonlyMap<string, string>,
): Stmt {
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "ReadInput" && e.scope === scope) {
      const next = mapping.get(e.name);
      if (next !== undefined && next !== e.name) {
        return { ...e, name: next };
      }
    }
    return e;
  };
  const lexprFn = (l: LExpr): LExpr => {
    if (l.kind === "LInput" && l.scope === scope) {
      const next = mapping.get(l.name);
      if (next !== undefined && next !== l.name) {
        return { ...l, name: next };
      }
    }
    return l;
  };
  return mapStmt(s, { expr: exprFn, lexpr: lexprFn });
}

function renameWriteOutputsInBody(
  s: Stmt,
  mapping: ReadonlyMap<string, string>,
): Stmt {
  // WriteOutput renames go through Stmt.kind === "WriteOutput";
  // matching LInput("Output", …) renames are handled by passing
  // scope: "Output" through `renameReadsInBody`.
  const stmtFn = (x: Stmt): Stmt => {
    if (x.kind === "WriteOutput") {
      const next = mapping.get(x.name);
      if (next !== undefined && next !== x.name) {
        return { ...x, name: next };
      }
    }
    return x;
  };
  return mapStmt(
    renameReadsInBody(s, "Output", mapping),
    { stmt: stmtFn },
  );
}

function bodyIsApplicable(stage: Stage | undefined, entry: EntryDef): boolean {
  return stage === undefined || entry.stage === stage;
}

/**
 * Rename `ReadInput(scope, oldName)` → `ReadInput(scope, newName)` and
 * the matching identifier-bearing decls. For `scope === "Input"`,
 * Entry `inputs[]` param decls get renamed too. For `scope ===
 * "Uniform"`, every `UniformDecl.name` matching the mapping is
 * renamed. For "Output", routing goes through `renameOutputs` instead
 * — pass the mapping there.
 *
 * Function (helper) bodies are walked unconditionally because helpers
 * can be called from any stage. Entry bodies are walked when no
 * `stage` is specified, or only the matching stage when one is.
 */
function renameInputsImpl(
  m: Module,
  stage: Stage | undefined,
  scope: InputScope,
  rawMapping: RenameMapping<string, Type>,
): Module {
  // Build the universe of (name, type) pairs the rename can target.
  // Same scoping rules as before: scope decides whether we look at
  // Entry inputs, UniformDecls, or in-body read sites.
  const universe: Array<readonly [string, Type]> = [];
  if (scope === "Input") {
    for (const v of m.values) {
      if (v.kind === "Entry" && bodyIsApplicable(stage, v.entry)) {
        for (const p of v.entry.inputs) universe.push([p.name, p.type]);
      }
    }
  } else if (scope === "Uniform") {
    for (const v of m.values) {
      if (v.kind === "Uniform") for (const u of v.uniforms) universe.push([u.name, u.type]);
    }
  } else {
    // Builtin / Closure / Output (Output is the read-side of an
    // output binding; rare). Use the read sites as the universe.
    const collect = (s: Stmt): void => {
      mapStmt(s, {
        expr: (e) => {
          if (e.kind === "ReadInput" && e.scope === scope) universe.push([e.name, e.type]);
          return e;
        },
        lexpr: (l) => {
          if (l.kind === "LInput" && l.scope === scope) universe.push([l.name, l.type]);
          return l;
        },
      });
    };
    for (const v of m.values) {
      if (v.kind === "Function") collect(v.body);
      if (v.kind === "Entry" && bodyIsApplicable(stage, v.entry)) collect(v.entry.body);
    }
  }

  const mapping = materialiseNameMapping(rawMapping, universe);
  if (mapping.size === 0) return m;
  checkDistinctTargets(mapping, "Inputs");

  const existing = new Set<string>(universe.map(([n]) => n));
  assertNoCollision(existing, mapping, "Inputs");

  const values = m.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      const body = renameReadsInBody(v.body, scope, mapping);
      return body === v.body ? v : { ...v, body };
    }
    if (v.kind === "Entry") {
      if (!bodyIsApplicable(stage, v.entry)) return v;
      let entry = v.entry;
      const body = renameReadsInBody(entry.body, scope, mapping);
      if (body !== entry.body) entry = { ...entry, body };
      if (scope === "Input") {
        let changed = false;
        const inputs = entry.inputs.map((p) => {
          const n = mapping.get(p.name);
          if (n !== undefined && n !== p.name) {
            changed = true;
            return { ...p, name: n };
          }
          return p;
        });
        if (changed) entry = { ...entry, inputs };
      }
      return entry === v.entry ? v : { ...v, entry };
    }
    if (v.kind === "Uniform" && scope === "Uniform") {
      let changed = false;
      const uniforms = v.uniforms.map((u): UniformDecl => {
        const n = mapping.get(u.name);
        if (n !== undefined && n !== u.name) {
          changed = true;
          return { ...u, name: n };
        }
        return u;
      });
      return changed ? { ...v, uniforms } : v;
    }
    return v;
  });
  return values === m.values ? m : { ...m, values };
}

export function renameInputs(
  m: Module,
  scope: InputScope,
  mapping: RenameMapping<string, Type>,
): Module {
  return renameInputsImpl(m, undefined, scope, mapping);
}

export function renameInputsInStage(
  m: Module,
  stage: Stage,
  scope: InputScope,
  mapping: RenameMapping<string, Type>,
): Module {
  return renameInputsImpl(m, stage, scope, mapping);
}

function renameOutputsImpl(
  m: Module,
  stage: Stage | undefined,
  rawMapping: RenameMapping<string, Type>,
): Module {
  const universe: Array<readonly [string, Type]> = [];
  for (const v of m.values) {
    if (v.kind === "Entry" && bodyIsApplicable(stage, v.entry)) {
      for (const p of v.entry.outputs) universe.push([p.name, p.type]);
    }
  }
  const mapping = materialiseNameMapping(rawMapping, universe);
  if (mapping.size === 0) return m;
  checkDistinctTargets(mapping, "Outputs");

  const existing = new Set<string>(universe.map(([n]) => n));
  assertNoCollision(existing, mapping, "Outputs");

  const values = m.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      const body = renameWriteOutputsInBody(v.body, mapping);
      return body === v.body ? v : { ...v, body };
    }
    if (v.kind === "Entry") {
      if (!bodyIsApplicable(stage, v.entry)) return v;
      let entry = v.entry;
      const body = renameWriteOutputsInBody(entry.body, mapping);
      if (body !== entry.body) entry = { ...entry, body };
      let changed = false;
      const outputs = entry.outputs.map((p) => {
        const n = mapping.get(p.name);
        if (n !== undefined && n !== p.name) {
          changed = true;
          return { ...p, name: n };
        }
        return p;
      });
      if (changed) entry = { ...entry, outputs };
      return entry === v.entry ? v : { ...v, entry };
    }
    return v;
  });
  return values === m.values ? m : { ...m, values };
}

export function renameOutputs(
  m: Module,
  mapping: RenameMapping<string, Type>,
): Module {
  return renameOutputsImpl(m, undefined, mapping);
}

export function renameOutputsInStage(
  m: Module,
  stage: Stage,
  mapping: RenameMapping<string, Type>,
): Module {
  return renameOutputsImpl(m, stage, mapping);
}

// ─── renameVaryings ─────────────────────────────────────────────────────

/**
 * Rename inter-stage varyings. Walks the module and renames:
 *   - every Entry whose stage === "vertex":   outputs by name
 *   - every Entry whose stage === "fragment": inputs by name
 * AND every WriteOutput in vertex-stage bodies + every ReadInput("Input", ...)
 * in fragment-stage bodies whose name matches.
 *
 * Does NOT touch:
 *   - vertex inputs (attributes — those aren't varyings)
 *   - fragment outputs (fragment-output linker handles those)
 *   - compute stages (no varyings)
 *
 * The standard collision policy applies: throws if any newName collides
 * with an existing varying, or if oldNames map to colliding newNames.
 *
 * Composition: this is a thin wrapper over `renameOutputsInStage("vertex",...)`
 * + `renameInputsInStage("fragment", "Input", ...)`. Either side can no-op
 * silently if the name only exists on the other (e.g. a VS output with no
 * matching FS reader); that's by design.
 */
export function renameVaryings(
  m: Module,
  mapping: RenameMapping<string, Type>,
): Module {
  // Type-aware contract (Option A from the spec):
  //   The function form is called once per (name, VS-output-type) pair.
  //   Both the VS-output rewrite and the matching FS-input rewrite use
  //   the same returned `newName`. If the FS input was somehow declared
  //   with a different type than the VS output (a pre-existing IR bug
  //   `linkCrossStage` would catch separately), the rename still pairs
  //   them — the type mismatch isn't introduced by this pass.
  //
  // For the Map form, we just thread it through; both sub-passes
  // resolve names against the same Map.
  if (isMap(mapping)) {
    if (mapping.size === 0) return m;
    let out = renameOutputsImpl(m, "vertex", mapping);
    out = renameInputsImpl(out, "fragment", "Input", mapping);
    return out;
  }

  // Function form: build the effective Map from VS outputs once, then
  // hand the materialised Map to both passes. Names that exist only in
  // FS inputs (no VS writer) won't appear in the VS-output universe,
  // so the function isn't asked about them — that's fine: the standard
  // varyings semantic is paired writers + readers.
  const vsUniverse: Array<readonly [string, Type]> = [];
  for (const v of m.values) {
    if (v.kind === "Entry" && v.entry.stage === "vertex") {
      for (const p of v.entry.outputs) vsUniverse.push([p.name, p.type]);
    }
  }
  const effective = materialiseNameMapping(mapping, vsUniverse);
  if (effective.size === 0) return m;
  let out = renameOutputsImpl(m, "vertex", effective);
  out = renameInputsImpl(out, "fragment", "Input", effective);
  return out;
}

// ─── renameVars ─────────────────────────────────────────────────────────

/**
 * Rename `Var.name` for each entry in `mapping`. Identity is by `Var`
 * reference; one fresh `Var` object is created per rename and every
 * `Var` / `LVar` / `Declare.var` referencing the original gets
 * retargeted to it. This preserves the "one canonical Var per body"
 * invariant `relinkVars` establishes.
 */
export function renameVars(
  m: Module,
  mapping: RenameMapping<Var, Var>,
): Module {
  // Build a Var → fresh-Var substitution. Reject collisions inside
  // the same body lazily — we don't have a per-body universe handy
  // (Vars span helpers), so we do it during the rewrite.
  //
  // For the Map form, iterate the map directly. For the function
  // form, walk the module to find every Var (Declare site is the
  // canonical, post-`relinkVars` source of identity), call the
  // function with each, and collect the rewrites. This keeps the
  // Var-identity invariant intact — we look at the same Var objects
  // the bodies actually reference.
  const rebound = new Map<Var, Var>();
  if (isMap(mapping)) {
    if (mapping.size === 0) return m;
    for (const [oldVar, newName] of mapping) {
      if (oldVar.name === newName) continue;
      rebound.set(oldVar, { ...oldVar, name: newName });
    }
  } else {
    const fn = mapping;
    const visit = (s: Stmt): void => {
      mapStmt(s, {
        stmt: (x) => {
          if (x.kind === "Declare") {
            if (!rebound.has(x.var)) {
              const next = fn(x.var, x.var);
              if (next !== undefined && next !== x.var.name) {
                rebound.set(x.var, { ...x.var, name: next });
              }
            }
          }
          return x;
        },
      });
    };
    for (const v of m.values) {
      if (v.kind === "Function") visit(v.body);
      if (v.kind === "Entry") visit(v.entry.body);
    }
  }
  if (rebound.size === 0) return m;

  const exprFn = (e: Expr): Expr => {
    if (e.kind === "Var") {
      const r = rebound.get(e.var);
      if (r) return { ...e, var: r };
    }
    return e;
  };
  const lexprFn = (l: LExpr): LExpr => {
    if (l.kind === "LVar") {
      const r = rebound.get(l.var);
      if (r) return { ...l, var: r };
    }
    return l;
  };
  const stmtFn = (s: Stmt): Stmt => {
    if (s.kind === "Declare") {
      const r = rebound.get(s.var);
      if (r) return { ...s, var: r };
    }
    return s;
  };
  const walk = (b: Stmt) => mapStmt(b, { expr: exprFn, lexpr: lexprFn, stmt: stmtFn });

  const values = m.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      const body = walk(v.body);
      return body === v.body ? v : { ...v, body };
    }
    if (v.kind === "Entry") {
      const body = walk(v.entry.body);
      return body === v.entry.body ? v : { ...v, entry: { ...v.entry, body } };
    }
    return v;
  });

  // Per-body collision check: if any rewritten body now declares two
  // Vars with the same name, throw.
  for (const v of values) {
    let body: Stmt | undefined;
    if (v.kind === "Function") body = v.body;
    else if (v.kind === "Entry") body = v.entry.body;
    if (!body) continue;
    const seen = new Set<string>();
    const collect = (s: Stmt): Stmt => {
      if (s.kind === "Declare") {
        if (seen.has(s.var.name)) {
          throw new Error(
            `renameVars: collision — body declares two vars named "${s.var.name}" after rename`,
          );
        }
        seen.add(s.var.name);
      }
      return mapStmt(s, { stmt: collect });
    };
    collect(body);
  }
  return { ...m, values };
}

// ─── renameTypes ────────────────────────────────────────────────────────

/**
 * Rename Struct types by name. Every `Type` node with `kind: "Struct"`
 * and a name in the mapping is rewritten — that includes Var types,
 * Expr/LExpr `type` annotations, RExpr `arrayType`, function
 * signatures, Entry params, UniformDecl/StorageBuffer/Sampler types,
 * and module-level `TypeDef`s.
 */
export function renameTypes(
  m: Module,
  mapping: ReadonlyMap<string, string>,
): Module {
  if (mapping.size === 0) return m;
  checkDistinctTargets(mapping, "Types");

  // Universe: every struct name declared in module.types or appearing
  // inside any value's type tree.
  const existing = new Set<string>();
  for (const t of m.types) if (t.kind === "Struct") existing.add(t.name);
  // Collect struct names from value types as a safety net.
  const collectFromType = (t: Type): void => {
    if (t.kind === "Struct") existing.add(t.name);
    if (t.kind === "Vector" || t.kind === "Matrix" || t.kind === "Array") {
      collectFromType(t.element);
    }
    if (t.kind === "Struct") for (const f of t.fields) collectFromType(f.type);
  };
  for (const v of m.values) {
    if (v.kind === "Constant") collectFromType(v.varType);
    if (v.kind === "Function") {
      collectFromType(v.signature.returnType);
      for (const p of v.signature.parameters) collectFromType(p.type);
    }
    if (v.kind === "Entry") {
      collectFromType(v.entry.returnType);
      for (const p of v.entry.inputs) collectFromType(p.type);
      for (const p of v.entry.outputs) collectFromType(p.type);
      for (const p of v.entry.arguments) collectFromType(p.type);
    }
    if (v.kind === "Uniform") for (const u of v.uniforms) collectFromType(u.type);
    if (v.kind === "StorageBuffer") collectFromType(v.layout);
    if (v.kind === "Sampler") collectFromType(v.type);
  }
  assertNoCollision(existing, mapping, "Types");

  const typeFn = (t: Type): Type => {
    if (t.kind === "Struct") {
      const next = mapping.get(t.name);
      if (next !== undefined && next !== t.name) {
        return { ...t, name: next };
      }
    }
    return t;
  };
  const tf = (t: Type): Type => mapType(t, typeFn);

  const types = m.types.map((td): TypeDef => {
    if (td.kind === "Struct") {
      const newName = mapping.get(td.name);
      let next: TypeDef = td;
      if (newName !== undefined && newName !== td.name) {
        next = { ...next, name: newName };
      }
      // Rewrite field types (which may themselves reference renamed structs).
      let fieldsChanged = false;
      const fields = next.fields.map((f) => {
        const ft = tf(f.type);
        if (ft === f.type) return f;
        fieldsChanged = true;
        return { ...f, type: ft };
      });
      if (fieldsChanged) next = { ...next, fields };
      return next;
    }
    return td;
  });

  const values = m.values.map((v): ValueDef => {
    switch (v.kind) {
      case "Constant": {
        const varType = tf(v.varType);
        const init = rewriteRExprTypes(v.init, tf);
        if (varType === v.varType && init === v.init) return v;
        return { ...v, varType, init };
      }
      case "Function": {
        const signature = rewriteSignatureTypes(v.signature, tf);
        const body = rewriteStmtTypes(v.body, tf);
        if (signature === v.signature && body === v.body) return v;
        return { ...v, signature, body };
      }
      case "Entry": {
        let entry = v.entry;
        const returnType = tf(entry.returnType);
        if (returnType !== entry.returnType) entry = { ...entry, returnType };
        const inputs = entry.inputs.map((p) => rewriteEntryParamTypes(p, tf));
        if (inputs.some((p, i) => p !== entry.inputs[i])) entry = { ...entry, inputs };
        const outputs = entry.outputs.map((p) => rewriteEntryParamTypes(p, tf));
        if (outputs.some((p, i) => p !== entry.outputs[i])) entry = { ...entry, outputs };
        const args = entry.arguments.map((p) => rewriteEntryParamTypes(p, tf));
        if (args.some((p, i) => p !== entry.arguments[i])) entry = { ...entry, arguments: args };
        const body = rewriteStmtTypes(entry.body, tf);
        if (body !== entry.body) entry = { ...entry, body };
        return entry === v.entry ? v : { ...v, entry };
      }
      case "Uniform": {
        let changed = false;
        const uniforms = v.uniforms.map((u): UniformDecl => {
          const type = tf(u.type);
          if (type === u.type) return u;
          changed = true;
          return { ...u, type };
        });
        return changed ? { ...v, uniforms } : v;
      }
      case "StorageBuffer": {
        const layout = tf(v.layout);
        return layout === v.layout ? v : { ...v, layout };
      }
      case "Sampler": {
        const type = tf(v.type);
        return type === v.type ? v : { ...v, type };
      }
    }
  });

  return { ...m, types, values };
}

// ─── renameEntries / renameFunctions ────────────────────────────────────

export function renameEntries(
  m: Module,
  mapping: ReadonlyMap<string, string>,
): Module {
  if (mapping.size === 0) return m;
  checkDistinctTargets(mapping, "Entries");

  const existing = new Set<string>();
  for (const v of m.values) if (v.kind === "Entry") existing.add(v.entry.name);
  assertNoCollision(existing, mapping, "Entries");

  const values = m.values.map((v): ValueDef => {
    if (v.kind !== "Entry") return v;
    const next = mapping.get(v.entry.name);
    if (next === undefined || next === v.entry.name) return v;
    return { ...v, entry: { ...v.entry, name: next } };
  });
  return values === m.values ? m : { ...m, values };
}

export function renameFunctions(
  m: Module,
  mapping: ReadonlyMap<string, string>,
): Module {
  if (mapping.size === 0) return m;
  checkDistinctTargets(mapping, "Functions");

  const existing = new Set<string>();
  for (const v of m.values) if (v.kind === "Function") existing.add(v.signature.name);
  assertNoCollision(existing, mapping, "Functions");

  // Rewrite the FunctionRef.signature.name on every Call expression
  // anywhere in the module (Entry bodies + Function bodies).
  const exprFn = (e: Expr): Expr => {
    if (e.kind === "Call") {
      const next = mapping.get(e.fn.signature.name);
      if (next !== undefined && next !== e.fn.signature.name) {
        const fn = {
          ...e.fn,
          signature: { ...e.fn.signature, name: next },
        };
        return { ...e, fn } as Expr;
      }
    }
    return e;
  };
  const walk = (s: Stmt): Stmt => mapStmt(s, { expr: exprFn });

  const values = m.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      let signature = v.signature;
      const next = mapping.get(signature.name);
      if (next !== undefined && next !== signature.name) {
        signature = { ...signature, name: next };
      }
      const body = walk(v.body);
      if (signature === v.signature && body === v.body) return v;
      return { ...v, signature, body };
    }
    if (v.kind === "Entry") {
      const body = walk(v.entry.body);
      return body === v.entry.body ? v : { ...v, entry: { ...v.entry, body } };
    }
    return v;
  });
  return { ...m, values };
}

// Re-export ExprBody just to make TS happy if any consumer wants the
// narrowed expression-kind identifier (avoids unused-import lint).
export type _ExprBody = ExprBody;
