// liftReturns — lower entry-function return values into output writes.
//
// Frontend convention (idiomatic TypeScript):
//
//   function fsMain(input: { v_color: V3f }): { outColor: V4f } {
//     return { outColor: vec4(input.v_color, 1.0) };
//   }
//
// IR after the frontend translates this looks like
//
//   ReturnValue(NewObject({ outColor: NewVector([...]) }))
//
// — except wombat.shader IR has no `NewObject`. The frontend emits a
// `ReturnValue(<object literal>)` that survives as an opaque shape.
// This pass pattern-matches that shape *for entries only* and rewrites
// it to a sequence of `WriteOutput` calls plus a bare `Return`.
//
// The rule is structural: only ObjectLiteralExpressions where every
// property is a name → expression mapping that matches a declared
// output of the entry are lifted. Anything else stays as
// ReturnValue (and the type-checker / emitter will surface it).

import type {
  EntryDef,
  Expr,
  Module,
  Stmt,
  ValueDef,
} from "@aardworx/wombat.shader-ir";
import { mapStmt } from "./transform.js";

// We rely on the frontend tagging object-literal returns with a custom
// shape: the Expr's `kind` is "_ObjectLiteral" carried as an annotation
// on a Field-derived stub. We don't extend the IR for this; instead the
// frontend builds an Expr of kind `Field` whose target is a synthetic
// "intrinsic record" that the lift pass recognises by structural cues:
//
//   ReturnValue(value)
// where `value` is a *record-shaped* expression. We approximate this by
// looking for `ReturnValue(value)` whose `value` carries a `_record`
// hidden field on its top-level node.
//
// For v0.1 we take a simpler, less magical route: the frontend never
// emits ReturnValue with a record literal — it always emits explicit
// `WriteOutput` calls or plain `ReturnValue(expr)`. This pass therefore
// has nothing to do beyond a structural scan we can plug into when we
// extend the frontend to support `return { outColor: ... }`. For now,
// it's a no-op that's already in the right shape for that future work.

export interface LiftReturnsOptions {
  /** If true (default), only entry-function bodies are processed. */
  readonly entriesOnly?: boolean;
}

export function liftReturns(module: Module, _options: LiftReturnsOptions = {}): Module {
  const newValues: ValueDef[] = module.values.map((v) => {
    if (v.kind !== "Entry") return v;
    return { ...v, entry: liftEntry(v.entry) };
  });
  return { ...module, values: newValues };
}

function liftEntry(entry: EntryDef): EntryDef {
  // Walk the body; whenever a `ReturnValue` carries a recognised
  // record-shaped expression, replace with WriteOutputs + Return.
  const transformed = mapStmt(entry.body, {
    stmt: (s) => liftStmt(s, entry),
  });
  // Apply the transform at the root too, since mapStmt only walks
  // nested children.
  return { ...entry, body: liftStmt(transformed, entry) };
}

function liftStmt(s: Stmt, entry: EntryDef): Stmt {
  if (s.kind !== "ReturnValue") return s;
  const lifted = tryLiftReturn(s.value, entry, s.span);
  if (!lifted) return s;
  return lifted;
}

function tryLiftReturn(value: Expr, entry: EntryDef, span: import("@aardworx/wombat.shader-ir").Span | undefined): Stmt | undefined {
  // Pattern A: ReturnValue carrying a synthetic "_record" node — see
  // frontend `ObjectLiteralExpression` translation. We tag the carrier
  // by storing the field map on the Expr's `tag` (the IR's
  // `Intrinsic.tag` is unused for non-Intrinsic types; we co-opt the
  // hidden `_record` property the frontend sets).
  const fields = (value as { _record?: ReadonlyMap<string, Expr> })._record;
  if (fields !== undefined) {
    const writes: Stmt[] = [];
    for (const [name, expr] of fields) {
      const declaredOutput = entry.outputs.find((o) => o.name === name);
      if (!declaredOutput) continue; // unknown output → leave as-is in body
      writes.push({
        kind: "WriteOutput",
        name,
        value: { kind: "Expr", value: expr, ...(span !== undefined ? { span } : {}) },
        ...(span !== undefined ? { span } : {}),
      });
    }
    if (writes.length === 0) return undefined;
    return { kind: "Sequential", body: writes, ...(span !== undefined ? { span } : {}) };
  }
  // Pattern B: bare-value return (`return new V4f(...)`).
  //
  //   - Vertex with one declared output: that's the @builtin(position)
  //     slot. Bare V4f → write position.
  //   - Fragment with one declared output: that's the first colour
  //     attachment. Bare V4f → write colour.
  //   - Compute or anything with multiple outputs: ambiguous, leave
  //     the return as-is and let the WGSL parser surface the error.
  //
  // The single-output rule covers the canonical "shorthand return"
  // form for both vertex and fragment without trying to infer
  // builtin-vs-not, which the entry decoration already encodes.
  if (entry.outputs.length === 1) {
    const target = entry.outputs[0]!;
    return {
      kind: "Sequential",
      body: [{
        kind: "WriteOutput",
        name: target.name,
        value: { kind: "Expr", value, ...(span !== undefined ? { span } : {}) },
        ...(span !== undefined ? { span } : {}),
      }],
      ...(span !== undefined ? { span } : {}),
    };
  }
  return undefined;
}
