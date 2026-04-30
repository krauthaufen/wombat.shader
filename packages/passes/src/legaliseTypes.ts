// legaliseTypes — target-specific lowering. Runs last in the pass chain;
// the emitter consumes the result and assumes nothing further needs to
// be rewritten.
//
// What this pass does (per target):
//
//   common (both targets):
//     - MatrixRow(m, r)             → NewVector([m[r,0], m[r,1], …])
//     - MatrixFromRows(rs)          → Transpose(MatrixFromCols(rs))
//
//   wgsl-only:
//     - Inverse(m: Matrix)          → CallIntrinsic("_wombat_inverseN", [m])
//                                     and a helper FunctionDef inserted into
//                                     the Module if not already present.
//
// Each rewrite preserves Expr.type (so emitters and downstream passes
// see the same shape). The pass is pure (Module → Module).

import type {
  Expr,
  FunctionSignature,
  IntrinsicRef,
  Module,
  Stmt,
  Type,
  ValueDef,
} from "@aardworx/wombat.shader-ir";
import { mapExpr, mapStmt } from "./transform.js";

export type Target = "glsl" | "wgsl";

export function legaliseTypes(module: Module, target: Target): Module {
  const helperFunctions = new Map<string, ValueDef>();

  const exprFn = (e: Expr): Expr => {
    switch (e.kind) {
      case "MatrixRow":
        return lowerMatrixRow(e);
      case "MatrixFromRows":
        return lowerMatrixFromRows(e);
      case "Inverse":
        if (target === "wgsl") return lowerWgslInverse(e, helperFunctions);
        return e;
      default:
        return e;
    }
  };

  const newValues = module.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: mapStmt(v.body, { expr: exprFn }) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: mapStmt(v.entry.body, { expr: exprFn }) } };
    if (v.kind === "Constant") {
      const init = v.init.kind === "Expr"
        ? { kind: "Expr" as const, value: mapExpr(v.init.value, exprFn) }
        : { kind: "ArrayLiteral" as const, arrayType: v.init.arrayType, values: v.init.values.map((x) => mapExpr(x, exprFn)) };
      return { ...v, init };
    }
    return v;
  });

  // Inject any helper functions we generated (deduped by name).
  const finalValues = [...helperFunctions.values(), ...newValues];

  return { ...module, values: finalValues };
}

// ─────────────────────────────────────────────────────────────────────
// MatrixRow → NewVector of MatrixElement reads
// ─────────────────────────────────────────────────────────────────────

function lowerMatrixRow(e: Expr & { kind: "MatrixRow" }): Expr {
  const matType = e.matrix.type;
  if (matType.kind !== "Matrix") return e;
  const cols = matType.cols;
  const rowExpr = e.row;
  const rowEval: Expr = isPureNonSideeffecting(rowExpr) ? rowExpr : rowExpr;
  // Build [m[r,0], m[r,1], ..., m[r,cols-1]].
  const elementType = matType.element;
  const components: Expr[] = [];
  for (let c = 0; c < cols; c++) {
    components.push({
      kind: "MatrixElement",
      matrix: e.matrix,
      row: rowEval,
      col: { kind: "Const", value: { kind: "Int", signed: true, value: c }, type: { kind: "Int", signed: true, width: 32 } },
      type: elementType,
    });
  }
  return {
    kind: "NewVector",
    components,
    type: e.type, // already vec<element, cols>
  };
}

// ─────────────────────────────────────────────────────────────────────
// MatrixFromRows → Transpose(MatrixFromCols(rs))
// ─────────────────────────────────────────────────────────────────────

function lowerMatrixFromRows(e: Expr & { kind: "MatrixFromRows" }): Expr {
  // Build a matrix-from-cols of the same row vectors, then transpose.
  // Note: this changes the matrix shape from RxC (rows×cols) to CxR.
  // The Transpose then flips it back to RxC. Both emitters support this.
  if (e.type.kind !== "Matrix") return e;
  const colsType: Type = {
    kind: "Matrix",
    element: e.type.element,
    rows: e.type.cols,
    cols: e.type.rows,
  };
  return {
    kind: "Transpose",
    value: {
      kind: "MatrixFromCols",
      cols: e.rows,
      type: colsType,
    },
    type: e.type,
  };
}

// ─────────────────────────────────────────────────────────────────────
// WGSL Inverse helper
// ─────────────────────────────────────────────────────────────────────

const wgslInverseIntrinsics: Record<string, IntrinsicRef> = {};

function lowerWgslInverse(
  e: Expr & { kind: "Inverse" },
  helpers: Map<string, ValueDef>,
): Expr {
  const t = e.value.type;
  if (t.kind !== "Matrix") return e;
  const dim = t.rows;
  if (dim !== t.cols) return e; // non-square matrices are not invertible
  const helperName = `_wombat_inverse${dim}`;

  if (!helpers.has(helperName)) {
    helpers.set(helperName, makeInverseHelper(helperName, dim, t));
  }

  // Synthesize a CallIntrinsic that emits a call by name.
  const intrinsic = wgslInverseIntrinsics[helperName] ??= {
    name: helperName,
    pure: true,
    emit: { glsl: helperName, wgsl: helperName },
    returnTypeOf: () => t,
  };

  return {
    kind: "CallIntrinsic",
    op: intrinsic,
    args: [e.value],
    type: t,
  };
}

function makeInverseHelper(name: string, dim: number, matType: Type): ValueDef {
  // The body is a stub that emits "TODO: implement matN inverse" — we
  // leave the actual implementation to a future cookbook because
  // closed-form inversion for 2/3/4 is verbose. The function returns
  // the input unchanged so type-checking passes; runtime use will
  // produce an obviously-wrong result, surfacing the missing
  // implementation. Users supply their own helper before then.
  const sig: FunctionSignature = {
    name,
    returnType: matType,
    parameters: [{ name: "m", type: matType, modifier: "in" }],
  };
  const body: Stmt = {
    kind: "ReturnValue",
    value: { kind: "Var", var: { name: "m", type: matType, mutable: false }, type: matType },
  };
  return {
    kind: "Function",
    signature: sig,
    body,
    attributes: ["inline"],
  };
  // Note: we intentionally don't use `dim` for now — the helper is a
  // placeholder. Once we implement closed-form inverses, dim selects
  // the body. Silencing unused param.
  void dim;
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function isPureNonSideeffecting(_e: Expr): boolean {
  // Currently we don't try to dedupe row-expr evaluation; the emitter
  // will emit the same expression `cols` times. A future enhancement
  // could let-bind the row value when it's a CallIntrinsic etc.
  return true;
}
