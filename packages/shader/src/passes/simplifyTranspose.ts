// Algebraic Transpose-propagation pass.
//
// Pushes `Transpose` IR nodes through the expression tree using
// well-known matrix identities, until each `Transpose` either cancels
// with another, gets reabsorbed by a row/col-orientation flip, or
// reaches a true leaf (a uniform read, a vertex input, etc.) where it
// has to stay and produce a real `transpose(...)` call at emit time.
//
// Identities applied (each uses post-order, i.e. children are
// simplified first; the rule below is the one that fires when we
// finally examine the parent `Transpose`):
//
//   T(T(X))                  → X
//   T(MatrixFromCols([cs]))  → MatrixFromRows([cs])
//   T(MatrixFromRows([rs]))  → MatrixFromCols([rs])
//   T(MulMatMat(A, B))       → MulMatMat(T(B), T(A))    (distribute)
//
// The distribution rule introduces fresh `Transpose` nodes that
// haven't been visited yet, so we recurse on them manually. Worst
// case the recursion stops at the operands of the original `MulMatMat`
// — at which point the new `T(B)` / `T(A)` either compose with another
// inner `Transpose` (cancel) or stay (real emit).
//
// Designed to run **before** `reverseMatrixOps`. After this pass, the
// only `Transpose` IR nodes left wrap genuine leaves (uniform reads,
// vertex inputs, function results we can't see through). The `MulMat*`
// /`Mat·v` rewrites in `reverseMatrixOps` then take care of the rest:
// `MulMatVec(Transpose(M), v)` absorbs into `MulMatVec(M, v)` (column-
// vec form), `MulMatMat`'s operand swap handles the remaining
// `MulMatMat(T(...), ...)` correctly, etc.
//
// This is the algebraic core of FShade's "row-major-on-CPU,
// column-major-on-GPU, no manual transposes anywhere" trick.

import type { Expr, Module, Type, ValueDef } from "../ir/index.js";
import { mapStmt } from "./transform.js";

/** @internal — module-level tag set when the pass runs (idempotent). */
const SIMPLIFIED_TAG = "__wombatTransposeSimplified";

export function simplifyTranspose(module: Module): Module {
  if ((module.meta as Record<string, unknown> | undefined)?.[SIMPLIFIED_TAG]) {
    return module;
  }
  const values = module.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      return { ...v, body: mapStmt(v.body, { expr: simplifyExpr }) };
    }
    if (v.kind === "Entry") {
      return { ...v, entry: { ...v.entry, body: mapStmt(v.entry.body, { expr: simplifyExpr }) } };
    }
    return v;
  });
  const meta = { ...(module.meta ?? {}), [SIMPLIFIED_TAG]: true } as NonNullable<Module["meta"]>;
  return { ...module, values, meta };
}

// ─────────────────────────────────────────────────────────────────────

/** Post-order: children are simplified before parents. */
function simplifyExpr(e: Expr): Expr {
  if (e.kind === "Transpose") return simplifyTransposeOf(e.value, e.type);
  return e;
}

/**
 * Returns the simplified form of `Transpose(value)`, given that
 * `value` has already been simplified. Recurses into the freshly-
 * introduced `Transpose` nodes from the `MulMatMat` distribution rule
 * (those weren't visited by the surrounding `mapExpr`).
 */
function simplifyTransposeOf(value: Expr, transposedType: Type): Expr {
  switch (value.kind) {
    case "Transpose":
      // T(T(X)) = X. Cancellation. The inner X has already been
      // simplified during the post-order pass.
      return value.value;
    case "MatrixFromCols":
      // Reinterpret: `mat-from-cols(cs).transpose()` ≡ `mat-from-rows(cs)`.
      return { kind: "MatrixFromRows", rows: value.cols, type: transposedType };
    case "MatrixFromRows":
      return { kind: "MatrixFromCols", cols: value.rows, type: transposedType };
    // Note: we DON'T distribute `T(A · B) → T(B) · T(A)`. Distribution
    // would move the Transpose IR away from the outer matmul — but the
    // common context for `T(A · B)` is composition with a vec-mul
    // (e.g. `T(A · B).mul(v)` in the NormalMatrix rebuild), and
    // `reverseMatrixOps`' MulMatVec absorb (`MulMatVec(T(M), v)` →
    // `MulMatVec(M, v)`, no flip) eliminates the Transpose *entirely*
    // when it can sit on top of a MulMatMat that flips its operands
    // anyway. Distributing destroys that opportunity and forces two
    // explicit `transpose(...)` calls in WGSL where a clean
    // `(B * A) * v` would do. See `tests/simplify-transpose.test.ts`
    // and `tests/matrix-reversal.test.ts`.
    default:
      // Genuine leaf — the Transpose has to stay. Emit will produce a
      // real `transpose(...)` WGSL call (or a downstream absorb may
      // still eat it).
      return { kind: "Transpose", value, type: transposedType };
  }
}

function transposeMatrixType(t: Type): Type {
  if (t.kind !== "Matrix") return t;
  return { kind: "Matrix", element: t.element, rows: t.cols, cols: t.rows };
}

/** Re-export the workhorse for unit-testing in isolation. */
export { simplifyTransposeOf as _simplifyTransposeOf };
