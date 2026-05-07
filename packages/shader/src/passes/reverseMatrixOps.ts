// Row/column-major reversal pass.
//
// wombat.base stores matrices row-major (Aardvark convention).
// GLSL/WGSL store matrices column-major. When we upload row-major
// data raw to the GPU, the shader sees the *transpose* of what the
// CPU computed.
//
// FShade's solution — and ours — is to leave the data alone and
// instead reverse every matrix operation in the IR. The math then
// agrees:
//
//   `mvp.mul(v)` on the CPU computes `M * v` row-major.
//   The shader receives `M_gpu = M_row^T` (transpose, by virtue of
//   memory interpretation). Emitting the *reversed* form
//   `v * M_gpu` evaluates to `v * M_row^T = (M_row * v)^T = M_row * v`
//   for vectors, which is exactly what the CPU intended.
//
// Likewise `A.mul(B)` becomes `B * A` in shader source — the GPU's
// `B_gpu * A_gpu = B_row^T * A_row^T = (A_row * B_row)^T`, which is
// the GPU's view of the row-major product the CPU would compute.
//
// `M44f.fromCols(c0, …, c3)` flips to a "from rows" emit so the
// shader's column-major construction sees the right shape after the
// row-major upload.
//
// `Transpose(M)` keeps an explicit `transpose(...)` in WGSL. It
// would be tempting to elide it on the grounds that "the GPU already
// sees `transpose(M_cpu)`, so `M.transpose()` is a free no-op" —
// FShade's reasoning if you stop after a single load. But the rest
// of the pass maintains the invariant *WGSL value of expression X =
// transpose(DSL value of X)*; that's the only reason the operand
// flips for `MulMatVec`, `MulMatMat`, `MatrixCol/Row`, and
// `MatrixFromCols/Rows` produce the right answer. Under that
// invariant the WGSL value of `Transpose(M)` must be
// `transpose(transpose(M_cpu)) = M_cpu`, *not* `transpose(M_cpu)` —
// so we leave the IR node alive and let emit produce the literal
// `transpose(M_wgsl)` WGSL call (= `M_cpu`). Vanishing it would be
// off by a transpose for any composed expression like
// `M.transpose().mul(v)` or
// `uni_nm · transpose(m33(InstanceTrafoInv))`.
//
// The pass tags the module so a second invocation is a no-op
// (idempotent). Run *after* `inferStorageAccess` and *before*
// `legaliseTypes` in `compileModule`.

import type { Expr, Module, Type, ValueDef } from "../ir/index.js";
import { mapStmt } from "./transform.js";

/** @internal — module-level tag set when the pass runs. */
const REVERSED_TAG = "__wombatMatrixReversed";

export function reverseMatrixOps(module: Module): Module {
  if ((module.meta as Record<string, unknown> | undefined)?.[REVERSED_TAG]) {
    return module;
  }
  const values = module.values.map((v): ValueDef => {
    if (v.kind === "Function") {
      return { ...v, body: mapStmt(v.body, { expr: rewrite }) };
    }
    if (v.kind === "Entry") {
      return { ...v, entry: { ...v.entry, body: mapStmt(v.entry.body, { expr: rewrite }) } };
    }
    return v;
  });
  const meta = { ...(module.meta ?? {}), [REVERSED_TAG]: true } as NonNullable<Module["meta"]>;
  return { ...module, values, meta };
}

// ─────────────────────────────────────────────────────────────────────

function rewrite(e: Expr): Expr {
  switch (e.kind) {
    case "MulMatVec": {
      // `M * v` row-major  →  `v * M` column-major (same result on
      // GPU after row-major upload). When the matrix is `Transpose(X)`
      // the transpose absorbs the operand swap: DSL
      // `transpose(X).mul(v)` is `transpose(X_cpu) · v`, which the
      // column-vec form `X_wgsl * v` already computes (since
      // `X_wgsl = transpose(X_cpu)`). Drop the `Transpose` and keep
      // `MulMatVec` — same math, no `transpose(...)` call in WGSL.
      //
      // We also chain-lower `(M1·M2)·v → M1·(M2·v)` (right-
      // associative mat-vec chain). Each mat-vec is 4 dot-4s (16
      // mul-adds for mat4·vec4); each mat-mat is 16 dot-4s (64
      // mul-adds). Re-associating saves ~60% of the work.  Bottom-
      // up traversal means the inner `MulMatMat` has already been
      // operand-swapped by the reverse pass; tracking the names with
      // the swap in mind: post-reverse `MulMatMat(M1', M2')`
      // represents the original DSL `M2' · M1'` (semantics
      // preserved by the operand flip + upload-trick), so the
      // re-associated form is `MulVecMat(MulVecMat(v, M1'), M2')`,
      // which emits `(v * M1') * M2'` — the right-associative chain.
      // CSE has already bound multi-use mat-mats to Vars (so a
      // literal `MulMatMat` here is single-use; lowering is always
      // a win).
      const result = e.lhs.type.kind === "Matrix"
        ? { kind: "Vector" as const, element: e.lhs.type.element, dim: e.lhs.type.cols }
        : e.type;
      if (e.lhs.kind === "Transpose") {
        return { kind: "MulMatVec", lhs: e.lhs.value, rhs: e.rhs, type: result };
      }
      if (e.lhs.kind === "MulMatMat") {
        const M1p = e.lhs.lhs;
        const M2p = e.lhs.rhs;
        const innerVecType: Type = M1p.type.kind === "Matrix" && e.rhs.type.kind === "Vector"
          ? { kind: "Vector", element: M1p.type.element, dim: M1p.type.cols }
          : e.rhs.type;
        const inner: Expr = { kind: "MulVecMat", lhs: e.rhs, rhs: M1p, type: innerVecType };
        return { kind: "MulVecMat", lhs: inner, rhs: M2p, type: result };
      }
      return { kind: "MulVecMat", lhs: e.rhs, rhs: e.lhs, type: result };
    }
    case "MulVecMat": {
      // Symmetric of the above (rare in user code, common after a
      // chain swap). Same `Transpose` absorption — `v * transpose(X)`
      // ≡ `M * v` ≡ `MulMatVec(X, v)` on the GPU. Same chain-lowering
      // for `v · (M1·M2) → (v · M1) · M2`.
      const result = e.rhs.type.kind === "Matrix"
        ? { kind: "Vector" as const, element: e.rhs.type.element, dim: e.rhs.type.rows }
        : e.type;
      if (e.rhs.kind === "Transpose") {
        return { kind: "MulVecMat", lhs: e.lhs, rhs: e.rhs.value, type: result };
      }
      if (e.rhs.kind === "MulMatMat") {
        const M1p = e.rhs.lhs;
        const M2p = e.rhs.rhs;
        const innerVecType: Type = M2p.type.kind === "Matrix" && e.lhs.type.kind === "Vector"
          ? { kind: "Vector", element: M2p.type.element, dim: M2p.type.rows }
          : e.lhs.type;
        const inner: Expr = { kind: "MulMatVec", lhs: M2p, rhs: e.lhs, type: innerVecType };
        return { kind: "MulMatVec", lhs: M1p, rhs: inner, type: result };
      }
      return { kind: "MulMatVec", lhs: e.rhs, rhs: e.lhs, type: result };
    }
    case "MulMatMat": {
      // `A * B` row-major  →  `B * A` column-major.
      const a = e.lhs.type, b = e.rhs.type;
      const result = a.kind === "Matrix" && b.kind === "Matrix"
        ? { kind: "Matrix" as const, element: a.element, rows: b.rows, cols: a.cols }
        : e.type;
      return { kind: "MulMatMat", lhs: e.rhs, rhs: e.lhs, type: result };
    }
    // `Transpose` is intentionally not rewritten — it survives to
    // emit, which produces a literal `transpose(...)` WGSL call. See
    // the file header for why "the GPU already has the transpose,
    // vanish it" is wrong as soon as the result composes with
    // anything else.
    case "MatrixFromCols": {
      // Reinterpret as a "from rows" construction in shader source —
      // when the row-major data is uploaded, the GPU sees columns
      // where the user supplied rows. Flip so the user-supplied
      // arguments end up where they're expected.
      return { kind: "MatrixFromRows", rows: e.cols, type: e.type };
    }
    case "MatrixFromRows":
      return { kind: "MatrixFromCols", cols: e.rows, type: e.type };
    case "MatrixRow": {
      // CPU's "row r" lives at GPU column r after the row-major
      // upload trick (the column-major-interpreted bytes line up
      // 1:1 with rows of the original CPU matrix). Swap to a
      // MatrixCol so the standard column-major emit yields `M[r]`.
      return { kind: "MatrixCol", matrix: e.matrix, col: e.row, type: e.type };
    }
    case "MatrixCol": {
      // Symmetric: CPU's "column c" no longer lines up with a
      // single GPU vec4 read; emit needs the per-row pick that the
      // standard `MatrixRow` emit already produces.
      return { kind: "MatrixRow", matrix: e.matrix, row: e.col, type: e.type };
    }
    default:
      return e;
  }
}
