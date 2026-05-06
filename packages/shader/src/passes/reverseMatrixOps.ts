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
// `Transpose(M)` becomes a no-op: the GPU already sees the
// transposed view.
//
// The pass tags the module so a second invocation is a no-op
// (idempotent). Run *after* `inferStorageAccess` and *before*
// `legaliseTypes` in `compileModule`.

import type { Expr, Module, ValueDef } from "../ir/index.js";
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
      // GPU after row-major upload).
      const result = e.lhs.type.kind === "Matrix"
        ? { kind: "Vector" as const, element: e.lhs.type.element, dim: e.lhs.type.cols }
        : e.type;
      return { kind: "MulVecMat", lhs: e.rhs, rhs: e.lhs, type: result };
    }
    case "MulVecMat": {
      // Symmetric of the above (rare in user code, common after a
      // chain swap).
      const result = e.rhs.type.kind === "Matrix"
        ? { kind: "Vector" as const, element: e.rhs.type.element, dim: e.rhs.type.rows }
        : e.type;
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
    case "Transpose": {
      // Row-major `M.transpose()` matches the GPU's existing view of
      // M (which is already the row-major transpose under column-
      // major interpretation). The transpose vanishes.
      return e.value;
    }
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
