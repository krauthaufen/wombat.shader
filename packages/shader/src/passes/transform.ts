// Tree-mapping helpers: rebuild an Expr / LExpr / RExpr / Stmt by mapping
// each child through a transform function. Every pass is built on these.
//
// Returning the same reference signals "no change" so passes can short-
// circuit; we use referential equality (===) on each child to detect that.

import type {
  Expr,
  LExpr,
  RExpr,
  Stmt,
} from "../ir/index.js";

export type ExprMapper = (e: Expr) => Expr;
export type LExprMapper = (l: LExpr) => LExpr;
export type StmtMapper = (s: Stmt) => Stmt;

// ─────────────────────────────────────────────────────────────────────
// Expression children
// ─────────────────────────────────────────────────────────────────────

export function mapExprChildren(e: Expr, fn: ExprMapper): Expr {
  switch (e.kind) {
    case "Var":
    case "Const":
      return e;
    case "ReadInput": {
      if (!e.index) return e;
      const i = fn(e.index);
      return i === e.index ? e : { ...e, index: i };
    }
    case "Call": {
      const args = mapArray(e.args, fn);
      return args === e.args ? e : { ...e, args };
    }
    case "CallIntrinsic": {
      const args = mapArray(e.args, fn);
      return args === e.args ? e : { ...e, args };
    }
    case "Conditional": {
      const cond = fn(e.cond), ifTrue = fn(e.ifTrue), ifFalse = fn(e.ifFalse);
      return cond === e.cond && ifTrue === e.ifTrue && ifFalse === e.ifFalse
        ? e : { ...e, cond, ifTrue, ifFalse };
    }
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
    case "Convert": {
      const value = fn(e.value);
      return value === e.value ? e : { ...e, value };
    }
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
    case "VecNeq": {
      const lhs = fn(e.lhs), rhs = fn(e.rhs);
      return lhs === e.lhs && rhs === e.rhs ? e : { ...e, lhs, rhs };
    }
    case "VecSwizzle": {
      const value = fn(e.value);
      return value === e.value ? e : { ...e, value };
    }
    case "VecItem": {
      const value = fn(e.value), index = fn(e.index);
      return value === e.value && index === e.index ? e : { ...e, value, index };
    }
    case "MatrixElement": {
      const matrix = fn(e.matrix), row = fn(e.row), col = fn(e.col);
      return matrix === e.matrix && row === e.row && col === e.col
        ? e : { ...e, matrix, row, col };
    }
    case "MatrixRow": {
      const matrix = fn(e.matrix), row = fn(e.row);
      return matrix === e.matrix && row === e.row ? e : { ...e, matrix, row };
    }
    case "MatrixCol": {
      const matrix = fn(e.matrix), col = fn(e.col);
      return matrix === e.matrix && col === e.col ? e : { ...e, matrix, col };
    }
    case "NewVector": {
      const components = mapArray(e.components, fn);
      return components === e.components ? e : { ...e, components };
    }
    case "NewMatrix": {
      const elements = mapArray(e.elements, fn);
      return elements === e.elements ? e : { ...e, elements };
    }
    case "MatrixFromRows": {
      const rows = mapArray(e.rows, fn);
      return rows === e.rows ? e : { ...e, rows };
    }
    case "MatrixFromCols": {
      const cols = mapArray(e.cols, fn);
      return cols === e.cols ? e : { ...e, cols };
    }
    case "Field": {
      const target = fn(e.target);
      return target === e.target ? e : { ...e, target };
    }
    case "Item": {
      const target = fn(e.target), index = fn(e.index);
      return target === e.target && index === e.index ? e : { ...e, target, index };
    }
    case "DebugPrintf": {
      const format = fn(e.format), args = mapArray(e.args, fn);
      return format === e.format && args === e.args ? e : { ...e, format, args };
    }
  }
}

function mapArray<T>(arr: readonly T[], fn: (t: T) => T): readonly T[] {
  let changed = false;
  const out: T[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const before = arr[i]!;
    const after = fn(before);
    out[i] = after;
    if (after !== before) changed = true;
  }
  return changed ? out : arr;
}

// ─────────────────────────────────────────────────────────────────────
// L-expression / R-expression children
// ─────────────────────────────────────────────────────────────────────

export function mapLExprChildren(
  l: LExpr,
  onLExpr: LExprMapper,
  onExpr: ExprMapper,
): LExpr {
  switch (l.kind) {
    case "LVar":
      return l;
    case "LField": {
      const target = onLExpr(l.target);
      return target === l.target ? l : { ...l, target };
    }
    case "LItem": {
      const target = onLExpr(l.target), index = onExpr(l.index);
      return target === l.target && index === l.index ? l : { ...l, target, index };
    }
    case "LSwizzle": {
      const target = onLExpr(l.target);
      return target === l.target ? l : { ...l, target };
    }
    case "LMatrixElement": {
      const matrix = onLExpr(l.matrix), row = onExpr(l.row), col = onExpr(l.col);
      return matrix === l.matrix && row === l.row && col === l.col
        ? l : { ...l, matrix, row, col };
    }
    case "LInput": {
      if (!l.index) return l;
      const i = onExpr(l.index);
      return i === l.index ? l : { ...l, index: i };
    }
  }
}

export function mapRExpr(r: RExpr, fn: ExprMapper): RExpr {
  if (r.kind === "Expr") {
    const value = fn(r.value);
    return value === r.value ? r : { ...r, value };
  }
  const values = mapArray(r.values, fn);
  return values === r.values ? r : { ...r, values };
}

// ─────────────────────────────────────────────────────────────────────
// Statement children
// ─────────────────────────────────────────────────────────────────────

export interface StmtChildMapper {
  /** Map every Expr in the immediate subtree of the Stmt. */
  expr?(e: Expr): Expr;
  /** Map every LExpr in the immediate subtree. */
  lexpr?(l: LExpr): LExpr;
  /** Map every nested Stmt. */
  stmt?(s: Stmt): Stmt;
}

export function mapStmtChildren(s: Stmt, m: StmtChildMapper): Stmt {
  const fe: ExprMapper = m.expr ?? ((e) => e);
  const fl: LExprMapper = m.lexpr ?? ((l) => l);
  const fs: StmtMapper = m.stmt ?? ((x) => x);

  switch (s.kind) {
    case "Nop":
    case "Return":
    case "Break":
    case "Continue":
    case "Discard":
    case "Barrier":
      return s;
    case "Expression": {
      const value = fe(s.value);
      return value === s.value ? s : { ...s, value };
    }
    case "Declare": {
      if (!s.init) return s;
      const init = mapRExpr(s.init, fe);
      return init === s.init ? s : { ...s, init };
    }
    case "Write": {
      const target = fl(s.target), value = fe(s.value);
      return target === s.target && value === s.value ? s : { ...s, target, value };
    }
    case "WriteOutput": {
      const idx = s.index ? fe(s.index) : undefined;
      const value = mapRExpr(s.value, fe);
      const indexChanged = idx !== s.index;
      const valueChanged = value !== s.value;
      if (!indexChanged && !valueChanged) return s;
      return { ...s, ...(idx !== undefined ? { index: idx } : {}), value };
    }
    case "Increment":
    case "Decrement": {
      const target = fl(s.target);
      return target === s.target ? s : { ...s, target };
    }
    case "Sequential":
    case "Isolated": {
      const body = mapArray(s.body, fs);
      return body === s.body ? s : { ...s, body };
    }
    case "ReturnValue": {
      const value = fe(s.value);
      return value === s.value ? s : { ...s, value };
    }
    case "If": {
      const cond = fe(s.cond), then_ = fs(s.then);
      const else_ = s.else ? fs(s.else) : undefined;
      if (cond === s.cond && then_ === s.then && else_ === s.else) return s;
      return { ...s, cond, then: then_, ...(else_ !== undefined ? { else: else_ } : {}) };
    }
    case "For": {
      const init = fs(s.init), cond = fe(s.cond), step = fs(s.step), body = fs(s.body);
      return init === s.init && cond === s.cond && step === s.step && body === s.body
        ? s : { ...s, init, cond, step, body };
    }
    case "While":
    case "DoWhile": {
      const cond = fe(s.cond), body = fs(s.body);
      return cond === s.cond && body === s.body ? s : { ...s, cond, body };
    }
    case "Switch": {
      const value = fe(s.value);
      let casesChanged = false;
      const cases = s.cases.map((c) => {
        const body = fs(c.body);
        if (body === c.body) return c;
        casesChanged = true;
        return { ...c, body };
      });
      const defaultStmt = s.default ? fs(s.default) : undefined;
      if (value === s.value && !casesChanged && defaultStmt === s.default) return s;
      return {
        ...s,
        value,
        cases,
        ...(defaultStmt !== undefined ? { default: defaultStmt } : {}),
      };
    }
  }
}

/** Recursive Expr map (post-order). */
export function mapExpr(e: Expr, fn: ExprMapper): Expr {
  const mapped = mapExprChildren(e, (c) => mapExpr(c, fn));
  return fn(mapped);
}

/** Recursive LExpr map. */
export function mapLExpr(l: LExpr, fnL: LExprMapper, fnE: ExprMapper): LExpr {
  const mapped = mapLExprChildren(
    l,
    (c) => mapLExpr(c, fnL, fnE),
    (c) => mapExpr(c, fnE),
  );
  return fnL(mapped);
}

/** Recursive Stmt map. Recurses into every nested Stmt. */
export function mapStmt(s: Stmt, m: StmtChildMapper): Stmt {
  const child: StmtChildMapper = {
    stmt: (c) => {
      const mapped = mapStmt(c, m);
      return m.stmt ? m.stmt(mapped) : mapped;
    },
  };
  if (m.expr) child.expr = (e) => mapExpr(e, m.expr!);
  if (m.lexpr) child.lexpr = (l) => mapLExpr(l, m.lexpr!, m.expr ?? ((e) => e));
  else if (m.expr) child.lexpr = (l) => mapLExpr(l, (x) => x, m.expr!);
  return mapStmtChildren(s, child);
}
