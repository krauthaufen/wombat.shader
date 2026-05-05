// Minimal visitor utilities. Enough for the emitters; passes will extend.

import type { Expr, LExpr, RExpr, Stmt } from "./types.js";

export interface ExprVisitor {
  pre?(e: Expr): void;
  post?(e: Expr): void;
}

export interface StmtVisitor {
  preStmt?(s: Stmt): void;
  postStmt?(s: Stmt): void;
  expr?: ExprVisitor;
}

/** Walk every Expr child of an Expr node, in syntactic order. */
export function visitExprChildren(e: Expr, fn: (child: Expr) => void): void {
  switch (e.kind) {
    case "Var":
    case "Const":
      return;
    case "ReadInput":
      if (e.index) fn(e.index);
      return;
    case "Call":
    case "CallIntrinsic":
      for (const a of e.args) fn(a);
      return;
    case "Conditional":
      fn(e.cond); fn(e.ifTrue); fn(e.ifFalse);
      return;
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
    case "Convert":
      fn(e.value);
      return;
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
    case "VecNeq":
      fn(e.lhs); fn(e.rhs);
      return;
    case "VecSwizzle":
      fn(e.value);
      return;
    case "VecItem":
      fn(e.value); fn(e.index);
      return;
    case "MatrixElement":
      fn(e.matrix); fn(e.row); fn(e.col);
      return;
    case "MatrixRow":
      fn(e.matrix); fn(e.row);
      return;
    case "MatrixCol":
      fn(e.matrix); fn(e.col);
      return;
    case "NewVector":
      for (const c of e.components) fn(c);
      return;
    case "NewMatrix":
      for (const c of e.elements) fn(c);
      return;
    case "MatrixFromRows":
      for (const r of e.rows) fn(r);
      return;
    case "MatrixFromCols":
      for (const c of e.cols) fn(c);
      return;
    case "Field":
      fn(e.target);
      return;
    case "Item":
      fn(e.target); fn(e.index);
      return;
    case "DebugPrintf":
      fn(e.format); for (const a of e.args) fn(a);
      return;
  }
}

/** Walk every LExpr / Expr child of an LExpr node. */
export function visitLExprChildren(
  l: LExpr,
  onLExpr: (child: LExpr) => void,
  onExpr: (child: Expr) => void,
): void {
  switch (l.kind) {
    case "LVar":
      return;
    case "LField":
      onLExpr(l.target); return;
    case "LItem":
      onLExpr(l.target); onExpr(l.index); return;
    case "LSwizzle":
      onLExpr(l.target); return;
    case "LMatrixElement":
      onLExpr(l.matrix); onExpr(l.row); onExpr(l.col); return;
    case "LInput":
      if (l.index) onExpr(l.index); return;
  }
}

/** Walk every Stmt and Expr in a Stmt subtree (depth-first, pre+post). */
export function visitStmt(s: Stmt, v: StmtVisitor): void {
  v.preStmt?.(s);
  switch (s.kind) {
    case "Nop":
    case "Return":
    case "Break":
    case "Continue":
    case "Discard":
    case "Barrier":
      break;
    case "Expression":
      visitExprDeep(s.value, v.expr);
      break;
    case "Declare":
      if (s.init) visitRExprDeep(s.init, v.expr);
      break;
    case "Write":
      visitLExprDeep(s.target, v.expr);
      visitExprDeep(s.value, v.expr);
      break;
    case "WriteOutput":
      if (s.index) visitExprDeep(s.index, v.expr);
      visitRExprDeep(s.value, v.expr);
      break;
    case "Increment":
    case "Decrement":
      visitLExprDeep(s.target, v.expr);
      break;
    case "Sequential":
    case "Isolated":
      for (const c of s.body) visitStmt(c, v);
      break;
    case "ReturnValue":
      visitExprDeep(s.value, v.expr);
      break;
    case "If":
      visitExprDeep(s.cond, v.expr);
      visitStmt(s.then, v);
      if (s.else) visitStmt(s.else, v);
      break;
    case "For":
      visitStmt(s.init, v);
      visitExprDeep(s.cond, v.expr);
      visitStmt(s.step, v);
      visitStmt(s.body, v);
      break;
    case "While":
    case "DoWhile":
      visitExprDeep(s.cond, v.expr);
      visitStmt(s.body, v);
      break;
    case "Loop":
      visitStmt(s.body, v);
      break;
    case "Switch":
      visitExprDeep(s.value, v.expr);
      for (const c of s.cases) visitStmt(c.body, v);
      if (s.default) visitStmt(s.default, v);
      break;
  }
  v.postStmt?.(s);
}

function visitExprDeep(e: Expr, v?: ExprVisitor): void {
  if (!v) return;
  v.pre?.(e);
  visitExprChildren(e, (c) => visitExprDeep(c, v));
  v.post?.(e);
}

function visitLExprDeep(l: LExpr, v?: ExprVisitor): void {
  if (!v) return;
  visitLExprChildren(l, (c) => visitLExprDeep(c, v), (c) => visitExprDeep(c, v));
}

function visitRExprDeep(r: RExpr, v?: ExprVisitor): void {
  if (!v) return;
  if (r.kind === "Expr") visitExprDeep(r.value, v);
  else for (const e of r.values) visitExprDeep(e, v);
}
