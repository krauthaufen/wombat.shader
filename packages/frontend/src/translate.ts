// TypeScript AST → IR translator. Walks an arrow function (or function
// declaration) body and produces an IR Stmt; walks expression nodes
// and produces IR Exprs.
//
// Conventions:
//
//   - Variables (`let x = ...`, `const x = ...`) become IR `Var` with
//     `mutable: true|false`.
//   - Method calls `a.add(b)` translate to corresponding IR ops
//     (Add/Sub/Mul/Div/Neg/Dot/Cross/Length/Normalize/…). The frontend
//     doesn't try to verify that `a`'s type is a registered shader
//     vector — it trusts the user's source. For unknown methods the
//     translator emits a `Call` with a synthetic `FunctionRef`.
//   - Free function calls (e.g. `sin(x)`, `vec3(a, b, c)`) translate
//     to either `CallIntrinsic` (intrinsic table), `NewVector` /
//     `NewMatrix` / `MatrixFromCols` (vec*/mat* constructors), or
//     a regular `Call` (assumed user fn).
//   - Property access `v.xyz` on a vector type (recognised by name)
//     becomes `VecSwizzle`. Other property access is `Field`.
//   - The element type of a TS `number` literal in the source can't
//     be distinguished without the type checker; we default to `f32`.
//     Annotations like `: i32` recover precision when needed.
//
// What's intentionally out of scope at this stage:
//
//   - Inferring types from arbitrary expressions. We do best-effort
//     propagation; unknown types fall back to `f32` or a `Void` stub.
//   - Generic functions. Each user function gets one signature.
//   - Operator overloading via `+` / `*` / etc. — methods only.

import ts from "typescript";
import type {
  Expr,
  ExprBody,
  LExpr,
  RExpr,
  Stmt,
  Type,
  Var,
  VecComp,
} from "@aardworx/wombat.shader-ir";
import { lookupIntrinsic } from "./intrinsics.js";
import { constructorTargetType, tryResolveTypeName, vectorDimOf } from "./types.js";

const Tvoid: Type = { kind: "Void" };
const Tbool: Type = { kind: "Bool" };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };
const Tf32: Type = { kind: "Float", width: 32 };

export interface TranslateOptions {
  readonly source: string;
  readonly file?: string;
}

export interface TranslatedFunction {
  readonly body: Stmt;
  readonly parameters: ReadonlyArray<{ name: string; type: Type }>;
  readonly returnType: Type;
}

export interface Diagnostic {
  readonly message: string;
  readonly file?: string;
  readonly start?: number;
  readonly end?: number;
}

export class TranslationError extends Error {
  constructor(message: string, readonly diagnostics: readonly Diagnostic[]) {
    super(message);
    this.name = "TranslationError";
  }
}

interface Ctx {
  readonly file: string;
  readonly source: ts.SourceFile;
  /** name → Var, used to map identifier reads to interned IR Vars. */
  readonly vars: Map<string, Var>;
  /**
   * Parameters whose TS type was an object-type-literal (e.g.
   * `input: { v_color: V3f }`) — treated as input-record placeholders.
   * Property accesses through them translate to `ReadInput("Input")`.
   * Maps the parameter name to a map of field name → IR type.
   */
  readonly inputRecords: Map<string, Map<string, Type>>;
  readonly diagnostics: Diagnostic[];
}

// ─────────────────────────────────────────────────────────────────────
// Public entry: translate an arrow function or function declaration
// ─────────────────────────────────────────────────────────────────────

/** Find a top-level function by name, translate it, and return the IR. */
export function translateFunction(
  options: TranslateOptions,
  fnName: string,
): TranslatedFunction {
  const file = options.file ?? "<input>";
  const source = ts.createSourceFile(file, options.source, ts.ScriptTarget.ES2022, true);
  const fn = findFunction(source, fnName);
  if (!fn) throw new TranslationError(`function "${fnName}" not found in ${file}`, []);
  const ctx: Ctx = { file, source, vars: new Map(), inputRecords: new Map(), diagnostics: [] };

  const params = (fn.parameters ?? []).map((p) => extractParameter(p, ctx));
  const body = fn.body ? translateBlock(fn.body, ctx) : ({ kind: "Nop" } as Stmt);
  const returnType = fn.type ? typeFromNode(fn.type, ctx) : Tvoid;

  if (ctx.diagnostics.length > 0) {
    throw new TranslationError(
      `${ctx.diagnostics.length} translation diagnostic(s) for "${fnName}"`,
      ctx.diagnostics,
    );
  }
  return { body, parameters: params, returnType };
}

interface FunctionLike {
  readonly parameters: readonly ts.ParameterDeclaration[];
  readonly body: ts.Block | undefined;
  readonly type: ts.TypeNode | undefined;
}

function findFunction(source: ts.SourceFile, name: string): FunctionLike | undefined {
  let result: FunctionLike | undefined;
  ts.forEachChild(source, (node) => {
    if (result) return;
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) {
      result = {
        parameters: node.parameters,
        body: node.body,
        type: node.type,
      };
      return;
    }
    if (ts.isVariableStatement(node)) {
      for (const decl of node.declarationList.declarations) {
        if (!ts.isIdentifier(decl.name) || decl.name.text !== name) continue;
        const init = decl.initializer;
        if (init && ts.isArrowFunction(init)) {
          // Arrow functions can have a concise body (Expression).
          const block = ts.isBlock(init.body)
            ? init.body
            : ts.factory.createBlock([ts.factory.createReturnStatement(init.body)], true);
          result = { parameters: init.parameters, body: block, type: init.type };
          return;
        }
        if (init && ts.isFunctionExpression(init)) {
          result = { parameters: init.parameters, body: init.body, type: init.type };
          return;
        }
      }
    }
  });
  return result;
}

function extractParameter(p: ts.ParameterDeclaration, ctx: Ctx): { name: string; type: Type } {
  if (!ts.isIdentifier(p.name)) {
    addDiagnostic(ctx, p, "frontend: parameter must be a simple identifier (no destructuring)");
    return { name: "_anon", type: Tvoid };
  }
  // Object-type-literal parameter: treat as an input record. Each
  // member becomes a separately resolvable input via property access.
  if (p.type && ts.isTypeLiteralNode(p.type)) {
    const fields = new Map<string, Type>();
    for (const member of p.type.members) {
      if (!ts.isPropertySignature(member)) continue;
      if (!member.name || !ts.isIdentifier(member.name)) continue;
      const t = member.type ? typeFromNode(member.type, ctx) : Tf32;
      fields.set(member.name.text, t);
    }
    ctx.inputRecords.set(p.name.text, fields);
    // Register a placeholder Var so identifier resolution doesn't bail
    // out, but its type is the synthetic struct shape (the body never
    // reads this Var directly — only via property access we intercept).
    const placeholder: Var = { name: p.name.text, type: Tvoid, mutable: false };
    ctx.vars.set(p.name.text, placeholder);
    return { name: p.name.text, type: Tvoid };
  }
  const type = p.type ? typeFromNode(p.type, ctx) : Tvoid;
  const v: Var = { name: p.name.text, type, mutable: false };
  ctx.vars.set(p.name.text, v);
  return { name: p.name.text, type };
}

// ─────────────────────────────────────────────────────────────────────
// Type nodes
// ─────────────────────────────────────────────────────────────────────

function typeFromNode(node: ts.TypeNode, ctx: Ctx): Type {
  if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
    const name = node.typeName.text;
    const t = tryResolveTypeName(name);
    if (t) return t;
    addDiagnostic(ctx, node, `frontend: unknown type reference "${name}"`);
    return Tvoid;
  }
  if (node.kind === ts.SyntaxKind.NumberKeyword) return Tf32;
  if (node.kind === ts.SyntaxKind.BooleanKeyword) return Tbool;
  if (node.kind === ts.SyntaxKind.VoidKeyword) return Tvoid;
  addDiagnostic(ctx, node, `frontend: unsupported type syntax (${ts.SyntaxKind[node.kind]})`);
  return Tvoid;
}

// ─────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────

function translateBlock(block: ts.Block, ctx: Ctx): Stmt {
  const stmts: Stmt[] = [];
  for (const child of block.statements) stmts.push(translateStmt(child, ctx));
  if (stmts.length === 0) return { kind: "Nop" };
  if (stmts.length === 1) return stmts[0]!;
  return { kind: "Sequential", body: stmts };
}

function translateStmt(node: ts.Statement, ctx: Ctx): Stmt {
  if (ts.isVariableStatement(node)) return translateVarDecl(node, ctx);
  if (ts.isExpressionStatement(node)) {
    const e = translateAssignmentLikeExpression(node.expression, ctx);
    if (e) return e; // statement-yielding form (e.g. assignment)
    return { kind: "Expression", value: translateExpr(node.expression, ctx) };
  }
  if (ts.isReturnStatement(node)) {
    if (!node.expression) return { kind: "Return" };
    return { kind: "ReturnValue", value: translateExpr(node.expression, ctx) };
  }
  if (ts.isIfStatement(node)) return translateIf(node, ctx);
  if (ts.isForStatement(node)) return translateFor(node, ctx);
  if (ts.isWhileStatement(node)) {
    return {
      kind: "While",
      cond: translateExpr(node.expression, ctx),
      body: translateStmt(node.statement, ctx),
    };
  }
  if (ts.isDoStatement(node)) {
    return {
      kind: "DoWhile",
      cond: translateExpr(node.expression, ctx),
      body: translateStmt(node.statement, ctx),
    };
  }
  if (ts.isBlock(node)) return translateBlock(node, ctx);
  if (node.kind === ts.SyntaxKind.BreakStatement) return { kind: "Break" };
  if (node.kind === ts.SyntaxKind.ContinueStatement) return { kind: "Continue" };
  if (node.kind === ts.SyntaxKind.EmptyStatement) return { kind: "Nop" };

  addDiagnostic(ctx, node, `frontend: unsupported statement (${ts.SyntaxKind[node.kind]})`);
  return { kind: "Nop" };
}

function translateVarDecl(node: ts.VariableStatement, ctx: Ctx): Stmt {
  const isConst = (node.declarationList.flags & ts.NodeFlags.Const) !== 0;
  const stmts: Stmt[] = [];
  for (const decl of node.declarationList.declarations) {
    if (!ts.isIdentifier(decl.name)) {
      addDiagnostic(ctx, decl, "frontend: declaration must use a simple identifier");
      continue;
    }
    const name = decl.name.text;
    const initExpr = decl.initializer ? translateExpr(decl.initializer, ctx) : undefined;
    const type = decl.type
      ? typeFromNode(decl.type, ctx)
      : initExpr?.type ?? Tvoid;
    const v: Var = { name, type, mutable: !isConst };
    ctx.vars.set(name, v);
    stmts.push({
      kind: "Declare", var: v,
      ...(initExpr ? { init: { kind: "Expr" as const, value: initExpr } satisfies RExpr } : {}),
    });
  }
  if (stmts.length === 0) return { kind: "Nop" };
  if (stmts.length === 1) return stmts[0]!;
  return { kind: "Sequential", body: stmts };
}

function translateIf(node: ts.IfStatement, ctx: Ctx): Stmt {
  return {
    kind: "If",
    cond: translateExpr(node.expression, ctx),
    then: translateStmt(node.thenStatement, ctx),
    ...(node.elseStatement ? { else: translateStmt(node.elseStatement, ctx) } : {}),
  };
}

function translateFor(node: ts.ForStatement, ctx: Ctx): Stmt {
  const init: Stmt = node.initializer
    ? ts.isVariableDeclarationList(node.initializer)
      ? translateForInitVar(node.initializer, ctx)
      : { kind: "Expression", value: translateExpr(node.initializer as ts.Expression, ctx) }
    : { kind: "Nop" };
  const cond: Expr = node.condition
    ? translateExpr(node.condition, ctx)
    : { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool };
  const step: Stmt = node.incrementor
    ? translateAssignmentLikeExpression(node.incrementor, ctx)
      ?? { kind: "Expression", value: translateExpr(node.incrementor, ctx) }
    : { kind: "Nop" };
  return {
    kind: "For",
    init, cond, step,
    body: translateStmt(node.statement, ctx),
  };
}

function translateForInitVar(decls: ts.VariableDeclarationList, ctx: Ctx): Stmt {
  const stmt: ts.VariableStatement = {
    declarationList: decls,
    modifiers: undefined,
    kind: ts.SyntaxKind.VariableStatement,
  } as unknown as ts.VariableStatement;
  // We synthesise a VariableStatement to reuse translateVarDecl. The
  // node-level fields above are enough for our walker — we never read
  // anything outside `declarationList`.
  return translateVarDecl(stmt, ctx);
}

/** Translate an expression that may be (or contain at top level) an
 * assignment-like form, returning the corresponding Stmt. Returns
 * undefined when the expression is a value-producing form. */
function translateAssignmentLikeExpression(node: ts.Expression, ctx: Ctx): Stmt | undefined {
  if (ts.isBinaryExpression(node)) {
    const op = node.operatorToken.kind;
    if (op === ts.SyntaxKind.EqualsToken) {
      return { kind: "Write", target: translateLExpr(node.left, ctx), value: translateExpr(node.right, ctx) };
    }
    const compound = COMPOUND_OPS[op as keyof typeof COMPOUND_OPS];
    if (compound) {
      const target = translateLExpr(node.left, ctx);
      const lhsExpr = lExprToExpr(target);
      const rhs = translateExpr(node.right, ctx);
      const combined: Expr = { kind: compound, lhs: lhsExpr, rhs, type: lhsExpr.type };
      return { kind: "Write", target, value: combined };
    }
  }
  if (ts.isPostfixUnaryExpression(node)) {
    if (node.operator === ts.SyntaxKind.PlusPlusToken) {
      return { kind: "Increment", target: translateLExpr(node.operand, ctx), prefix: false };
    }
    if (node.operator === ts.SyntaxKind.MinusMinusToken) {
      return { kind: "Decrement", target: translateLExpr(node.operand, ctx), prefix: false };
    }
  }
  if (ts.isPrefixUnaryExpression(node)) {
    if (node.operator === ts.SyntaxKind.PlusPlusToken) {
      return { kind: "Increment", target: translateLExpr(node.operand, ctx), prefix: true };
    }
    if (node.operator === ts.SyntaxKind.MinusMinusToken) {
      return { kind: "Decrement", target: translateLExpr(node.operand, ctx), prefix: true };
    }
  }
  return undefined;
}

const COMPOUND_OPS: Record<number, "Add" | "Sub" | "Mul" | "Div" | "Mod"> = {
  [ts.SyntaxKind.PlusEqualsToken]: "Add",
  [ts.SyntaxKind.MinusEqualsToken]: "Sub",
  [ts.SyntaxKind.AsteriskEqualsToken]: "Mul",
  [ts.SyntaxKind.SlashEqualsToken]: "Div",
  [ts.SyntaxKind.PercentEqualsToken]: "Mod",
};

// ─────────────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────────────

function translateExpr(node: ts.Expression, ctx: Ctx): Expr {
  if (ts.isParenthesizedExpression(node)) return translateExpr(node.expression, ctx);
  if (ts.isIdentifier(node)) return translateIdentifier(node, ctx);
  if (ts.isNumericLiteral(node) || node.kind === ts.SyntaxKind.NumericLiteral) {
    const text = node.getText(ctx.source);
    const isInt = !/[.eE]/.test(text);
    if (isInt) return { kind: "Const", value: { kind: "Int", signed: true, value: parseInt(text, 10) }, type: Ti32 };
    return { kind: "Const", value: { kind: "Float", value: parseFloat(text) }, type: Tf32 };
  }
  if (node.kind === ts.SyntaxKind.TrueKeyword) return { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool };
  if (node.kind === ts.SyntaxKind.FalseKeyword) return { kind: "Const", value: { kind: "Bool", value: false }, type: Tbool };
  if (ts.isCallExpression(node)) return translateCall(node, ctx);
  if (ts.isPropertyAccessExpression(node)) return translateProperty(node, ctx);
  if (ts.isElementAccessExpression(node)) return translateElement(node, ctx);
  if (ts.isBinaryExpression(node)) return translateBinary(node, ctx);
  if (ts.isPrefixUnaryExpression(node)) return translatePrefixUnary(node, ctx);
  if (ts.isConditionalExpression(node)) {
    return {
      kind: "Conditional",
      cond: translateExpr(node.condition, ctx),
      ifTrue: translateExpr(node.whenTrue, ctx),
      ifFalse: translateExpr(node.whenFalse, ctx),
      type: translateExpr(node.whenTrue, ctx).type,
    };
  }
  if (ts.isNewExpression(node)) return translateNew(node, ctx);
  if (ts.isAsExpression(node)) {
    // `expr as T` — adopt the asserted type but keep the value.
    const inner = translateExpr(node.expression, ctx);
    const type = typeFromNode(node.type, ctx);
    return { ...inner, type } as Expr;
  }

  addDiagnostic(ctx, node, `frontend: unsupported expression (${ts.SyntaxKind[node.kind]})`);
  return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
}

function translateIdentifier(node: ts.Identifier, ctx: Ctx): Expr {
  const v = ctx.vars.get(node.text);
  if (v) return { kind: "Var", var: v, type: v.type };
  // Unresolved identifier — treat as ReadInput("Uniform"); a smarter
  // pass would distinguish Input vs Uniform based on shipped builtin
  // tables. For now uniforms are the common case for free names.
  return { kind: "ReadInput", scope: "Uniform", name: node.text, type: Tf32 };
}

function translateBinary(node: ts.BinaryExpression, ctx: Ctx): Expr {
  const lhs = translateExpr(node.left, ctx);
  const rhs = translateExpr(node.right, ctx);
  const k = BIN_OP_KIND[node.operatorToken.kind];
  if (k) {
    const type = ARITH_OPS.has(k) ? lhs.type : Tbool;
    return { kind: k, lhs, rhs, type } as Expr;
  }
  addDiagnostic(ctx, node, `frontend: unsupported binary operator (${ts.SyntaxKind[node.operatorToken.kind]})`);
  return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
}

const BIN_OP_KIND: Record<number, ExprBody["kind"] | undefined> = {
  [ts.SyntaxKind.PlusToken]: "Add",
  [ts.SyntaxKind.MinusToken]: "Sub",
  [ts.SyntaxKind.AsteriskToken]: "Mul",
  [ts.SyntaxKind.SlashToken]: "Div",
  [ts.SyntaxKind.PercentToken]: "Mod",
  [ts.SyntaxKind.AmpersandAmpersandToken]: "And",
  [ts.SyntaxKind.BarBarToken]: "Or",
  [ts.SyntaxKind.AmpersandToken]: "BitAnd",
  [ts.SyntaxKind.BarToken]: "BitOr",
  [ts.SyntaxKind.CaretToken]: "BitXor",
  [ts.SyntaxKind.LessThanLessThanToken]: "ShiftLeft",
  [ts.SyntaxKind.GreaterThanGreaterThanToken]: "ShiftRight",
  [ts.SyntaxKind.EqualsEqualsToken]: "Eq",
  [ts.SyntaxKind.EqualsEqualsEqualsToken]: "Eq",
  [ts.SyntaxKind.ExclamationEqualsToken]: "Neq",
  [ts.SyntaxKind.ExclamationEqualsEqualsToken]: "Neq",
  [ts.SyntaxKind.LessThanToken]: "Lt",
  [ts.SyntaxKind.LessThanEqualsToken]: "Le",
  [ts.SyntaxKind.GreaterThanToken]: "Gt",
  [ts.SyntaxKind.GreaterThanEqualsToken]: "Ge",
};

const ARITH_OPS = new Set<string>([
  "Add", "Sub", "Mul", "Div", "Mod",
  "BitAnd", "BitOr", "BitXor", "ShiftLeft", "ShiftRight",
]);

function translatePrefixUnary(node: ts.PrefixUnaryExpression, ctx: Ctx): Expr {
  const value = translateExpr(node.operand, ctx);
  switch (node.operator) {
    case ts.SyntaxKind.MinusToken: return { kind: "Neg", value, type: value.type };
    case ts.SyntaxKind.ExclamationToken: return { kind: "Not", value, type: Tbool };
    case ts.SyntaxKind.TildeToken: return { kind: "BitNot", value, type: value.type };
    case ts.SyntaxKind.PlusToken: return value; // unary plus is a no-op
    default:
      addDiagnostic(ctx, node, `frontend: unsupported unary operator (${ts.SyntaxKind[node.operator]})`);
      return value;
  }
}

function translateCall(node: ts.CallExpression, ctx: Ctx): Expr {
  // Method call?
  if (ts.isPropertyAccessExpression(node.expression)) {
    const target = translateExpr(node.expression.expression, ctx);
    const method = node.expression.name.text;
    const args = node.arguments.map((a) => translateExpr(a, ctx));
    return translateMethodCall(target, method, args, node, ctx);
  }
  // Free-function call.
  if (ts.isIdentifier(node.expression)) {
    const name = node.expression.text;
    const args = node.arguments.map((a) => translateExpr(a, ctx));
    // 1. Vector / matrix constructor (vec3, mat4, …)?
    const ctorType = constructorTargetType(name);
    if (ctorType) {
      if (ctorType.kind === "Vector") return { kind: "NewVector", components: args, type: ctorType };
      if (ctorType.kind === "Matrix") return { kind: "MatrixFromCols", cols: args, type: ctorType };
    }
    // 2. Known intrinsic?
    const intr = lookupIntrinsic(name);
    if (intr) {
      const argTypes = args.map((a) => a.type);
      return { kind: "CallIntrinsic", op: intr, args, type: intr.returnTypeOf(argTypes) };
    }
    // 3. User function — synthesise a FunctionRef stub. The caller
    //    Module-level pass should match this against a user `Function`
    //    decl by id; for now we just record the call.
    addDiagnostic(ctx, node, `frontend: unknown function "${name}" — leaving as Call(stub)`);
    return {
      kind: "Call",
      fn: { id: name, signature: { name, returnType: Tvoid, parameters: [] }, pure: true },
      args,
      type: Tvoid,
    };
  }
  addDiagnostic(ctx, node, "frontend: unsupported call shape (callee must be Identifier or x.method)");
  return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
}

// ─────────────────────────────────────────────────────────────────────
// Method calls on shader vector / matrix types
// ─────────────────────────────────────────────────────────────────────

function translateMethodCall(
  target: Expr,
  method: string,
  args: readonly Expr[],
  node: ts.Node,
  ctx: Ctx,
): Expr {
  switch (method) {
    case "add": return mkBin("Add", target, args[0]!);
    case "sub": return mkBin("Sub", target, args[0]!);
    case "mul": return mkMul(target, args[0]!);
    case "div": return mkBin("Div", target, args[0]!);
    case "neg": return { kind: "Neg", value: target, type: target.type };
    case "dot": {
      const elementType = target.type.kind === "Vector" ? target.type.element : Tf32;
      return { kind: "Dot", lhs: target, rhs: args[0]!, type: elementType };
    }
    case "cross": return { kind: "Cross", lhs: target, rhs: args[0]!, type: target.type };
    case "length": return { kind: "Length", value: target, type: Tf32 };
    case "normalize": {
      const intr = lookupIntrinsic("normalize")!;
      return { kind: "CallIntrinsic", op: intr, args: [target], type: target.type };
    }
    case "transpose": {
      // transpose flips matrix dims.
      const t = target.type;
      const flipped: Type = t.kind === "Matrix"
        ? { kind: "Matrix", element: t.element, rows: t.cols, cols: t.rows }
        : t;
      return { kind: "Transpose", value: target, type: flipped };
    }
    case "inverse": return { kind: "Inverse", value: target, type: target.type };
    case "determinant": {
      const elementType = target.type.kind === "Matrix" ? target.type.element : Tf32;
      return { kind: "Determinant", value: target, type: elementType };
    }
    default:
      addDiagnostic(ctx, node, `frontend: unknown method ".${method}" on ${describeType(target.type)}`);
      return target;
  }
}

function mkBin(kind: "Add" | "Sub" | "Mul" | "Div" | "Mod", lhs: Expr, rhs: Expr): Expr {
  return { kind, lhs, rhs, type: lhs.type };
}

function mkMul(lhs: Expr, rhs: Expr): Expr {
  // Disambiguate matrix×vec vs vec×matrix vs matrix×matrix vs scalar.
  const lt = lhs.type, rt = rhs.type;
  if (lt.kind === "Matrix" && rt.kind === "Matrix") {
    const result: Type = { kind: "Matrix", element: lt.element, rows: lt.rows, cols: rt.cols };
    return { kind: "MulMatMat", lhs, rhs, type: result };
  }
  if (lt.kind === "Matrix" && rt.kind === "Vector") {
    const result: Type = { kind: "Vector", element: lt.element, dim: lt.rows };
    return { kind: "MulMatVec", lhs, rhs, type: result };
  }
  if (lt.kind === "Vector" && rt.kind === "Matrix") {
    const result: Type = { kind: "Vector", element: lt.element, dim: rt.cols };
    return { kind: "MulVecMat", lhs, rhs, type: result };
  }
  return { kind: "Mul", lhs, rhs, type: lt };
}

// ─────────────────────────────────────────────────────────────────────
// Property and element access
// ─────────────────────────────────────────────────────────────────────

function translateProperty(node: ts.PropertyAccessExpression, ctx: Ctx): Expr {
  // Input-record short-circuit: `inputRecord.field` → ReadInput("Input").
  if (ts.isIdentifier(node.expression)) {
    const fields = ctx.inputRecords.get(node.expression.text);
    if (fields) {
      const fieldName = node.name.text;
      const fieldType = fields.get(fieldName) ?? Tf32;
      return { kind: "ReadInput", scope: "Input", name: fieldName, type: fieldType };
    }
  }
  const target = translateExpr(node.expression, ctx);
  const propName = node.name.text;
  // Recognise swizzle (.x, .xy, .xyz, .rgba etc.)
  if (target.type.kind === "Vector") {
    const swiz = parseSwizzle(propName);
    if (swiz) {
      const dim = target.type.dim;
      if (swiz.every((c) => (SWIZ_INDEX[c] ?? 99) < dim)) {
        const newDim = swiz.length;
        const type: Type = newDim === 1
          ? target.type.element
          : { kind: "Vector", element: target.type.element, dim: newDim as 2 | 3 | 4 };
        if (newDim === 1) {
          return { kind: "VecSwizzle", value: target, comps: swiz, type };
        }
        return { kind: "VecSwizzle", value: target, comps: swiz, type };
      }
    }
  }
  return { kind: "Field", target, name: propName, type: Tf32 };
}

const SWIZ_INDEX: Record<string, number> = {
  x: 0, y: 1, z: 2, w: 3,
  r: 0, g: 1, b: 2, a: 3,
  s: 0, t: 1, p: 2, q: 3,
};

function parseSwizzle(name: string): VecComp[] | undefined {
  if (name.length < 1 || name.length > 4) return undefined;
  const out: VecComp[] = [];
  for (const c of name) {
    const idx = SWIZ_INDEX[c];
    if (idx === undefined) return undefined;
    out.push((["x", "y", "z", "w"] as const)[idx]!);
  }
  return out;
}

function translateElement(node: ts.ElementAccessExpression, ctx: Ctx): Expr {
  const target = translateExpr(node.expression, ctx);
  const index = translateExpr(node.argumentExpression, ctx);
  if (target.type.kind === "Vector") {
    return { kind: "VecItem", value: target, index, type: target.type.element };
  }
  if (target.type.kind === "Matrix") {
    // m[c] = column c (a vector). matches GLSL/WGSL semantics.
    return {
      kind: "MatrixCol", matrix: target, col: index,
      type: { kind: "Vector", element: target.type.element, dim: target.type.rows },
    };
  }
  return { kind: "Item", target, index, type: Tf32 };
}

// ─────────────────────────────────────────────────────────────────────
// L-expressions (assignment targets)
// ─────────────────────────────────────────────────────────────────────

function translateLExpr(node: ts.Expression, ctx: Ctx): LExpr {
  if (ts.isIdentifier(node)) {
    const v = ctx.vars.get(node.text);
    if (v) return { kind: "LVar", var: v, type: v.type };
    addDiagnostic(ctx, node, `frontend: writing to undeclared identifier "${node.text}"`);
    return { kind: "LVar", var: { name: node.text, type: Tvoid, mutable: true }, type: Tvoid };
  }
  if (ts.isPropertyAccessExpression(node)) {
    const target = translateLExpr(node.expression, ctx);
    const propName = node.name.text;
    if (target.type.kind === "Vector") {
      const swiz = parseSwizzle(propName);
      if (swiz) {
        const newDim = swiz.length;
        const type: Type = newDim === 1
          ? target.type.element
          : { kind: "Vector", element: target.type.element, dim: newDim as 2 | 3 | 4 };
        return { kind: "LSwizzle", target, comps: swiz, type };
      }
    }
    return { kind: "LField", target, name: propName, type: Tf32 };
  }
  if (ts.isElementAccessExpression(node)) {
    const target = translateLExpr(node.expression, ctx);
    const index = translateExpr(node.argumentExpression, ctx);
    return { kind: "LItem", target, index, type: Tf32 };
  }
  addDiagnostic(ctx, node, `frontend: unsupported assignment target (${ts.SyntaxKind[node.kind]})`);
  return { kind: "LVar", var: { name: "_err", type: Tvoid, mutable: true }, type: Tvoid };
}

function lExprToExpr(l: LExpr): Expr {
  // Used by compound assignment translation: `x += y` → `x = x + y`,
  // where the RHS uses an Expr form of `x`.
  switch (l.kind) {
    case "LVar": return { kind: "Var", var: l.var, type: l.var.type };
    case "LField": return { kind: "Field", target: lExprToExpr(l.target), name: l.name, type: l.type };
    case "LItem": return { kind: "Item", target: lExprToExpr(l.target), index: l.index, type: l.type };
    case "LSwizzle": return { kind: "VecSwizzle", value: lExprToExpr(l.target), comps: l.comps, type: l.type };
    case "LMatrixElement": {
      const m = lExprToExpr(l.matrix);
      return { kind: "MatrixElement", matrix: m, row: l.row, col: l.col, type: l.type };
    }
    case "LInput":
      return l.index !== undefined
        ? { kind: "ReadInput", scope: l.scope, name: l.name, index: l.index, type: l.type }
        : { kind: "ReadInput", scope: l.scope, name: l.name, type: l.type };
  }
}

// ─────────────────────────────────────────────────────────────────────
// `new V3f(...)` style constructions
// ─────────────────────────────────────────────────────────────────────

function translateNew(node: ts.NewExpression, ctx: Ctx): Expr {
  if (!ts.isIdentifier(node.expression)) {
    addDiagnostic(ctx, node, "frontend: `new ...` callee must be a simple identifier");
    return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
  }
  const name = node.expression.text;
  const ctorType = tryResolveTypeName(name);
  if (ctorType) {
    const args = (node.arguments ?? []).map((a) => translateExpr(a, ctx));
    if (ctorType.kind === "Vector") return { kind: "NewVector", components: args, type: ctorType };
    if (ctorType.kind === "Matrix") return { kind: "MatrixFromCols", cols: args, type: ctorType };
  }
  // Fall through: treat as a user struct constructor.
  void vectorDimOf; // referenced only via constructorTargetType; quiet unused-import
  addDiagnostic(ctx, node, `frontend: unknown constructor "new ${name}"`);
  return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
}

// ─────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────

function addDiagnostic(ctx: Ctx, node: ts.Node, message: string): void {
  ctx.diagnostics.push({
    message,
    file: ctx.file,
    start: node.getStart(ctx.source),
    end: node.getEnd(),
  });
}

function describeType(t: Type): string {
  switch (t.kind) {
    case "Vector": return `vec${t.dim}<${describeType(t.element)}>`;
    case "Matrix": return `mat${t.rows}x${t.cols}<${describeType(t.element)}>`;
    case "Float": return "f32";
    case "Int": return t.signed ? "i32" : "u32";
    case "Bool": return "bool";
    default: return t.kind;
  }
}
