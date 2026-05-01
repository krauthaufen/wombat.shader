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
} from "../ir/index.js";
import { lookupIntrinsic } from "./intrinsics.js";
import { constructorTargetType, tryResolveTypeName, vectorDimOf } from "./types.js";

const Tvoid: Type = { kind: "Void" };
const Tbool: Type = { kind: "Bool" };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };
const Tf32: Type = { kind: "Float", width: 32 };

export interface TranslateOptions {
  readonly source: string;
  readonly file?: string;
  /**
   * Types the translator should use when it encounters a free
   * identifier (one not declared as a parameter, let, or const).
   * Without this every unresolved name defaults to `f32`.
   */
  readonly externalTypes?: ReadonlyMap<string, Type>;
  /**
   * Closure-captured identifiers: name → IR type. Free identifiers
   * matching one of these names lower to `ReadInput("Closure", name)`
   * instead of `ReadInput("Uniform", name)`. The build plugin
   * populates this from the arrow function's lexical environment so
   * the runtime `resolveHoles` pass can substitute the captured
   * values at compile time.
   */
  readonly closureTypes?: ReadonlyMap<string, Type>;
  /**
   * Uniform namespaces: `nsName → { fieldName → IR type }`. Property
   * accesses through `nsName` lower directly to
   * `ReadInput("Uniform", fieldName, type)`. The build plugin
   * populates this from `declare const u: { … }` ambient declarations
   * so `u.tint` in the body emits a uniform read for `tint`.
   */
  readonly uniformNamespaces?: ReadonlyMap<string, ReadonlyMap<string, Type>>;
  /**
   * Pre-computed signatures for top-level helper functions in the
   * same source. When a body calls one of these by name, the
   * translator emits a properly-typed `Call(FunctionRef)` instead of
   * the unresolved `Call(stub)` path.
   */
  readonly helperSignatures?: ReadonlyMap<string, import("../ir/index.js").FunctionSignature>;
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
   * Top-level `const NAME = literal;` declarations. Lookups inline
   * the literal at the use site so the emitter never sees a dangling
   * identifier. Values are stored as fully-typed `Expr`s (Const nodes
   * for numeric/boolean literals).
   */
  readonly moduleConsts: Map<string, Expr>;
  /**
   * Parameters whose TS type was an object-type-literal (e.g.
   * `input: { v_color: V3f }`) — treated as input-record placeholders.
   * Property accesses through them translate to `ReadInput("Input")`.
   * Maps the parameter name to a map of field name → IR type.
   */
  readonly inputRecords: Map<string, Map<string, Type>>;
  /**
   * Parameters typed `ComputeBuiltins` (and similar). Property accesses
   * through them translate to `ReadInput("Builtin", <semantic>)`. The
   * map's value is the field-name → IR type; the parameter name is
   * never read in the body.
   */
  readonly builtinRecords: Map<string, Map<string, { type: Type; semantic: BuiltinName }>>;
  /** Types of free identifiers (uniforms / samplers / etc.). */
  readonly externalTypes: ReadonlyMap<string, Type>;
  /** Types of closure-captured identifiers (build-plugin-provided). */
  readonly closureTypes: ReadonlyMap<string, Type>;
  /** Uniform namespaces (build-plugin-provided). */
  readonly uniformNamespaces: ReadonlyMap<string, ReadonlyMap<string, Type>>;
  /** Pre-computed signatures for top-level helper functions. */
  readonly helperSignatures: ReadonlyMap<string, import("../ir/index.js").FunctionSignature>;
  readonly diagnostics: Diagnostic[];
}

// ─── Builtin parameter-shape recognition ────────────────────────────
//
// camelCase (TS) → snake_case (BuiltinSemantic in IR).

type BuiltinName =
  | "global_invocation_id"
  | "local_invocation_id"
  | "local_invocation_index"
  | "workgroup_id"
  | "num_workgroups"
  | "vertex_index"
  | "instance_index"
  | "front_facing"
  | "frag_depth"
  | "position";

const Tu32: Type = { kind: "Int", signed: false, width: 32 };
const Tvec3u: Type = { kind: "Vector", element: Tu32, dim: 3 };
const Tvec4f: Type = { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 };
const Tvec2f: Type = { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 2 };

const COMPUTE_BUILTINS: Record<string, { type: Type; semantic: BuiltinName }> = {
  globalInvocationId: { type: Tvec3u, semantic: "global_invocation_id" },
  localInvocationId:  { type: Tvec3u, semantic: "local_invocation_id" },
  localInvocationIndex: { type: Tu32, semantic: "local_invocation_index" },
  workgroupId:        { type: Tvec3u, semantic: "workgroup_id" },
  numWorkgroups:      { type: Tvec3u, semantic: "num_workgroups" },
};
const VERTEX_BUILTIN_IN: Record<string, { type: Type; semantic: BuiltinName }> = {
  vertexIndex:   { type: Tu32, semantic: "vertex_index" },
  instanceIndex: { type: Tu32, semantic: "instance_index" },
};
const FRAGMENT_BUILTIN_IN: Record<string, { type: Type; semantic: BuiltinName }> = {
  fragCoord:    { type: Tvec4f, semantic: "position" },
  frontFacing:  { type: { kind: "Bool" }, semantic: "front_facing" },
  pointCoord:   { type: Tvec2f, semantic: "position" },
};

const BUILTIN_PARAM_SHAPES: Record<string, Record<string, { type: Type; semantic: BuiltinName }>> = {
  ComputeBuiltins: COMPUTE_BUILTINS,
  VertexBuiltinIn: VERTEX_BUILTIN_IN,
  FragmentBuiltinIn: FRAGMENT_BUILTIN_IN,
};

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
  const ctx: Ctx = {
    file, source,
    vars: new Map(),
    moduleConsts: collectModuleConsts(source),
    inputRecords: new Map(),
    builtinRecords: new Map(),
    externalTypes: options.externalTypes ?? new Map(),
    closureTypes: options.closureTypes ?? new Map(),
    uniformNamespaces: options.uniformNamespaces ?? new Map(),
    helperSignatures: options.helperSignatures ?? new Map(),
    diagnostics: [],
  };

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
  // Builtin record (e.g. `b: ComputeBuiltins`). Property accesses
  // through it lower to `ReadInput("Builtin", <semantic>)`.
  if (p.type && ts.isTypeReferenceNode(p.type) && ts.isIdentifier(p.type.typeName)) {
    const shape = BUILTIN_PARAM_SHAPES[p.type.typeName.text];
    if (shape) {
      const fields = new Map<string, { type: Type; semantic: BuiltinName }>(Object.entries(shape));
      ctx.builtinRecords.set(p.name.text, fields);
      const placeholder: Var = { name: p.name.text, type: Tvoid, mutable: false };
      ctx.vars.set(p.name.text, placeholder);
      return { name: p.name.text, type: Tvoid };
    }
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
  // An object-type-literal as a return type is an "output record"
  // shape (see translateObjectLiteral / liftReturns). We don't model
  // it as a struct in the IR — the caller declares the actual outputs
  // in the EntryRequest, and liftReturns matches by name. Returning
  // Void here is harmless because the entry's body never reads the
  // return type.
  if (ts.isTypeLiteralNode(node)) return Tvoid;
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
  return tagStmt(node, ctx, translateStmtInner(node, ctx));
}

function translateStmtInner(node: ts.Statement, ctx: Ctx): Stmt {
  if (ts.isVariableStatement(node)) return translateVarDecl(node, ctx);
  if (ts.isExpressionStatement(node)) {
    const e = translateAssignmentLikeExpression(node.expression, ctx);
    if (e) return e; // statement-yielding form (e.g. assignment)
    // Barrier intrinsics lower directly to Barrier statements.
    const expr = node.expression;
    if (ts.isCallExpression(expr) && ts.isIdentifier(expr.expression)) {
      const name = expr.expression.text;
      if (name === "workgroupBarrier") return { kind: "Barrier", scope: "workgroup" };
      if (name === "storageBarrier")   return { kind: "Barrier", scope: "storage" };
      if (name === "discard")          return { kind: "Discard" };
    }
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
  return tagExpr(node, ctx, translateExprInner(node, ctx));
}

function translateExprInner(node: ts.Expression, ctx: Ctx): Expr {
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
    // `expr as T`: for scalar-primitive targets (i32/u32/f32/bool) emit
    // a Convert so WGSL/GLSL get the right cast op. For other targets
    // (e.g. vector shape annotation), relabel without a Convert.
    const inner = translateExpr(node.expression, ctx);
    const type = typeFromNode(node.type, ctx);
    if (isScalarPrimitive(type) && isScalarPrimitive(inner.type)
        && !sameScalar(type, inner.type)) {
      return { kind: "Convert", value: inner, type };
    }
    return { ...inner, type } as Expr;
  }
  if (ts.isObjectLiteralExpression(node)) return translateObjectLiteral(node, ctx);

  addDiagnostic(ctx, node, `frontend: unsupported expression (${ts.SyntaxKind[node.kind]})`);
  return { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
}

function translateIdentifier(node: ts.Identifier, ctx: Ctx): Expr {
  const v = ctx.vars.get(node.text);
  if (v) return { kind: "Var", var: v, type: v.type };
  const k = ctx.moduleConsts.get(node.text);
  if (k) return k;
  // Closure-captured identifiers (build-plugin-supplied) take
  // precedence over uniforms — same name in both maps means closure.
  const closureType = ctx.closureTypes.get(node.text);
  if (closureType) {
    return { kind: "ReadInput", scope: "Closure", name: node.text, type: closureType };
  }
  // Unresolved identifier — try the externalTypes table first (uniform /
  // sampler / storage names declared at module level), else fall back
  // to f32. Treat as ReadInput("Uniform").
  const t = ctx.externalTypes.get(node.text) ?? Tf32;
  return { kind: "ReadInput", scope: "Uniform", name: node.text, type: t };
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
  // Property-access call: either a static factory on a shipped type
  // (`M44f.fromCols(...)`, `V4f.zero` — wait, fields are not calls)
  // or an instance method (`v.add(other)`).
  if (ts.isPropertyAccessExpression(node.expression)) {
    // Static factory on a shipped math type? `M44f.fromCols(...)` and
    // `M44f.fromRows(...)` lower directly to the matching IR node.
    if (ts.isIdentifier(node.expression.expression)) {
      const typeName = node.expression.expression.text;
      const methodName = node.expression.name.text;
      const knownType = tryResolveTypeName(typeName);
      if (knownType && knownType.kind === "Matrix") {
        const args = node.arguments.map((a) => translateExpr(a, ctx));
        if (methodName === "fromCols") {
          return { kind: "MatrixFromCols", cols: args, type: knownType };
        }
        if (methodName === "fromRows") {
          return { kind: "MatrixFromRows", rows: args, type: knownType };
        }
      }
    }
    const target = translateExpr(node.expression.expression, ctx);
    const method = node.expression.name.text;
    const args = node.arguments.map((a) => translateExpr(a, ctx));
    return translateMethodCall(target, method, args, node, ctx);
  }
  // Free-function call.
  if (ts.isIdentifier(node.expression)) {
    const name = node.expression.text;
    const args = node.arguments.map((a) => translateExpr(a, ctx));
    // 1. Known intrinsic? (Note: `vec*()` / `mat*()` constructor
    //    calls were dropped — users write `new V*f(...)` /
    //    `M*f.fromCols(...)` instead, both of which work for CPU
    //    and GPU code via wombat.base.)
    const intr = lookupIntrinsic(name);
    if (intr) {
      const argTypes = args.map((a) => a.type);
      return { kind: "CallIntrinsic", op: intr, args, type: intr.returnTypeOf(argTypes) };
    }
    // 3. Known helper function — emit a properly-typed Call.
    const sig = ctx.helperSignatures.get(name);
    if (sig) {
      return {
        kind: "Call",
        fn: { id: name, signature: sig, pure: true },
        args,
        type: sig.returnType,
      };
    }
    // 4. Unknown — leave as Call(stub) and emit a diagnostic.
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
    case "lengthSquared":
      // No native intrinsic; lower to dot(v, v).
      return { kind: "Dot", lhs: target, rhs: target, type: Tf32 };
    case "distance":
    case "distanceSquared": {
      const intr = lookupIntrinsic("distance")!;
      const dist: Expr = { kind: "CallIntrinsic", op: intr, args: [target, args[0]!], type: Tf32 };
      if (method === "distance") return dist;
      return { kind: "Mul", lhs: dist, rhs: dist, type: Tf32 };
    }
    case "normalize": {
      const intr = lookupIntrinsic("normalize")!;
      return { kind: "CallIntrinsic", op: intr, args: [target], type: target.type };
    }
    // Element-wise math methods → existing intrinsic table.
    case "abs":
    case "floor":
    case "ceil":
    case "round":
    case "fract":
    case "sign": {
      const intr = lookupIntrinsic(method)!;
      return { kind: "CallIntrinsic", op: intr, args: [target], type: target.type };
    }
    case "min":
    case "max": {
      const intr = lookupIntrinsic(method)!;
      return { kind: "CallIntrinsic", op: intr, args: [target, args[0]!], type: target.type };
    }
    case "clamp": {
      const intr = lookupIntrinsic("clamp")!;
      return { kind: "CallIntrinsic", op: intr, args: [target, args[0]!, args[1]!], type: target.type };
    }
    case "lerp":
    case "mix": {
      const intr = lookupIntrinsic("mix")!;
      return { kind: "CallIntrinsic", op: intr, args: [target, args[0]!, args[1]!], type: target.type };
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
    // Uniform namespace: `u.field` lowers to ReadInput("Uniform").
    const ns = ctx.uniformNamespaces.get(node.expression.text);
    if (ns) {
      const fieldName = node.name.text;
      const fieldType = ns.get(fieldName);
      if (fieldType) {
        return { kind: "ReadInput", scope: "Uniform", name: fieldName, type: fieldType };
      }
      addDiagnostic(ctx, node, `frontend: unknown uniform "${fieldName}" on namespace "${node.expression.text}"`);
    }
    const builtins = ctx.builtinRecords.get(node.expression.text);
    if (builtins) {
      const fieldName = node.name.text;
      const entry = builtins.get(fieldName);
      if (entry) {
        return { kind: "ReadInput", scope: "Builtin", name: entry.semantic, type: entry.type };
      }
      addDiagnostic(ctx, node, `frontend: unknown builtin "${fieldName}" on ${node.expression.text}`);
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
  const elementType = target.type.kind === "Array" ? target.type.element : Tf32;
  return { kind: "Item", target, index, type: elementType };
}

// ─────────────────────────────────────────────────────────────────────
// L-expressions (assignment targets)
// ─────────────────────────────────────────────────────────────────────

function translateLExpr(node: ts.Expression, ctx: Ctx): LExpr {
  if (ts.isIdentifier(node)) {
    const v = ctx.vars.get(node.text);
    if (v) return { kind: "LVar", var: v, type: v.type };
    // Free identifier — assume it's a module-level binding (uniform,
    // sampler, storage buffer). Use the externalTypes table to give the
    // synthetic Var the right type so subsequent indexing knows the
    // element type, and storage-access inference can identify it.
    const externalType = ctx.externalTypes.get(node.text);
    if (externalType) {
      const synth: Var = { name: node.text, type: externalType, mutable: true };
      return { kind: "LVar", var: synth, type: externalType };
    }
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
    const elementType = target.type.kind === "Array"
      ? target.type.element
      : target.type.kind === "Vector"
      ? target.type.element
      : Tf32;
    return { kind: "LItem", target, index, type: elementType };
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

/**
 * Object-literal return values become a synthetic carrier that
 * `liftReturns` rewrites into per-output WriteOutput statements.
 * Carrier shape: a `Const(Null)` whose hidden `_record` property
 * holds the name → expression map. The IR shape is JSON-safe; the
 * `_record` field is invisible to JSON.stringify but visible to
 * pattern-matching passes.
 */
function translateObjectLiteral(node: ts.ObjectLiteralExpression, ctx: Ctx): Expr {
  const fields = new Map<string, Expr>();
  for (const prop of node.properties) {
    if (ts.isPropertyAssignment(prop) && ts.isIdentifier(prop.name)) {
      fields.set(prop.name.text, translateExpr(prop.initializer, ctx));
    } else if (ts.isShorthandPropertyAssignment(prop)) {
      // `{ outColor }` — short for `{ outColor: outColor }`
      fields.set(prop.name.text, translateExpr(prop.name, ctx));
    } else {
      addDiagnostic(ctx, prop, "frontend: unsupported object literal property");
    }
  }
  // Build the carrier expression. `_record` is non-enumerable so JSON
  // serialisation skips it; passes that pattern-match access it
  // directly via `(value as any)._record`.
  const carrier: Expr = { kind: "Const", value: { kind: "Null" }, type: Tvoid };
  Object.defineProperty(carrier, "_record", {
    value: fields,
    enumerable: false,
    writable: false,
  });
  return carrier;
}

function collectModuleConsts(source: ts.SourceFile): Map<string, Expr> {
  // Top-level `const X = <numeric/boolean literal>;` declarations.
  // Inlined at the use site so the emitter never sees a dangling free
  // identifier. We only handle plain numeric and boolean literals (and
  // unary-minus numerics) — anything else stays unresolved and falls
  // through to the externalTypes path.
  const out = new Map<string, Expr>();
  ts.forEachChild(source, (node) => {
    if (!ts.isVariableStatement(node)) return;
    const isConst = (node.declarationList.flags & ts.NodeFlags.Const) !== 0;
    if (!isConst) return;
    for (const d of node.declarationList.declarations) {
      if (!ts.isIdentifier(d.name) || !d.initializer) continue;
      const lit = literalToConst(d.initializer);
      if (lit) out.set(d.name.text, lit);
    }
  });
  return out;
}

function literalToConst(e: ts.Expression): Expr | undefined {
  if (ts.isNumericLiteral(e) || e.kind === ts.SyntaxKind.NumericLiteral) {
    const text = e.getText();
    const isInt = !/[.eE]/.test(text);
    if (isInt) {
      return { kind: "Const", value: { kind: "Int", signed: true, value: parseInt(text, 10) }, type: Ti32 };
    }
    return { kind: "Const", value: { kind: "Float", value: parseFloat(text) }, type: Tf32 };
  }
  if (e.kind === ts.SyntaxKind.TrueKeyword) return { kind: "Const", value: { kind: "Bool", value: true }, type: Tbool };
  if (e.kind === ts.SyntaxKind.FalseKeyword) return { kind: "Const", value: { kind: "Bool", value: false }, type: Tbool };
  if (ts.isPrefixUnaryExpression(e) && e.operator === ts.SyntaxKind.MinusToken) {
    const inner = literalToConst(e.operand);
    if (inner && inner.kind === "Const") {
      if (inner.value.kind === "Int") {
        return { kind: "Const", value: { ...inner.value, value: -inner.value.value }, type: inner.type };
      }
      if (inner.value.kind === "Float") {
        return { kind: "Const", value: { ...inner.value, value: -inner.value.value }, type: inner.type };
      }
    }
  }
  return undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Span propagation
//
// Every translated Stmt and Expr gets tagged with the originating
// TS span so the WGSL/GLSL emitters can build a v3 source map back
// to the user's TS source. We wrap `translateStmt` / `translateExpr`
// at the entry points; nested calls go through the wrappers so the
// whole IR ends up tagged. Helpers (e.g. `translateBlock`,
// `translateMethodCall`) return Exprs/Stmts that already carry the
// inner span — we let the outer wrapper override with the more
// specific node's span if it's a different range.
// ─────────────────────────────────────────────────────────────────────

function spanOf(node: ts.Node, ctx: Ctx): { file: string; start: number; end: number } {
  return { file: ctx.file, start: node.getStart(ctx.source), end: node.getEnd() };
}

function tagStmt(node: ts.Node, ctx: Ctx, s: Stmt): Stmt {
  // Don't overwrite a more specific span an inner call set —
  // pre-existing span typically points at a child node which is
  // strictly narrower than `node` and more useful for the source map.
  if (s.span) return s;
  return { ...s, span: spanOf(node, ctx) } as Stmt;
}

function tagExpr(node: ts.Node, ctx: Ctx, e: Expr): Expr {
  if (e.span) return e;
  const tagged = { ...e, span: spanOf(node, ctx) } as Expr;
  // The object-literal-return carrier from `translateObjectLiteral`
  // stashes its field map on a non-enumerable `_record` field. The
  // spread above strips it — re-attach so liftReturns still finds it.
  const record = (e as { _record?: ReadonlyMap<string, Expr> })._record;
  if (record !== undefined) {
    Object.defineProperty(tagged, "_record", {
      value: record, enumerable: false, writable: false,
    });
  }
  return tagged;
}

function isScalarPrimitive(t: Type): boolean {
  return t.kind === "Int" || t.kind === "Float" || t.kind === "Bool";
}
function sameScalar(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "Int" && b.kind === "Int") {
    return a.signed === b.signed && a.width === b.width;
  }
  return true;
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
