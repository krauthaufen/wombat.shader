// Public API barrel for `@aardworx/wombat.shader-ir`.

export type {
  // spans
  Span,
  // types
  Type,
  StructField,
  SamplerTarget,
  SampledType,
  StorageAccess,
  StorageTextureFormat,
  // variables / literals
  Var,
  Literal,
  // function references
  Parameter,
  FunctionSignature,
  FunctionRef,
  IntrinsicRef,
  // expressions
  Expr,
  ExprBody,
  LExpr,
  LExprBody,
  RExpr,
  VecComp,
  InputScope,
  // statements
  Stmt,
  SwitchCase,
  BarrierScope,
  // module shape
  ValueDef,
  TypeDef,
  EntryDef,
  EntryParameter,
  EntryDecoration,
  ParamDecoration,
  BuiltinSemantic,
  Stage,
  UniformDecl,
  BindingPoint,
  FnAttr,
  Module,
  ModuleMeta,
} from "./types.js";

export {
  Tvoid,
  Tbool,
  Ti32,
  Tu32,
  Tf32,
  Vec,
  Mat,
} from "./types.js";

export {
  visitStmt,
  visitExprChildren,
  visitLExprChildren,
} from "./visit.js";

export type { ExprVisitor, StmtVisitor } from "./visit.js";

export { serialise, deserialise } from "./serialize.js";
export type { IntrinsicRegistry } from "./serialize.js";

export { hashModule, hashValue, combineHashes, stableStringify } from "./hash.js";
export { prettyPrint } from "./pretty.js";
export { buildSourceMap } from "./sourcemap.js";
export type { SourceMap, BuildSourceMapInput } from "./sourcemap.js";
