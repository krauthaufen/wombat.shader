// Public API for `@aardworx/wombat.shader-passes`.

export { foldConstants, foldExpr, foldStmt } from "./foldConstants.js";
export { dce, dceStmt } from "./dce.js";
export { cse, cseStmt } from "./cse.js";
export { inlinePass, inlineAllAttributed } from "./inline.js";
export type { InlinePolicy } from "./inline.js";

export {
  isPure,
  hasSideEffects,
  freeVarsExpr,
  freeVarsLExpr,
  freeVarsStmt,
  readInputs,
  writtenOutputs,
  liveAfter,
  isRExprPure,
} from "./analysis.js";
export type { ScopedName } from "./analysis.js";

export {
  mapExpr,
  mapLExpr,
  mapStmt,
  mapExprChildren,
  mapLExprChildren,
  mapRExpr,
  mapStmtChildren,
} from "./transform.js";
export type {
  ExprMapper,
  LExprMapper,
  StmtMapper,
  StmtChildMapper,
} from "./transform.js";

export {
  substVar,
  substInput,
  substVars,
  substVarsExpr,
} from "./substitute.js";
