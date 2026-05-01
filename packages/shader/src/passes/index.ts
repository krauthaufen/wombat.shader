// Public API for `@aardworx/wombat.shader-passes`.

export { foldConstants, foldExpr, foldStmt } from "./foldConstants.js";
export { dce, dceStmt } from "./dce.js";
export { cse, cseStmt } from "./cse.js";
export { inlinePass, inlineAllAttributed } from "./inline.js";
export type { InlinePolicy } from "./inline.js";
export { reduceUniforms } from "./reduceUniforms.js";
export { pruneCrossStage } from "./pruneCrossStage.js";
export { composeStages } from "./composeStages.js";
export { linkFragmentOutputs } from "./linkFragmentOutputs.js";
export type { FragmentOutputLayout } from "./linkFragmentOutputs.js";
export { linkCrossStage, paramKey } from "./linkCrossStage.js";
export {
  effectDependencies,
  type OutputDep,
} from "./effectDeps.js";
export { legaliseTypes } from "./legaliseTypes.js";
export type { Target } from "./legaliseTypes.js";
export { liftReturns } from "./liftReturns.js";
export type { LiftReturnsOptions } from "./liftReturns.js";
export { inferStorageAccess } from "./inferStorageAccess.js";
export { resolveHoles } from "./resolveHoles.js";
export type { HoleValue, Holes } from "./resolveHoles.js";
export { reverseMatrixOps } from "./reverseMatrixOps.js";

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
