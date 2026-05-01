// Vector types — re-exported from `@aardworx/wombat.base`.
//
// Single source of truth: the same `V*f` / `V*i` / `V*ui` / `V*b`
// classes are used for CPU math (in user app code) and inside shader
// markers. The shader frontend recognises them by name via
// `tryResolveTypeName` and lowers references to the right
// `Vector(<element>, <dim>)` IR types. No duplication, no brand
// drift.
//
// `boperators` recognises the `__aardworxMathBrand` field on each
// class and rewrites `a + b` / `a * b` etc. to method-call form
// before the shader plugin sees the AST, so user shader code can
// use plain math operators throughout.

export {
  V2b, V3b, V4b,
  V2i, V3i, V4i,
  V2ui, V3ui, V4ui,
  V2f, V3f, V4f,
} from "@aardworx/wombat.base";
