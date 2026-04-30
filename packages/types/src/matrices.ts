// Matrix types — re-exported from `@aardworx/wombat.base`.
//
// The shader frontend recognises `M22f` / `M33f` / `M44f` and the
// rectangular `M{R}{C}f` variants by name and lowers references to
// `Matrix(Float, R, C)` IR types.

export {
  M22f, M33f, M44f,
  M23f, M24f, M32f, M34f, M42f, M43f,
} from "@aardworx/wombat.base";
