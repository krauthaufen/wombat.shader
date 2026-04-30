// Public API for `@aardworx/wombat.shader-frontend`.
//
// `parseShader` is the top-level entry point: hands it a source string
// and a list of entry-point requests, get back an IR Module.
//
// `translateFunction` is a lower-level handle for tools that already
// have a TS source and want IR for one named function without
// generating a Module wrapper.

export { parseShader } from "./buildModule.js";
export type { EntryRequest, ParseShaderInput } from "./buildModule.js";

export { translateFunction, TranslationError } from "./translate.js";
export type { TranslateOptions, TranslatedFunction, Diagnostic } from "./translate.js";

export { resolveTypeName, tryResolveTypeName } from "./types.js";
export { lookupIntrinsic, INTRINSICS } from "./intrinsics.js";
