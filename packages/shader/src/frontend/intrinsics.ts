// Intrinsic table — TS function names that translate to IR
// `CallIntrinsic` nodes. The result type is computed from the argument
// types (mirroring shader-types' generic overloads).

import type { IntrinsicRef, Type } from "../ir/index.js";

const Tf32: Type = { kind: "Float", width: 32 };
const Tbool: Type = { kind: "Bool" };
const Tvec = (e: Type, dim: 2 | 3 | 4): Type => ({ kind: "Vector", element: e, dim });

function elementWise(args: readonly Type[]): Type {
  return args[0] ?? Tf32;
}
function scalarOnly(): Type {
  return Tf32;
}
function vec4Result(): Type {
  return Tvec(Tf32, 4);
}
function dotResult(args: readonly Type[]): Type {
  // dot returns a scalar of the operand element type.
  const a = args[0];
  if (a && a.kind === "Vector") return a.element;
  return Tf32;
}
function lengthResult(): Type {
  return Tf32;
}

function makePure(name: string, returnTypeOf: (args: readonly Type[]) => Type): IntrinsicRef {
  return {
    name, pure: true,
    emit: { glsl: name, wgsl: name },
    returnTypeOf,
  };
}

/** Pure intrinsic with target-specific spellings. */
function makePureWithEmit(
  name: string,
  emit: { glsl: string; wgsl: string },
  returnTypeOf: (args: readonly Type[]) => Type,
): IntrinsicRef {
  return { name, pure: true, emit, returnTypeOf };
}

const SAMPLER_BIND = (name: string): IntrinsicRef => ({
  name, pure: true, samplerBinding: true,
  emit: {
    glsl: name,
    wgsl: name === "texture" ? "textureSample"
        : name === "texelFetch" ? "textureLoad"
        : name,
  },
  returnTypeOf: vec4Result,
});

function atomicScalar(args: readonly Type[]): Type {
  // First arg is the atomic storage element reference; result is its
  // (non-atomic) underlying scalar type. The translator hands us the
  // pre-inference type (`Int(...)`); after the inference pass the
  // buffer's element becomes `AtomicI32`/`AtomicU32`, but the call
  // expression's result type remains the unwrapped scalar.
  const a = args[0];
  if (a && a.kind === "AtomicI32") return { kind: "Int", signed: true, width: 32 };
  if (a && a.kind === "AtomicU32") return { kind: "Int", signed: false, width: 32 };
  if (a && a.kind === "Int") return a;
  return { kind: "Int", signed: false, width: 32 };
}
function atomicOp(name: string, wgsl: string = name): IntrinsicRef {
  return {
    name, pure: false, atomic: true,
    emit: { glsl: name, wgsl },
    returnTypeOf: atomicScalar,
  };
}

const Tu32: Type = { kind: "Int", signed: false, width: 32 };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };
const Tvec2f: Type = Tvec(Tf32, 2);
const Tvec4f: Type = Tvec(Tf32, 4);

function u32Result(): Type { return Tu32; }
function i32Result(): Type { return Ti32; }
function vec2fResult(): Type { return Tvec2f; }
function vec4fResult(): Type { return Tvec4f; }

export const INTRINSICS: Record<string, IntrinsicRef> = {
  // trig / transcendental
  sin: makePure("sin", elementWise),
  cos: makePure("cos", elementWise),
  tan: makePure("tan", elementWise),
  asin: makePure("asin", scalarOnly),
  acos: makePure("acos", scalarOnly),
  atan: makePure("atan", scalarOnly),
  atan2: makePure("atan", scalarOnly),
  sinh: makePure("sinh", elementWise),
  cosh: makePure("cosh", elementWise),
  tanh: makePure("tanh", elementWise),
  asinh: makePure("asinh", scalarOnly),
  acosh: makePure("acosh", scalarOnly),
  atanh: makePure("atanh", scalarOnly),
  degrees: makePure("degrees", elementWise),
  radians: makePure("radians", elementWise),
  trunc: makePure("trunc", elementWise),
  exp: makePure("exp", elementWise),
  exp2: makePure("exp2", scalarOnly),
  log: makePure("log", scalarOnly),
  log2: makePure("log2", scalarOnly),
  pow: makePure("pow", elementWise),
  sqrt: makePure("sqrt", elementWise),
  inversesqrt: makePure("inversesqrt", elementWise),

  // utility
  abs: makePure("abs", elementWise),
  sign: makePure("sign", elementWise),
  floor: makePure("floor", elementWise),
  ceil: makePure("ceil", elementWise),
  fract: makePure("fract", elementWise),
  round: makePure("round", elementWise),
  mod: makePure("mod", elementWise),
  min: makePure("min", elementWise),
  max: makePure("max", elementWise),
  clamp: makePure("clamp", elementWise),
  mix: makePure("mix", elementWise),
  step: makePure("step", elementWise),
  smoothstep: makePure("smoothstep", elementWise),

  // geometric
  // `length` / `dot` / `cross` are provided exclusively as method
  // forms (`v.length()`, `a.dot(b)`, `a.cross(b)`); they translate
  // to dedicated IR nodes (`Length`, `Dot`, `Cross`) and don't go
  // through the intrinsic table. Removing the free-function form
  // eliminates a redundant code path.
  distance: makePure("distance", lengthResult),
  normalize: makePure("normalize", elementWise),
  reflect: makePure("reflect", elementWise),
  refract: makePure("refract", elementWise),
  faceforward: makePure("faceforward", elementWise),

  // derivatives (fragment-only). GLSL spells them `dFdx`/`dFdy`,
  // WGSL spells them `dpdx`/`dpdy`; `fwidth` is the same in both.
  // WGSL also exposes explicit fine/coarse variants — WebKit's
  // Metal backend handles the explicit forms more reliably than
  // the implementation-defined plain dpdx/dpdy, so we expose all
  // three.
  dFdx:       makePureWithEmit("dFdx",       { glsl: "dFdx",       wgsl: "dpdx" },       elementWise),
  dFdy:       makePureWithEmit("dFdy",       { glsl: "dFdy",       wgsl: "dpdy" },       elementWise),
  dFdxFine:   makePureWithEmit("dFdxFine",   { glsl: "dFdxFine",   wgsl: "dpdxFine" },   elementWise),
  dFdyFine:   makePureWithEmit("dFdyFine",   { glsl: "dFdyFine",   wgsl: "dpdyFine" },   elementWise),
  dFdxCoarse: makePureWithEmit("dFdxCoarse", { glsl: "dFdxCoarse", wgsl: "dpdxCoarse" }, elementWise),
  dFdyCoarse: makePureWithEmit("dFdyCoarse", { glsl: "dFdyCoarse", wgsl: "dpdyCoarse" }, elementWise),
  fwidth:       makePure("fwidth", elementWise),
  fwidthFine:   makePureWithEmit("fwidthFine",   { glsl: "fwidthFine",   wgsl: "fwidthFine" },   elementWise),
  fwidthCoarse: makePureWithEmit("fwidthCoarse", { glsl: "fwidthCoarse", wgsl: "fwidthCoarse" }, elementWise),

  // any/all over vec<bool>
  any: makePure("any", () => Tbool),
  all: makePure("all", () => Tbool),

  // sampler functions
  texture: SAMPLER_BIND("texture"),
  textureLod: SAMPLER_BIND("textureLod"),
  textureGrad: SAMPLER_BIND("textureGrad"),
  texelFetch: SAMPLER_BIND("texelFetch"),
  textureSampleCompare: {
    name: "textureSampleCompare", pure: true, samplerBinding: true,
    emit: { glsl: "texture", wgsl: "textureSampleCompare" },
    returnTypeOf: () => Tf32,
  },
  // packing / unpacking — WGSL-name-keyed; GLSL emits the
  // ES 3.00 spelling. Result types are u32 (pack) or vec*<f32>
  // (unpack) per the spec.
  pack2x16float: makePureWithEmit(
    "pack2x16float",
    { glsl: "packHalf2x16", wgsl: "pack2x16float" },
    u32Result,
  ),
  unpack2x16float: makePureWithEmit(
    "unpack2x16float",
    { glsl: "unpackHalf2x16", wgsl: "unpack2x16float" },
    vec2fResult,
  ),
  pack2x16unorm: makePureWithEmit(
    "pack2x16unorm",
    { glsl: "packUnorm2x16", wgsl: "pack2x16unorm" },
    u32Result,
  ),
  unpack2x16unorm: makePureWithEmit(
    "unpack2x16unorm",
    { glsl: "unpackUnorm2x16", wgsl: "unpack2x16unorm" },
    vec2fResult,
  ),
  pack2x16snorm: makePureWithEmit(
    "pack2x16snorm",
    { glsl: "packSnorm2x16", wgsl: "pack2x16snorm" },
    u32Result,
  ),
  unpack2x16snorm: makePureWithEmit(
    "unpack2x16snorm",
    { glsl: "unpackSnorm2x16", wgsl: "unpack2x16snorm" },
    vec2fResult,
  ),
  pack4x8unorm: makePureWithEmit(
    "pack4x8unorm",
    { glsl: "packUnorm4x8", wgsl: "pack4x8unorm" },
    u32Result,
  ),
  unpack4x8unorm: makePureWithEmit(
    "unpack4x8unorm",
    { glsl: "unpackUnorm4x8", wgsl: "unpack4x8unorm" },
    vec4fResult,
  ),
  pack4x8snorm: makePureWithEmit(
    "pack4x8snorm",
    { glsl: "packSnorm4x8", wgsl: "pack4x8snorm" },
    u32Result,
  ),
  unpack4x8snorm: makePureWithEmit(
    "unpack4x8snorm",
    { glsl: "unpackSnorm4x8", wgsl: "unpack4x8snorm" },
    vec4fResult,
  ),

  // bit ops — WGSL-name-keyed; GLSL emits the ES 3.00 spelling.
  countOneBits: makePureWithEmit(
    "countOneBits",
    { glsl: "bitCount", wgsl: "countOneBits" },
    i32Result,
  ),
  extractBits: makePureWithEmit(
    "extractBits",
    { glsl: "bitfieldExtract", wgsl: "extractBits" },
    elementWise,
  ),
  insertBits: makePureWithEmit(
    "insertBits",
    { glsl: "bitfieldInsert", wgsl: "insertBits" },
    elementWise,
  ),
  reverseBits: makePureWithEmit(
    "reverseBits",
    { glsl: "bitfieldReverse", wgsl: "reverseBits" },
    elementWise,
  ),
  firstLeadingBit: makePureWithEmit(
    "firstLeadingBit",
    { glsl: "findMSB", wgsl: "firstLeadingBit" },
    i32Result,
  ),
  firstTrailingBit: makePureWithEmit(
    "firstTrailingBit",
    { glsl: "findLSB", wgsl: "firstTrailingBit" },
    i32Result,
  ),

  // atomics — first arg is a storage element ref; emitter prepends `&`.
  atomicLoad: atomicOp("atomicLoad"),
  atomicStore: { name: "atomicStore", pure: false, atomic: true,
    emit: { glsl: "atomicStore", wgsl: "atomicStore" },
    returnTypeOf: () => ({ kind: "Void" }) },
  atomicAdd: atomicOp("atomicAdd"),
  atomicSub: atomicOp("atomicSub"),
  atomicMin: atomicOp("atomicMin"),
  atomicMax: atomicOp("atomicMax"),
  atomicAnd: atomicOp("atomicAnd"),
  atomicOr: atomicOp("atomicOr"),
  atomicXor: atomicOp("atomicXor"),
  atomicExchange: atomicOp("atomicExchange"),
  atomicCompareExchangeWeak: atomicOp("atomicCompareExchangeWeak"),

  // textureStore writes to a storage-texture element; impure so DCE keeps it.
  textureStore: {
    name: "textureStore", pure: false, samplerBinding: true,
    emit: { glsl: "imageStore", wgsl: "textureStore" },
    returnTypeOf: () => ({ kind: "Void" }),
  },
  // textureLoad reads a texel — same WGSL spelling for sampled and
  // storage textures. We rely on the surrounding type to disambiguate.
  textureLoad: {
    name: "textureLoad", pure: true, samplerBinding: true,
    emit: { glsl: "imageLoad", wgsl: "textureLoad" },
    returnTypeOf: vec4Result,
  },
  textureGather: {
    name: "textureGather", pure: true, samplerBinding: true,
    emit: { glsl: "textureGather", wgsl: "textureGather" },
    returnTypeOf: vec4Result,
  },

  textureSize: makePure("textureSize", (args) => {
    // returns ivec of texture's dimensionality
    const a = args[0];
    if (a && a.kind === "Sampler") {
      const dim = a.target === "1D" ? 1 : a.target === "3D" ? 3 : 2;
      return Tvec({ kind: "Int", signed: true, width: 32 }, dim as 2 | 3);
    }
    return Tvec({ kind: "Int", signed: true, width: 32 }, 2);
  }),
};

export function lookupIntrinsic(name: string): IntrinsicRef | undefined {
  return INTRINSICS[name];
}
