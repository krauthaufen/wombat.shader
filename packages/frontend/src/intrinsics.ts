// Intrinsic table — TS function names that translate to IR
// `CallIntrinsic` nodes. The result type is computed from the argument
// types (mirroring shader-types' generic overloads).

import type { IntrinsicRef, Type } from "@aardworx/wombat.shader-ir";

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

const SAMPLER_BIND = (name: string): IntrinsicRef => ({
  name, pure: true, samplerBinding: true,
  emit: { glsl: name, wgsl: name === "texture" ? "textureSample" : name },
  returnTypeOf: vec4Result,
});

export const INTRINSICS: Record<string, IntrinsicRef> = {
  // trig / transcendental
  sin: makePure("sin", elementWise),
  cos: makePure("cos", elementWise),
  tan: makePure("tan", elementWise),
  asin: makePure("asin", scalarOnly),
  acos: makePure("acos", scalarOnly),
  atan: makePure("atan", scalarOnly),
  atan2: makePure("atan", scalarOnly),
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
  length: makePure("length", lengthResult),
  distance: makePure("distance", lengthResult),
  dot: makePure("dot", dotResult),
  cross: makePure("cross", elementWise),
  normalize: makePure("normalize", elementWise),
  reflect: makePure("reflect", elementWise),
  refract: makePure("refract", elementWise),
  faceforward: makePure("faceforward", elementWise),

  // derivatives (fragment-only)
  dFdx: makePure("dFdx", elementWise),
  dFdy: makePure("dFdy", elementWise),
  fwidth: makePure("fwidth", elementWise),

  // any/all over vec<bool>
  any: makePure("any", () => Tbool),
  all: makePure("all", () => Tbool),

  // sampler functions
  texture: SAMPLER_BIND("texture"),
  textureLod: SAMPLER_BIND("textureLod"),
  textureGrad: SAMPLER_BIND("textureGrad"),
  texelFetch: SAMPLER_BIND("texelFetch"),
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
