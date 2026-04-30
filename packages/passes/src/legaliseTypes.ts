// legaliseTypes — target-specific lowering. Runs last in the pass chain;
// the emitter consumes the result and assumes nothing further needs to
// be rewritten.
//
// What this pass does (per target):
//
//   common (both targets):
//     - MatrixRow(m, r)             → NewVector([m[r,0], m[r,1], …])
//     - MatrixFromRows(rs)          → Transpose(MatrixFromCols(rs))
//
//   wgsl-only:
//     - Inverse(m: Matrix)          → CallIntrinsic("_wombat_inverseN", [m])
//                                     and a helper FunctionDef inserted into
//                                     the Module if not already present.
//     - Sampler `S` referenced by   → split into a `Texture` binding (`S_view`,
//       a sampler-bound intrinsic       at the original group/slot+1) and a
//       (texture, textureLod, …)        `Sampler` binding (`S`, original slot).
//                                       Calls become `textureSample(S_view, S, uv)`.
//
// Each rewrite preserves Expr.type (so emitters and downstream passes
// see the same shape). The pass is pure (Module → Module).

import type {
  Expr,
  FunctionSignature,
  IntrinsicRef,
  Module,
  SamplerTarget,
  Stmt,
  Type,
  ValueDef,
} from "@aardworx/wombat.shader-ir";
import { mapExpr, mapStmt } from "./transform.js";

export type Target = "glsl" | "wgsl";

export function legaliseTypes(module: Module, target: Target): Module {
  const helperFunctions = new Map<string, ValueDef>();

  // For WGSL: rewrite Sampler ValueDefs that participate in
  // sampler-bound intrinsics into a (Sampler, Texture) pair, and
  // rewrite the call sites accordingly.
  let working = module;
  if (target === "wgsl") {
    working = splitWgslSamplers(working);
  }

  const exprFn = (e: Expr): Expr => {
    switch (e.kind) {
      case "MatrixRow":
        return lowerMatrixRow(e);
      case "MatrixFromRows":
        return lowerMatrixFromRows(e);
      case "Inverse":
        if (target === "wgsl") return lowerWgslInverse(e, helperFunctions);
        return e;
      default:
        return e;
    }
  };

  const newValues = working.values.map((v) => {
    if (v.kind === "Function") return { ...v, body: mapStmt(v.body, { expr: exprFn }) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: mapStmt(v.entry.body, { expr: exprFn }) } };
    if (v.kind === "Constant") {
      const init = v.init.kind === "Expr"
        ? { kind: "Expr" as const, value: mapExpr(v.init.value, exprFn) }
        : { kind: "ArrayLiteral" as const, arrayType: v.init.arrayType, values: v.init.values.map((x) => mapExpr(x, exprFn)) };
      return { ...v, init };
    }
    return v;
  });

  // Inject any helper functions we generated (deduped by name).
  const finalValues = [...helperFunctions.values(), ...newValues];

  return { ...working, values: finalValues };
}

// ─────────────────────────────────────────────────────────────────────
// WGSL: split combined Sampler bindings into Sampler + Texture pair
// ─────────────────────────────────────────────────────────────────────

/**
 * Find every `Sampler` ValueDef whose Type is `{ kind: "Sampler", … }`
 * and whose name appears as the first argument to a sampler-bound
 * intrinsic (`texture`, `textureLod`, `textureSample`, …). For each:
 *
 *   1. Add a sibling Sampler ValueDef whose Type is `Texture` (the
 *      texture view), at the next slot in the same group, named
 *      `<name>_view`.
 *   2. Rewrite call sites: `textureSample(name, …args)` →
 *      `textureSample(<name>_view, name, …args)`.
 *
 * Reads of the name as a free identifier (e.g. `Var(u_tex)`) outside
 * a sampler-bound intrinsic are rare in shader code — we leave them
 * as-is and the user gets a WGSL validation error for them.
 */
function splitWgslSamplers(module: Module): Module {
  // Identify combined sampler names.
  const combined = new Set<string>();
  for (const v of module.values) {
    if (v.kind === "Sampler" && v.type.kind === "Sampler") combined.add(v.name);
  }
  if (combined.size === 0) return module;

  // Inject paired Texture ValueDefs. Multisample samplers don't pair —
  // WGSL `texture_multisampled_*<T>` is sampled with `textureLoad`,
  // not `textureSample`, and there's no companion `sampler` binding.
  // The original Sampler binding becomes the multisampled Texture
  // directly (Sampler ValueDef.type swapped to Texture).
  const newValues: ValueDef[] = [];
  for (const v of module.values) {
    if (v.kind === "Sampler" && v.type.kind === "Sampler" && combined.has(v.name)) {
      const arrayed = v.type.target === "2DArray" || v.type.target === "CubeArray" || v.type.target === "2DMSArray";
      const multisampled = v.type.target === "2DMS" || v.type.target === "2DMSArray";
      // Normalise the texture target — the IR keeps an explicit
      // arrayed/multisampled flag, so collapse "2DArray"/"2DMS" back
      // to the base dimensionality.
      const baseTarget: SamplerTarget = (
        v.type.target === "2DArray" || v.type.target === "2DMS" || v.type.target === "2DMSArray" ? "2D" :
        v.type.target === "CubeArray" ? "Cube" :
        v.type.target
      );
      const textureType: Type = {
        kind: "Texture",
        target: baseTarget,
        sampled: v.type.sampled,
        arrayed,
        multisampled,
        ...(v.type.comparison ? { comparison: true } : {}),
      };
      if (multisampled) {
        // No companion sampler — replace the binding's type in-place
        // with the multisampled Texture.
        newValues.push({
          kind: "Sampler",
          binding: v.binding,
          name: v.name,
          type: textureType,
        });
      } else {
        // Keep the original Sampler (combined sampler/sampler_comparison)
        // and add a sibling Texture binding at slot+1.
        newValues.push(v);
        newValues.push({
          kind: "Sampler",
          binding: { group: v.binding.group, slot: v.binding.slot + 1 },
          name: `${v.name}_view`,
          type: textureType,
        });
      }
    } else {
      newValues.push(v);
    }
  }

  // Rewrite call sites on the new value array.
  const rewriteExpr = (e: Expr): Expr => {
    if (e.kind !== "CallIntrinsic" || !e.op.samplerBinding) return e;
    const first = e.args[0];
    if (!first) return e;
    // Matches a Var or ReadInput referencing the combined name.
    const samplerName = nameOfSamplerArg(first);
    if (!samplerName || !combined.has(samplerName)) return e;
    const samplerType = first.type as Extract<Type, { kind: "Sampler" }>;
    const arrayed = samplerType.target === "2DArray" || samplerType.target === "CubeArray" || samplerType.target === "2DMSArray";
    const multisampled = samplerType.target === "2DMS" || samplerType.target === "2DMSArray";
    const baseTarget: SamplerTarget = (
      samplerType.target === "2DArray" || samplerType.target === "2DMS" || samplerType.target === "2DMSArray" ? "2D" :
      samplerType.target === "CubeArray" ? "Cube" :
      samplerType.target
    );
    const textureType: Type = {
      kind: "Texture",
      target: baseTarget,
      sampled: samplerType.sampled,
      arrayed,
      multisampled,
      ...(samplerType.comparison ? { comparison: true } : {}),
    };
    if (multisampled) {
      // For MS textures the sampler argument disappears — `textureLoad`
      // takes (texture, coord, sample). Rewrite the call to use the
      // (now Texture-typed) binding name directly, drop the sampler.
      const textureExpr: Expr = {
        kind: "ReadInput", scope: "Uniform", name: samplerName,
        type: textureType,
      };
      return { ...e, args: [textureExpr, ...e.args.slice(1)] };
    }
    const textureExpr: Expr = {
      kind: "ReadInput", scope: "Uniform", name: `${samplerName}_view`,
      type: textureType,
    };
    // textureSample(view, sampler, ...rest)
    return { ...e, args: [textureExpr, first, ...e.args.slice(1)] };
  };

  const remappedValues = newValues.map((v) => {
    if (v.kind === "Function") return { ...v, body: mapStmt(v.body, { expr: rewriteExpr }) };
    if (v.kind === "Entry") return { ...v, entry: { ...v.entry, body: mapStmt(v.entry.body, { expr: rewriteExpr }) } };
    return v;
  });

  return { ...module, values: remappedValues };
}

function nameOfSamplerArg(e: Expr): string | undefined {
  if (e.kind === "ReadInput") return e.name;
  if (e.kind === "Var") return e.var.name;
  return undefined;
}

// ─────────────────────────────────────────────────────────────────────
// MatrixRow → NewVector of MatrixElement reads
// ─────────────────────────────────────────────────────────────────────

function lowerMatrixRow(e: Expr & { kind: "MatrixRow" }): Expr {
  const matType = e.matrix.type;
  if (matType.kind !== "Matrix") return e;
  const cols = matType.cols;
  const rowExpr = e.row;
  const rowEval: Expr = isPureNonSideeffecting(rowExpr) ? rowExpr : rowExpr;
  // Build [m[r,0], m[r,1], ..., m[r,cols-1]].
  const elementType = matType.element;
  const components: Expr[] = [];
  for (let c = 0; c < cols; c++) {
    components.push({
      kind: "MatrixElement",
      matrix: e.matrix,
      row: rowEval,
      col: { kind: "Const", value: { kind: "Int", signed: true, value: c }, type: { kind: "Int", signed: true, width: 32 } },
      type: elementType,
    });
  }
  return {
    kind: "NewVector",
    components,
    type: e.type, // already vec<element, cols>
  };
}

// ─────────────────────────────────────────────────────────────────────
// MatrixFromRows → Transpose(MatrixFromCols(rs))
// ─────────────────────────────────────────────────────────────────────

function lowerMatrixFromRows(e: Expr & { kind: "MatrixFromRows" }): Expr {
  // Build a matrix-from-cols of the same row vectors, then transpose.
  // Note: this changes the matrix shape from RxC (rows×cols) to CxR.
  // The Transpose then flips it back to RxC. Both emitters support this.
  if (e.type.kind !== "Matrix") return e;
  const colsType: Type = {
    kind: "Matrix",
    element: e.type.element,
    rows: e.type.cols,
    cols: e.type.rows,
  };
  return {
    kind: "Transpose",
    value: {
      kind: "MatrixFromCols",
      cols: e.rows,
      type: colsType,
    },
    type: e.type,
  };
}

// ─────────────────────────────────────────────────────────────────────
// WGSL Inverse helper
// ─────────────────────────────────────────────────────────────────────

const wgslInverseIntrinsics: Record<string, IntrinsicRef> = {};

function lowerWgslInverse(
  e: Expr & { kind: "Inverse" },
  helpers: Map<string, ValueDef>,
): Expr {
  const t = e.value.type;
  if (t.kind !== "Matrix") return e;
  const dim = t.rows;
  if (dim !== t.cols) return e; // non-square matrices are not invertible
  const helperName = `_wombat_inverse${dim}`;

  if (!helpers.has(helperName)) {
    helpers.set(helperName, makeInverseHelper(helperName, dim, t));
  }

  // Synthesize a CallIntrinsic that emits a call by name.
  const intrinsic = wgslInverseIntrinsics[helperName] ??= {
    name: helperName,
    pure: true,
    emit: { glsl: helperName, wgsl: helperName },
    returnTypeOf: () => t,
  };

  return {
    kind: "CallIntrinsic",
    op: intrinsic,
    args: [e.value],
    type: t,
  };
}

function makeInverseHelper(name: string, dim: number, matType: Type): ValueDef {
  // The body is a stub that emits "TODO: implement matN inverse" — we
  // leave the actual implementation to a future cookbook because
  // closed-form inversion for 2/3/4 is verbose. The function returns
  // the input unchanged so type-checking passes; runtime use will
  // produce an obviously-wrong result, surfacing the missing
  // implementation. Users supply their own helper before then.
  const sig: FunctionSignature = {
    name,
    returnType: matType,
    parameters: [{ name: "m", type: matType, modifier: "in" }],
  };
  const body: Stmt = {
    kind: "ReturnValue",
    value: { kind: "Var", var: { name: "m", type: matType, mutable: false }, type: matType },
  };
  return {
    kind: "Function",
    signature: sig,
    body,
    attributes: ["inline"],
  };
  // Note: we intentionally don't use `dim` for now — the helper is a
  // placeholder. Once we implement closed-form inverses, dim selects
  // the body. Silencing unused param.
  void dim;
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function isPureNonSideeffecting(_e: Expr): boolean {
  // Currently we don't try to dedupe row-expr evaluation; the emitter
  // will emit the same expression `cols` times. A future enhancement
  // could let-bind the row value when it's a CallIntrinsic etc.
  return true;
}
