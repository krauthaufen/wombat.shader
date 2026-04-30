// Hole resolution — closure-captured values get inlined as IR
// constants.
//
// The build-time plugin produces IR templates whose closure references
// translate to `ReadInput(scope: "Closure", name, type)`. At link time
// (or directly at compile time when the values are statically known),
// `resolveHoles` walks the module, looks up each hole's name in the
// supplied value map, and substitutes a typed `Const` / `NewVector` /
// `MatrixFromCols` expression. After this pass runs, the IR carries
// no `"Closure"` reads — every hole is fully inlined.
//
// This is the constant-only path. Adaptive (`aval`-driven) holes will
// follow later, lifting holes to runtime-bound uniform decls instead.

import type {
  Expr,
  Module,
  Type,
  ValueDef,
} from "@aardworx/wombat.shader-ir";
import { mapStmt } from "./transform.js";

/**
 * A scalar JS value, an array (for vectors / matrices), or a plain
 * object with `.x`/`.y`/`.z`/`.w` for vectors. The pass uses the IR
 * hole's own type to drive the conversion shape; pass whatever is
 * convenient and the pass picks it apart.
 */
export type HoleValue =
  | number
  | boolean
  | readonly number[]
  | readonly (readonly number[])[]
  | { readonly x: number; readonly y: number; readonly z?: number; readonly w?: number };

export type Holes = Readonly<Record<string, HoleValue>>;

export function resolveHoles(module: Module, holes: Holes): Module {
  const values = module.values.map((v): ValueDef => {
    if (v.kind === "Entry") {
      const body = mapStmt(v.entry.body, { expr: (e) => substExpr(e, holes) });
      return body === v.entry.body ? v : { ...v, entry: { ...v.entry, body } };
    }
    if (v.kind === "Function") {
      const body = mapStmt(v.body, { expr: (e) => substExpr(e, holes) });
      return body === v.body ? v : { ...v, body };
    }
    return v;
  });
  return values === module.values ? module : { ...module, values };
}

// ─────────────────────────────────────────────────────────────────────

function substExpr(e: Expr, holes: Holes): Expr {
  if (e.kind !== "ReadInput" || e.scope !== "Closure") return e;
  if (!(e.name in holes)) {
    throw new Error(`resolveHoles: missing value for closure hole "${e.name}"`);
  }
  return holeToExpr(holes[e.name]!, e.type, e.name);
}

function holeToExpr(value: HoleValue, type: Type, name: string): Expr {
  switch (type.kind) {
    case "Float": {
      const n = scalarToNumber(value, name);
      return { kind: "Const", value: { kind: "Float", value: n }, type };
    }
    case "Int": {
      const n = scalarToNumber(value, name);
      return {
        kind: "Const",
        value: { kind: "Int", signed: type.signed, value: Math.trunc(n) },
        type,
      };
    }
    case "Bool": {
      if (typeof value !== "boolean") {
        throw new Error(`resolveHoles: hole "${name}" expected boolean, got ${typeof value}`);
      }
      return { kind: "Const", value: { kind: "Bool", value }, type };
    }
    case "Vector": {
      const comps = vectorComponents(value, type.dim, name);
      const elementType = type.element;
      const args: Expr[] = comps.map((n) => scalarConst(n, elementType));
      return { kind: "NewVector", components: args, type };
    }
    case "Matrix": {
      // Accept an array of column vectors (each itself any HoleValue
      // that decodes to a Vector of `rows` components).
      const cols = matrixColumns(value, type.cols, name);
      const colVecType: Type = { kind: "Vector", element: type.element, dim: type.rows };
      const colExprs: Expr[] = cols.map((c) => holeToExpr(c, colVecType, name));
      return { kind: "MatrixFromCols", cols: colExprs, type };
    }
    default:
      throw new Error(`resolveHoles: hole "${name}" has unsupported type ${type.kind}`);
  }
}

function scalarConst(n: number, t: Type): Expr {
  if (t.kind === "Float") {
    return { kind: "Const", value: { kind: "Float", value: n }, type: t };
  }
  if (t.kind === "Int") {
    return {
      kind: "Const",
      value: { kind: "Int", signed: t.signed, value: Math.trunc(n) },
      type: t,
    };
  }
  if (t.kind === "Bool") {
    return { kind: "Const", value: { kind: "Bool", value: !!n }, type: t };
  }
  throw new Error(`resolveHoles: cannot lower scalar to ${t.kind}`);
}

function scalarToNumber(value: HoleValue, name: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  throw new Error(
    `resolveHoles: scalar hole "${name}" expected number, got ${describeValue(value)}`,
  );
}

function vectorComponents(value: HoleValue, dim: number, name: string): number[] {
  if (Array.isArray(value) && value.every((v) => typeof v === "number")) {
    if (value.length !== dim) {
      throw new Error(`resolveHoles: vector hole "${name}" expected ${dim} components, got ${value.length}`);
    }
    return value as number[];
  }
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const v = value as { x?: number; y?: number; z?: number; w?: number };
    const comps = [v.x, v.y, v.z, v.w].slice(0, dim);
    if (comps.some((c) => typeof c !== "number")) {
      throw new Error(`resolveHoles: vector hole "${name}" missing required component`);
    }
    return comps as number[];
  }
  throw new Error(`resolveHoles: vector hole "${name}" got ${describeValue(value)}`);
}

function matrixColumns(value: HoleValue, cols: number, name: string): readonly HoleValue[] {
  if (!Array.isArray(value)) {
    throw new Error(`resolveHoles: matrix hole "${name}" expected array of columns, got ${describeValue(value)}`);
  }
  if (value.length !== cols) {
    throw new Error(`resolveHoles: matrix hole "${name}" expected ${cols} columns, got ${value.length}`);
  }
  return value as readonly HoleValue[];
}

function describeValue(v: unknown): string {
  if (Array.isArray(v)) return `array(length=${v.length})`;
  if (v === null) return "null";
  return typeof v;
}

