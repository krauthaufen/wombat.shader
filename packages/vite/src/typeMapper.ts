// TypeScript Type → IR Type mapping. Used by the plugin to:
//
//   - Resolve the IR type of a closure capture from its TS type.
//   - Walk the type literal of `declare const u: { … }` to build a
//     uniform-block shape.
//
// We recognise the shipped `@aardworx/wombat.shader-types` brands by
// symbol name. Anything else falls through.

import ts from "typescript";
import type {
  SamplerTarget,
  StorageAccess,
  StorageTextureFormat,
  Type as IRType,
} from "@aardworx/wombat.shader-ir";
import { tryResolveTypeName } from "@aardworx/wombat.shader-frontend";

export function tsTypeToIR(t: ts.Type, checker: ts.TypeChecker): IRType | undefined {
  // Adaptive `aval<T>` — unwrap to its inner IR type. Detection is by
  // symbol name + arity, so this works whether `aval` comes from
  // `@aardworx/adaptive` or any user-side alias matching the shape.
  const unwrapped = unwrapAval(t, checker);
  if (unwrapped) return unwrapped;
  // Named brand (V3f, M44f, Sampler2D, …).
  const sym = t.getSymbol() ?? t.aliasSymbol;
  if (sym) {
    const named = tryResolveTypeName(sym.getName());
    if (named) return named;
  }
  // Primitive scalars by flag.
  if (t.flags & ts.TypeFlags.Boolean || t.flags & ts.TypeFlags.BooleanLiteral) {
    return { kind: "Bool" };
  }
  if (t.flags & ts.TypeFlags.Number || t.flags & ts.TypeFlags.NumberLiteral) {
    return { kind: "Float", width: 32 };
  }
  // Array — ts.TypeReference with target.symbol.name === "Array".
  if (checker.isArrayType?.(t)) {
    const elem = checker.getTypeArguments(t as ts.TypeReference)[0];
    if (elem) {
      const inner = tsTypeToIR(elem, checker);
      if (inner) return { kind: "Array", element: inner, length: "runtime" };
    }
  }
  return undefined;
}

/**
 * `true` if a TS type is `aval<T>` for some shader-mappable T.
 * Drives the specialization/uniform split: aval captures are uniform
 * bindings, plain values specialize as constants.
 */
export function isAvalType(t: ts.Type, _checker: ts.TypeChecker): boolean {
  const sym = t.getSymbol() ?? t.aliasSymbol;
  if (!sym) return false;
  return sym.getName() === "aval" || sym.getName() === "AVal";
}

function unwrapAval(t: ts.Type, checker: ts.TypeChecker): IRType | undefined {
  if (!isAvalType(t, checker)) return undefined;
  const args = checker.getTypeArguments(t as ts.TypeReference);
  const inner = args[0];
  if (!inner) return undefined;
  return tsTypeToIR(inner, checker);
}

// ─── Storage-buffer / storage-texture type recognition ───────────────

const STORAGE_TEXTURE_KIND: Record<string, { target: SamplerTarget; arrayed: boolean }> = {
  StorageTexture2D:      { target: "2D", arrayed: false },
  StorageTexture3D:      { target: "3D", arrayed: false },
  StorageTexture2DArray: { target: "2D", arrayed: true },
};

function aliasName(t: ts.Type): string | undefined {
  return t.aliasSymbol?.getName() ?? t.getSymbol()?.getName();
}

/**
 * Walk a symbol's declaration to find an explicit type annotation
 * `Storage<…>` or `StorageTexture*<…>`. We use the declaration node
 * because TS doesn't preserve `aliasSymbol` reliably for type
 * aliases that resolve to intersection types — the brand info gets
 * unwrapped before the type queries can see it.
 */
function declaredTypeRef(sym: ts.Symbol | undefined): ts.TypeReferenceNode | undefined {
  const decl = sym?.declarations?.[0];
  if (!decl) return undefined;
  if (ts.isVariableDeclaration(decl) || ts.isParameter(decl) || ts.isPropertySignature(decl)) {
    if (decl.type && ts.isTypeReferenceNode(decl.type) && ts.isIdentifier(decl.type.typeName)) {
      return decl.type;
    }
  }
  return undefined;
}

function declaredTypeName(sym: ts.Symbol | undefined): string | undefined {
  const ref = declaredTypeRef(sym);
  if (!ref) return undefined;
  return (ref.typeName as ts.Identifier).text;
}

export function isStorageBufferType(t: ts.Type, sym?: ts.Symbol): boolean {
  if (aliasName(t) === "Storage") return true;
  if (sym && declaredTypeName(sym) === "Storage") return true;
  return false;
}

export function unwrapStorageBuffer(t: ts.Type, checker: ts.TypeChecker, sym?: ts.Symbol): {
  layout: IRType;
  access: "read" | "read_write";
} | undefined {
  // Prefer the alias path (cheaper); fall back to the declaration node.
  const aliasArgs = t.aliasTypeArguments;
  if (aliasName(t) === "Storage" && aliasArgs && aliasArgs[0]) {
    const layout = tsTypeToIR(aliasArgs[0], checker);
    if (!layout) return undefined;
    const access = (stringLiteral(aliasArgs[1]) as "read" | "read_write" | undefined) ?? "read_write";
    return { layout, access };
  }
  const ref = declaredTypeRef(sym);
  if (!ref || (ref.typeName as ts.Identifier).text !== "Storage") return undefined;
  const args = ref.typeArguments;
  if (!args || args.length < 1) return undefined;
  const layoutType = checker.getTypeAtLocation(args[0]!);
  const layout = tsTypeToIR(layoutType, checker);
  if (!layout) return undefined;
  let access: "read" | "read_write" = "read_write";
  if (args[1]) {
    const accessType = checker.getTypeAtLocation(args[1]);
    const v = stringLiteral(accessType);
    if (v === "read" || v === "read_write") access = v;
  }
  return { layout, access };
}

export function isStorageTextureType(t: ts.Type, sym?: ts.Symbol): boolean {
  const a = aliasName(t);
  if (a && a in STORAGE_TEXTURE_KIND) return true;
  const d = declaredTypeName(sym);
  if (d && d in STORAGE_TEXTURE_KIND) return true;
  return false;
}

export function unwrapStorageTexture(t: ts.Type, checker: ts.TypeChecker, sym?: ts.Symbol): IRType | undefined {
  const aliasArgs = t.aliasTypeArguments;
  const aName = aliasName(t);
  if (aName && aName in STORAGE_TEXTURE_KIND && aliasArgs && aliasArgs.length >= 1) {
    return makeStorageTexture(aName, aliasArgs[0]!, aliasArgs[1], () => undefined);
  }
  const ref = declaredTypeRef(sym);
  if (!ref) return undefined;
  const name = (ref.typeName as ts.Identifier).text;
  if (!(name in STORAGE_TEXTURE_KIND)) return undefined;
  const args = ref.typeArguments;
  if (!args || args.length < 1) return undefined;
  const fmtType = checker.getTypeAtLocation(args[0]!);
  const accType = args[1] ? checker.getTypeAtLocation(args[1]) : undefined;
  return makeStorageTexture(name, fmtType, accType, () => undefined);
}

function makeStorageTexture(
  name: string,
  fmtType: ts.Type,
  accType: ts.Type | undefined,
  _unused: () => undefined,
): IRType | undefined {
  const format = stringLiteral(fmtType) as StorageTextureFormat | undefined;
  if (!format) return undefined;
  const access = (stringLiteral(accType) as StorageAccess | undefined) ?? "write";
  const kind = STORAGE_TEXTURE_KIND[name]!;
  return {
    kind: "StorageTexture",
    target: kind.target,
    format,
    access,
    arrayed: kind.arrayed,
  };
}

function stringLiteral(t: ts.Type | undefined): string | undefined {
  if (!t) return undefined;
  if (t.flags & ts.TypeFlags.StringLiteral) return (t as ts.StringLiteralType).value;
  return undefined;
}

/**
 * Walk an object type's property list to build (name, IR type) pairs.
 * Used to turn `declare const u: { mvp: M44f; tint: V3f }` into a list
 * of uniform decls.
 */
export function objectFieldsToIR(
  t: ts.Type,
  checker: ts.TypeChecker,
): Array<{ name: string; type: IRType }> {
  const out: Array<{ name: string; type: IRType }> = [];
  const props = checker.getPropertiesOfType(t);
  for (const prop of props) {
    const decl = prop.valueDeclaration ?? prop.declarations?.[0];
    if (!decl) continue;
    const propType = checker.getTypeOfSymbolAtLocation(prop, decl);
    const ir = tsTypeToIR(propType, checker);
    if (!ir) continue;
    out.push({ name: prop.getName(), type: ir });
  }
  return out;
}

/**
 * `true` if the symbol's first declaration is in a `declare const ...`
 * (or other ambient context) — i.e. has no runtime value.
 */
export function isAmbientDeclaration(sym: ts.Symbol): boolean {
  const decls = sym.declarations ?? [];
  for (const d of decls) {
    // VariableDeclaration with `declare` ancestor.
    if (ts.isVariableDeclaration(d)) {
      const stmt = d.parent?.parent;
      if (stmt && ts.canHaveModifiers(stmt)) {
        const mods = ts.getModifiers(stmt);
        if (mods?.some((m) => m.kind === ts.SyntaxKind.DeclareKeyword)) return true;
      }
      // .d.ts file: every declaration is ambient.
      const sf = d.getSourceFile();
      if (sf.isDeclarationFile) return true;
    }
    if (d.getSourceFile().isDeclarationFile) return true;
  }
  return false;
}
