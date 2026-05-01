// Public API: parse a TypeScript source string + a list of named entry
// points, build a wombat.shader IR Module ready for the optimisation
// passes and emitters.

import type {
  EntryDecoration,
  EntryDef,
  EntryParameter,
  FunctionSignature,
  Module,
  Parameter,
  ParamDecoration,
  Stage,
  Type,
  ValueDef,
} from "../ir/index.js";
import { translateFunction, type TranslateOptions } from "./translate.js";
import { tryResolveTypeName } from "./types.js";
import ts from "typescript";

export interface EntryRequest {
  /** Top-level function name in the source. */
  readonly name: string;
  /** vertex / fragment / compute. */
  readonly stage: Stage;
  /** Optional explicit input parameter list; if omitted, inferred from the function's first parameter type. */
  readonly inputs?: readonly EntryParameter[];
  /** Optional explicit output list. If omitted, the entry has no declared outputs (caller is responsible for runtime binding via `WriteOutput` names). */
  readonly outputs?: readonly EntryParameter[];
  /**
   * Compute-stage workgroup dimensions. Falls back to a JSDoc tag
   * `@workgroupSize x [y [z]]` on the function declaration. If neither
   * is set, the WGSL emitter defaults to `(1, 1, 1)`.
   */
  readonly workgroupSize?: readonly [number, number?, number?];
}

export interface ParseShaderInput {
  readonly source: string;
  readonly file?: string;
  readonly entries: readonly EntryRequest[];
  /**
   * Type information for free identifiers the frontend wouldn't be
   * able to resolve from the source alone (uniforms, samplers, storage
   * buffers declared at module level). The translator looks here when
   * it encounters a reference to a name that isn't in scope; without
   * this it defaults the type to `f32`, which breaks downstream
   * dispatch (e.g. `m.mul(v)` when `m` is mat4 should produce
   * MulMatVec, not scalar Mul).
   */
  readonly externalTypes?: ReadonlyMap<string, Type>;
  /** Closure-captured identifiers; see TranslateOptions for semantics. */
  readonly closureTypes?: ReadonlyMap<string, Type>;
  /** Uniform namespaces; see TranslateOptions for semantics. */
  readonly uniformNamespaces?: ReadonlyMap<string, ReadonlyMap<string, Type>>;
  /**
   * Top-level helper functions to translate alongside the entries.
   * Each named helper becomes a `Function` ValueDef in the output
   * Module; calls to a helper from any entry / other-helper body
   * resolve to a properly-typed `Call(FunctionRef)`. Helpers may
   * reference uniforms / closures the same way entries do.
   */
  readonly helpers?: readonly string[];
}

export function parseShader(input: ParseShaderInput): Module {
  const opt: TranslateOptions = { source: input.source };
  if (input.file !== undefined) (opt as { file?: string }).file = input.file;
  if (input.externalTypes) (opt as { externalTypes?: ReadonlyMap<string, Type> }).externalTypes = input.externalTypes;
  if (input.closureTypes) (opt as { closureTypes?: ReadonlyMap<string, Type> }).closureTypes = input.closureTypes;
  if (input.uniformNamespaces) (opt as { uniformNamespaces?: ReadonlyMap<string, ReadonlyMap<string, Type>> }).uniformNamespaces = input.uniformNamespaces;
  // Pre-compute helper-function signatures so calls from entries
  // (and from other helpers) lower to typed `Call(FunctionRef)`
  // rather than the unresolved-stub fallback.
  const helperSignatures = extractHelperSignatures(input);
  if (helperSignatures.size > 0) {
    (opt as { helperSignatures?: ReadonlyMap<string, FunctionSignature> }).helperSignatures = helperSignatures;
  }
  const values: ValueDef[] = [];
  for (const req of input.entries) {
    const fn = translateFunction(opt, req.name);
    const derived = deriveParamShape(input, req);
    // For compute entries with a `ComputeBuiltins` parameter, the
    // builtins go into `arguments`, not `inputs`. When the caller
    // supplies an explicit `req.inputs` we still need any
    // FragmentBuiltinIn / VertexBuiltinIn members the body uses (via
    // a second function parameter) — merge them on top so the body's
    // `b.fragCoord` reads have a corresponding @builtin parameter.
    const baseInputs = req.inputs ?? derived.inputs;
    const builtinDerived = derived.inputs.filter((p) => p.decorations.some((d) => d.kind === "Builtin"));
    const inputs = req.inputs !== undefined && builtinDerived.length > 0
      ? mergeBuiltinInputs(baseInputs, builtinDerived)
      : baseInputs;
    const entryArgs = derived.arguments;
    const decorations: EntryDecoration[] = [];
    const ws = req.workgroupSize ?? derived.workgroupSize;
    if (ws) {
      const [x, y, z] = ws;
      decorations.push({
        kind: "WorkgroupSize",
        x,
        ...(y !== undefined ? { y } : {}),
        ...(z !== undefined ? { z } : {}),
      });
    }
    const entry: EntryDef = {
      name: req.name,
      stage: req.stage,
      inputs,
      outputs: req.outputs ?? [],
      arguments: entryArgs,
      returnType: fn.returnType,
      body: fn.body,
      decorations,
    };
    values.push({ kind: "Entry", entry });
  }
  // Translate every named helper. Helpers come before entries in the
  // module's value list so the WGSL/GLSL emitters see the function
  // declaration before it's called (forward declarations work in
  // both targets, but ordered emit is conventional).
  if (input.helpers && input.helpers.length > 0) {
    const helperValues: ValueDef[] = [];
    for (const name of input.helpers) {
      const fn = translateFunction(opt, name);
      const sig = helperSignatures.get(name);
      if (!sig) continue;
      helperValues.push({
        kind: "Function",
        signature: sig,
        body: fn.body,
      });
    }
    return { types: [], values: [...helperValues, ...values] };
  }
  return { types: [], values };
}

// ─────────────────────────────────────────────────────────────────────
// Helper-function signature extraction
// ─────────────────────────────────────────────────────────────────────

function extractHelperSignatures(input: ParseShaderInput): Map<string, FunctionSignature> {
  const out = new Map<string, FunctionSignature>();
  if (!input.helpers || input.helpers.length === 0) return out;
  const sf = ts.createSourceFile(input.file ?? "<input>", input.source, ts.ScriptTarget.ES2022, true);
  for (const name of input.helpers) {
    const decl = findHelperDecl(sf, name);
    if (!decl) {
      throw new Error(
        `parseShader: helper "${name}" not found at top level of ${input.file ?? "<input>"}`,
      );
    }
    const params: Parameter[] = [];
    for (const p of decl.parameters) {
      if (!ts.isIdentifier(p.name)) continue;
      const t: Type = p.type ? typeFromHelperParam(p.type) ?? Tvoid : Tvoid;
      params.push({ name: p.name.text, type: t, modifier: "in" });
    }
    const returnType: Type = decl.type ? typeFromHelperParam(decl.type) ?? Tvoid : Tvoid;
    out.set(name, { name, returnType, parameters: params });
  }
  return out;
}

function findHelperDecl(
  sf: ts.SourceFile,
  name: string,
): { parameters: readonly ts.ParameterDeclaration[]; type: ts.TypeNode | undefined } | undefined {
  for (const node of sf.statements) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) {
      return { parameters: node.parameters, type: node.type };
    }
    if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (!ts.isIdentifier(d.name) || d.name.text !== name) continue;
        const init = d.initializer;
        if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
          return { parameters: init.parameters, type: init.type };
        }
      }
    }
  }
  return undefined;
}

function typeFromHelperParam(node: ts.TypeNode): Type | undefined {
  if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
    return tryResolveTypeName(node.typeName.text);
  }
  if (node.kind === ts.SyntaxKind.NumberKeyword) return { kind: "Float", width: 32 };
  if (node.kind === ts.SyntaxKind.BooleanKeyword) return { kind: "Bool" };
  if (node.kind === ts.SyntaxKind.VoidKeyword)    return { kind: "Void" };
  return undefined;
}

const Tvoid: Type = { kind: "Void" };

interface DerivedShape {
  readonly inputs: readonly EntryParameter[];
  readonly arguments: readonly EntryParameter[];
  readonly workgroupSize?: readonly [number, number?, number?];
}

/**
 * Inspect the first parameter of the source function declaration plus
 * any leading JSDoc, and derive:
 *   - `inputs`:    flat list from a TypeLiteral (FShade pattern).
 *   - `arguments`: per-stage builtins from a `ComputeBuiltins` /
 *                  `VertexBuiltinIn` / `FragmentBuiltinIn` reference.
 *   - `workgroupSize`: parsed from a `@workgroupSize` JSDoc tag.
 */
function deriveParamShape(
  input: ParseShaderInput,
  req: EntryRequest,
): DerivedShape {
  const source = ts.createSourceFile(input.file ?? "<input>", input.source, ts.ScriptTarget.ES2022, true);
  let fnNode: ts.Node | undefined;
  let firstParamType: ts.TypeNode | undefined;
  let secondParamType: ts.TypeNode | undefined;
  ts.forEachChild(source, (node) => {
    if (fnNode) return;
    if (ts.isFunctionDeclaration(node) && node.name?.text === req.name) {
      fnNode = node;
      firstParamType = node.parameters[0]?.type;
      secondParamType = node.parameters[1]?.type;
    } else if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (ts.isIdentifier(d.name) && d.name.text === req.name && d.initializer) {
          if (ts.isArrowFunction(d.initializer) || ts.isFunctionExpression(d.initializer)) {
            fnNode = node;
            firstParamType = d.initializer.parameters[0]?.type;
            secondParamType = d.initializer.parameters[1]?.type;
          }
        }
      }
    }
  });
  const ws = fnNode ? parseWorkgroupSizeJsdoc(fnNode) : undefined;
  // Second-param builtin record (e.g. `b: FragmentBuiltinIn`): every
  // member with a `BuiltinName` semantic surfaces as an entry input
  // with `kind: "Builtin"`, even though the body's reference is via
  // `b.member`. Without this, the emitter has no parameter declared
  // for the builtin and the WGSL refers to an unbound name like
  // `position.z` — composeStages's fusion magnified the surprise but
  // the underlying gap is here.
  const secondaryBuiltins: EntryParameter[] = [];
  if (secondParamType && ts.isTypeReferenceNode(secondParamType) && ts.isIdentifier(secondParamType.typeName)) {
    const shape = secondaryBuiltinShape(secondParamType.typeName.text);
    if (shape) {
      for (const [name, spec] of Object.entries(shape)) {
        secondaryBuiltins.push(paramOfBuiltin(name, spec.type, spec.semantic));
      }
    }
  }
  if (!firstParamType) return { inputs: secondaryBuiltins, arguments: [], ...(ws ? { workgroupSize: ws } : {}) };
  if (ts.isTypeReferenceNode(firstParamType) && ts.isIdentifier(firstParamType.typeName)) {
    const args = builtinArgumentsForRecord(firstParamType.typeName.text);
    if (args) return { inputs: secondaryBuiltins, arguments: args, ...(ws ? { workgroupSize: ws } : {}) };
  }
  return {
    inputs: [...inputsFromTypeNode(firstParamType), ...secondaryBuiltins],
    arguments: [],
    ...(ws ? { workgroupSize: ws } : {}),
  };
}

const COMPUTE_BUILTIN_PARAMS: readonly EntryParameter[] = [
  paramOfBuiltin("globalInvocationId", { kind: "Vector", element: { kind: "Int", signed: false, width: 32 }, dim: 3 }, "global_invocation_id"),
  paramOfBuiltin("localInvocationId",  { kind: "Vector", element: { kind: "Int", signed: false, width: 32 }, dim: 3 }, "local_invocation_id"),
  paramOfBuiltin("localInvocationIndex", { kind: "Int", signed: false, width: 32 }, "local_invocation_index"),
  paramOfBuiltin("workgroupId",        { kind: "Vector", element: { kind: "Int", signed: false, width: 32 }, dim: 3 }, "workgroup_id"),
  paramOfBuiltin("numWorkgroups",      { kind: "Vector", element: { kind: "Int", signed: false, width: 32 }, dim: 3 }, "num_workgroups"),
];

function paramOfBuiltin(
  name: string,
  type: Type,
  semantic: import("../ir/index.js").BuiltinSemantic,
): EntryParameter {
  return {
    name,
    type,
    semantic,
    decorations: [{ kind: "Builtin", value: semantic }],
  };
}

function builtinArgumentsForRecord(typeName: string): readonly EntryParameter[] | undefined {
  if (typeName === "ComputeBuiltins") return COMPUTE_BUILTIN_PARAMS;
  // Vertex/fragment builtins are surfaced via inputs (decorations carry
  // `Builtin` so the emitter handles them through the input struct).
  return undefined;
}

const VERTEX_BUILTIN_IN_SHAPE: Record<string, { type: Type; semantic: import("../ir/index.js").BuiltinSemantic }> = {
  vertexIndex:   { type: { kind: "Int", signed: false, width: 32 }, semantic: "vertex_index" },
  instanceIndex: { type: { kind: "Int", signed: false, width: 32 }, semantic: "instance_index" },
};
const FRAGMENT_BUILTIN_IN_SHAPE: Record<string, { type: Type; semantic: import("../ir/index.js").BuiltinSemantic }> = {
  fragCoord:    { type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 }, semantic: "position" },
};

function secondaryBuiltinShape(typeName: string): Record<string, { type: Type; semantic: import("../ir/index.js").BuiltinSemantic }> | undefined {
  if (typeName === "VertexBuiltinIn")   return VERTEX_BUILTIN_IN_SHAPE;
  if (typeName === "FragmentBuiltinIn") return FRAGMENT_BUILTIN_IN_SHAPE;
  return undefined;
}

function mergeBuiltinInputs(
  base: readonly EntryParameter[],
  builtins: readonly EntryParameter[],
): EntryParameter[] {
  const seen = new Set(base.map((p) => p.name));
  const out = [...base];
  for (const p of builtins) {
    if (!seen.has(p.name)) out.push(p);
  }
  return out;
}

function parseWorkgroupSizeJsdoc(node: ts.Node): readonly [number, number?, number?] | undefined {
  const jsDocs = (ts.getJSDocTags(node) ?? []).filter((t) => t.tagName.text === "workgroupSize");
  for (const tag of jsDocs) {
    const text = typeof tag.comment === "string"
      ? tag.comment
      : (tag.comment ?? []).map((c) => (typeof c === "string" ? c : c.text)).join(" ");
    const nums = text.split(/[\s,]+/).filter(Boolean).map((s) => parseInt(s, 10)).filter((n) => Number.isFinite(n));
    if (nums.length >= 1) return [nums[0]!, nums[1], nums[2]] as [number, number?, number?];
  }
  return undefined;
}

function inputsFromTypeNode(node: ts.TypeNode): EntryParameter[] {
  if (ts.isTypeLiteralNode(node)) {
    const out: EntryParameter[] = [];
    let nextLoc = 0;
    for (const member of node.members) {
      if (!ts.isPropertySignature(member)) continue;
      if (!member.name || !ts.isIdentifier(member.name)) continue;
      if (!member.type) continue;
      const t = typeFromNodeShallow(member.type);
      if (!t) continue;
      const decorations: ParamDecoration[] = [{ kind: "Location", value: nextLoc++ }];
      out.push({
        name: member.name.text,
        type: t,
        semantic: capitalise(member.name.text),
        decorations,
      });
    }
    return out;
  }
  return [];
}

function typeFromNodeShallow(node: ts.TypeNode): Type | undefined {
  if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)) {
    return tryResolveTypeName(node.typeName.text);
  }
  if (node.kind === ts.SyntaxKind.NumberKeyword) {
    return { kind: "Float", width: 32 };
  }
  if (node.kind === ts.SyntaxKind.BooleanKeyword) {
    return { kind: "Bool" };
  }
  return undefined;
}

function capitalise(s: string): string {
  if (s.startsWith("a_")) return capitalise(s.slice(2));
  if (s.startsWith("v_")) return capitalise(s.slice(2));
  return s.length > 0 ? s[0]!.toUpperCase() + s.slice(1) : s;
}
