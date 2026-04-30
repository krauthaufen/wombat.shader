// Public API: parse a TypeScript source string + a list of named entry
// points, build a wombat.shader IR Module ready for the optimisation
// passes and emitters.

import type {
  EntryDef,
  EntryParameter,
  Module,
  ParamDecoration,
  Stage,
  Type,
  ValueDef,
} from "@aardworx/wombat.shader-ir";
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
}

export function parseShader(input: ParseShaderInput): Module {
  const opt: TranslateOptions = { source: input.source };
  if (input.file !== undefined) (opt as { file?: string }).file = input.file;
  if (input.externalTypes) (opt as { externalTypes?: ReadonlyMap<string, Type> }).externalTypes = input.externalTypes;
  const values: ValueDef[] = [];
  for (const req of input.entries) {
    const fn = translateFunction(opt, req.name);
    const inputs = req.inputs ?? deriveInputsFromFirstParam(input, req);
    const entry: EntryDef = {
      name: req.name,
      stage: req.stage,
      inputs,
      outputs: req.outputs ?? [],
      arguments: [],
      returnType: fn.returnType,
      body: fn.body,
      decorations: [],
    };
    values.push({ kind: "Entry", entry });
  }
  return { types: [], values };
}

/**
 * Inspect the first parameter of the source function declaration. If
 * its type is a single named type whose name we recognise as a vector
 * or scalar, treat that as the entry's only input. If the parameter
 * type is a TypeLiteral (object shape), treat each member as a
 * separate input parameter (the canonical FShade pattern of an input
 * record).
 */
function deriveInputsFromFirstParam(
  input: ParseShaderInput,
  req: EntryRequest,
): readonly EntryParameter[] {
  const source = ts.createSourceFile(input.file ?? "<input>", input.source, ts.ScriptTarget.ES2022, true);
  let firstParamType: ts.TypeNode | undefined;
  ts.forEachChild(source, (node) => {
    if (firstParamType) return;
    if (ts.isFunctionDeclaration(node) && node.name?.text === req.name) {
      firstParamType = node.parameters[0]?.type;
    } else if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (ts.isIdentifier(d.name) && d.name.text === req.name && d.initializer) {
          if (ts.isArrowFunction(d.initializer) || ts.isFunctionExpression(d.initializer)) {
            firstParamType = d.initializer.parameters[0]?.type;
          }
        }
      }
    }
  });
  if (!firstParamType) return [];
  return inputsFromTypeNode(firstParamType);
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
