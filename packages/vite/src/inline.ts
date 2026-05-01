// Inline-shader transform.
//
// Scans a TS/TSX source for calls to the build-time markers
// `vertex(arrow)` / `fragment(arrow)` / `compute(arrow)` from
// `@aardworx/wombat.shader-runtime`, walks each inline arrow
// function with the wombat.shader frontend, and replaces each call
// site with a `__wombat_stage(<frozen IR template>, { closures … })`
// expression that the runtime turns into a `Stage`.
//
// Closure captures (free identifiers in the arrow body that aren't
// shader intrinsics, vector / matrix / sampler imports, or marker
// imports) get encoded as `ReadInput("Closure", name, type)` in the
// IR. The plugin resolves each capture's IR type from one of two
// sources:
//
//   1. An explicit annotation on its declaration:
//        const tint: V3f = ...;
//   2. A `new V3f(...)` / `new M44f(...)` initializer.
//
// Anything else fails with a clear diagnostic asking the user to
// annotate. (Full TS type-checker integration is a follow-up.)

import ts from "typescript";
import MagicString from "magic-string";
import type { EntryRequest } from "@aardworx/wombat.shader/frontend";
import { parseShader, tryResolveTypeName } from "@aardworx/wombat.shader/frontend";
import { liftReturns } from "@aardworx/wombat.shader/passes";
import {
  hashModule,
  type EntryDef,
  type EntryParameter,
  type Expr,
  type Module,
  type ParamDecoration,
  type Stage as StageKind,
  type Stmt,
  type Type,
  type UniformDecl,
  type ValueDef,
} from "@aardworx/wombat.shader/ir";
import { TypeResolver } from "./typeResolver.js";
import {
  isAmbientDeclaration, isAvalType,
  isStorageBufferType, isStorageTextureType,
  objectFieldsToIR, tsTypeToIR,
  unwrapStorageBuffer, unwrapStorageTexture,
} from "./typeMapper.js";

const MARKER_NAMES = new Set<MarkerName>(["vertex", "fragment", "compute"]);
type MarkerName = "vertex" | "fragment" | "compute";

const SHIPPED_TYPE_NAMES = new Set([
  // primitives + scalar shorthands
  "i32", "u32", "f32", "bool", "number", "boolean", "void",
  // vectors
  "V2b","V3b","V4b","V2i","V3i","V4i","V2u","V3u","V4u","V2f","V3f","V4f",
  // matrices
  "M22f","M33f","M44f","M23f","M24f","M32f","M34f","M42f","M43f",
  // samplers
  "Sampler2D","Sampler3D","SamplerCube","Sampler2DArray","SamplerCubeArray",
  "ISampler2D","USampler2D","Sampler2DShadow",
  "Sampler2DMS","ISampler2DMS","USampler2DMS",
  // constructors / intrinsics: see SHIPPED_INTRINSIC_NAMES below.
]);

const SHIPPED_INTRINSIC_NAMES = new Set([
  // ctors
  "vec2","vec3","vec4","ivec2","ivec3","ivec4","uvec2","uvec3","uvec4",
  "mat2","mat3","mat4","mat2x2","mat3x3","mat4x4",
  "mat2x3","mat2x4","mat3x2","mat3x4","mat4x2","mat4x3",
  // pure math
  "sin","cos","tan","asin","acos","atan","atan2",
  "exp","exp2","log","log2","pow","sqrt","inversesqrt",
  "abs","sign","floor","ceil","round","fract","mod",
  "min","max","clamp","mix","step","smoothstep",
  "length","distance","dot","cross","normalize","reflect","refract","faceforward",
  "dFdx","dFdy","fwidth","any","all",
  "texture","textureLod","textureGrad","texelFetch","textureSize",
  "textureLoad","textureStore","textureGather","textureSampleCompare",
  // atomics + barriers + misc
  "atomicLoad","atomicStore","atomicAdd","atomicSub","atomicMin","atomicMax",
  "atomicAnd","atomicOr","atomicXor","atomicExchange","atomicCompareExchangeWeak",
  "workgroupBarrier","storageBarrier","discard",
]);

const STAGE_BUILTIN_RECORDS = new Set([
  "ComputeBuiltins","VertexBuiltinIn","FragmentBuiltinIn",
]);

// JS / DOM globals we should never treat as captures.
const GLOBAL_NAMES = new Set([
  "console","Math","Number","String","Boolean","Array","Object","JSON",
  "undefined","null","true","false","NaN","Infinity",
]);

export interface TransformResult {
  readonly code: string;
  /** v3 source map (returned via Vite's `transform` hook). */
  readonly map: object | null;
}

/**
 * Transform a TS/TSX source string. Returns `null` if the file
 * doesn't reference any inline-shader markers. Throws with a clear
 * diagnostic on a malformed marker call.
 *
 * The optional `resolver` provides cross-file type resolution. With
 * it the plugin recognises ambient `declare const u: { … }` blocks
 * (in the same file, in `.d.ts`, or in any imported module) as
 * uniform bindings. Without it, free-identifier types fall back to
 * the in-file annotation/`new V*f` heuristic and ambient detection
 * is skipped.
 */
export function transformInlineShaders(
  source: string,
  fileName: string,
  resolver?: TypeResolver,
): TransformResult | null {
  // Cheap reject — most files don't contain shader markers.
  if (!/\b(vertex|fragment|compute)\s*\(/.test(source)) return null;
  if (!sourceImportsRuntime(source)) return null;

  // Update the resolver's in-memory copy so the type checker sees the
  // same source the plugin is walking. Then prefer its source file
  // (the one TS will type-check) over a fresh AST parse.
  let sf: ts.SourceFile;
  let checker: ts.TypeChecker | undefined;
  if (resolver) {
    resolver.setFile(fileName, source);
    sf = resolver.getSourceFile(fileName) ?? ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true);
    checker = resolver.getChecker();
  } else {
    sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true);
  }
  const moduleConsts = collectModuleConsts(sf);

  const replacements: Array<{ start: number; end: number; text: string; marker: MarkerName }> = [];

  function visit(node: ts.Node): void {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)
        && MARKER_NAMES.has(node.expression.text as MarkerName)) {
      const marker = node.expression.text as MarkerName;
      const text = transformMarkerCall(
        node,
        marker,
        sf,
        moduleConsts,
        fileName,
        checker,
      );
      replacements.push({ start: node.getStart(sf), end: node.getEnd(), text, marker });
      // Don't descend — the body is consumed by the frontend.
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(sf);

  if (replacements.length === 0) return null;

  // Apply replacements via magic-string so we get a v3 source map for
  // free. Each marker call's region is overwritten in place; the map
  // points each output character back at its origin span (which for
  // the entire __wombat_stage(...) replacement is the original
  // marker call site).
  const ms = new MagicString(source);
  for (const r of replacements) {
    ms.overwrite(r.start, r.end, r.text);
  }
  // Auto-prepend the runtime imports for whichever markers actually
  // appeared. Compute uses its own factory (`computeShader`) so it
  // returns a `ComputeShader`, distinct from the graphics-stage `Effect`.
  const needStage = replacements.some(r => r.marker !== "compute");
  const needCompute = replacements.some(r => r.marker === "compute");
  const imports: string[] = [];
  if (needStage) imports.push("stage as __wombat_stage");
  if (needCompute) imports.push("computeShader as __wombat_compute");
  ms.prepend(`import { ${imports.join(", ")} } from "@aardworx/wombat.shader";\n`);

  return {
    code: ms.toString(),
    map: ms.generateMap({
      source: fileName,
      includeContent: true,
      hires: true,
    }) as unknown as object,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Marker-call rewriting
// ─────────────────────────────────────────────────────────────────────

interface ModuleConst {
  readonly type?: Type;
}

function transformMarkerCall(
  call: ts.CallExpression,
  marker: MarkerName,
  sf: ts.SourceFile,
  moduleConsts: ReadonlyMap<string, ModuleConst>,
  fileName: string,
  checker: ts.TypeChecker | undefined,
): string {
  const firstArg = call.arguments[0];
  let arrow: ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration | undefined;
  if (firstArg && (ts.isArrowFunction(firstArg) || ts.isFunctionExpression(firstArg))) {
    arrow = firstArg;
  } else if (firstArg && ts.isIdentifier(firstArg)) {
    // Function reference — look up a top-level declaration in the same
    // source file. Cross-file resolution would need TS type-checker
    // integration; for now we restrict to same-file references.
    arrow = findTopLevelFunction(sf, firstArg.text);
    if (!arrow) {
      throw error(
        sf, firstArg,
        `wombat.shader: \`${marker}(${firstArg.text})\` — could not find a top-level function declaration ` +
        `or \`const ${firstArg.text} = (…) => …\` in this file.`,
      );
    }
  } else {
    throw error(
      sf, call,
      `wombat.shader: \`${marker}(...)\` expects a literal arrow function, function expression, ` +
      `or an identifier referring to a top-level function declaration in the same file.`,
    );
  }

  // Discover transitively-called helper functions starting from this
  // arrow body. For each helper we'll synthesise into the shader
  // source, and its free identifiers contribute to the same uniform
  // / closure classification as the arrow itself.
  const helpers = discoverHelpers(arrow, sf, checker);

  // Collect free identifiers (identifiers used inside the arrow that
  // are not bound by the arrow's own parameters / local lets / consts,
  // and are not shipped names from @aardworx/wombat.shader-types).
  // Also collect from helpers — their free identifiers must be
  // resolved the same way; helpers reading `u.tint` or `time` should
  // surface those uniform decls in the parent module.
  const captureNodes = collectCaptureUses(arrow, sf);
  for (const h of helpers) {
    captureNodes.push(...collectCaptureUses(h, sf));
  }
  // Helper names themselves aren't captures.
  const helperNames = new Set(helpers.map((h) => helperName(h)));
  const filteredCaptureNodes = captureNodes.filter((n) => !helperNames.has(n.text));
  const captureNames = uniqueNames(filteredCaptureNodes);

  // Classify each captured name into one of three buckets:
  //   - uniform namespace  (ambient, object-typed: `declare const u: {…}`)
  //   - loose uniform       (ambient, scalar/vector-typed: `declare const time: number`)
  //   - closure capture     (regular runtime declaration)
  // The TS type checker drives this when available; without it we
  // fall back to the same-file annotation/`new V*f` heuristic and
  // skip ambient detection entirely.
  const closureTypes = new Map<string, Type>();
  const looseUniforms = new Map<string, Type>();
  const avalUniforms = new Map<string, Type>();
  const samplerCaptures = new Map<string, Type>();
  const storageBufferCaptures = new Map<string, { layout: Type; access: "read" | "read_write" }>();
  const storageTextureCaptures = new Map<string, Type>();
  const uniformNamespaces = new Map<string, Map<string, Type>>();

  for (const name of captureNames) {
    const useNode = filteredCaptureNodes.find((n) => n.text === name)!;
    const classification = classifyFreeIdentifier(useNode, name, checker, moduleConsts);
    switch (classification.kind) {
      case "uniform-namespace":
        uniformNamespaces.set(name, classification.fields);
        break;
      case "uniform-loose":
        looseUniforms.set(name, classification.type);
        break;
      case "uniform-aval":
        avalUniforms.set(name, classification.type);
        break;
      case "uniform-sampler":
        samplerCaptures.set(name, classification.type);
        break;
      case "storage-buffer":
        storageBufferCaptures.set(name, { layout: classification.layout, access: classification.access });
        break;
      case "storage-texture":
        storageTextureCaptures.set(name, classification.type);
        break;
      case "closure":
        closureTypes.set(name, classification.type);
        break;
      case "unresolved":
        throw error(
          sf, useNode,
          `wombat.shader: cannot determine type of "${name}". ` +
          `Annotate its declaration (e.g. \`const ${name}: V3f = ...\`) ` +
          `or declare it as a uniform (\`declare const ${name}: V3f\`).`,
        );
    }
  }

  // Synthesise a self-contained shader source so the frontend has
  // exactly the function it needs, plus any module-level consts the
  // user referenced (those are inlined by the frontend's
  // `collectModuleConsts`, so we only need the function itself).
  // WGSL forbids identifiers starting with two underscores, so the
  // synthesised entry-point name has to use a single underscore prefix.
  const fnName = `wombat_${marker}_${call.getStart(sf).toString(16)}`;
  const helperSources: string[] = [];
  const helperList: string[] = [];
  for (const h of helpers) {
    const name = helperName(h);
    helperSources.push(synthesiseHelperSource(h, name, sf));
    helperList.push(name);
  }
  // Type-fallback chain for the arrow's first parameter and return:
  //   1. The TS type-checker's contextual type, when available.
  //      Covers generic args (`vertex<I,O>(...)`), explicit param
  //      annotations, body-driven inference — TS handles all three
  //      uniformly via the same `checker.getContextualType`.
  //   2. The marker's typeArguments[0] / [1] read straight from the
  //      AST. Used when no resolver was passed in (e.g. tests).
  //   3. None — falls back to whatever was on the lambda itself.
  const inferred = checker !== undefined ? inferArrowSignature(checker, arrow) : undefined;
  const inputTypeArg = inferred?.input ?? call.typeArguments?.[0];
  const outputTypeArg = inferred?.output ?? call.typeArguments?.[1];
  const fnSource = [
    synthesiseFunctionSource(arrow, fnName, sf, inputTypeArg, outputTypeArg),
    ...helperSources,
  ].join("\n\n");

  const stageKind: StageKind = marker;
  const entry: EntryRequest = { name: fnName, stage: stageKind };

  let module: Module;
  try {
    // Loose uniforms, aval-typed captures, sampler captures,
    // storage buffers, and storage textures all ride the existing
    // externalTypes path: free identifiers matching one of these
    // names lower to ReadInput("Uniform", name, type). The
    // classification difference shows up in (1) the kind of
    // ValueDef emitted (Uniform / Sampler / StorageBuffer) and
    // (2) which getter map carries the runtime binding source.
    const storageBufferTypes = new Map<string, Type>(
      [...storageBufferCaptures.entries()].map(([name, sb]) => [name, sb.layout]),
    );
    const externalTypes = (looseUniforms.size + avalUniforms.size + samplerCaptures.size + storageBufferCaptures.size + storageTextureCaptures.size) > 0
      ? new Map<string, Type>([
          ...looseUniforms, ...avalUniforms, ...samplerCaptures,
          ...storageBufferTypes, ...storageTextureCaptures,
        ])
      : undefined;
    module = parseShader({
      source: fnSource,
      file: fileName,
      entries: [entry],
      closureTypes,
      ...(externalTypes ? { externalTypes } : {}),
      ...(uniformNamespaces.size > 0 ? { uniformNamespaces } : {}),
      ...(helperList.length > 0 ? { helpers: helperList } : {}),
    });
  } catch (e) {
    let msg = e instanceof Error ? e.message : String(e);
    if (e && typeof e === "object" && "diagnostics" in e) {
      const ds = (e as { diagnostics: ReadonlyArray<{ message: string }> }).diagnostics;
      if (ds && ds.length > 0) {
        msg += " — " + ds.map((d) => d.message).join("; ");
      }
    }
    throw error(sf, call, `wombat.shader: frontend rejected \`${marker}\` body — ${msg}`);
  }

  // Add Uniform ValueDef(s) for every uniform reference, Sampler
  // ValueDef(s) for sampler / storage-texture captures, and
  // StorageBuffer ValueDef(s) for `Storage<…>` captures.
  const allLoose = new Map<string, Type>([...looseUniforms, ...avalUniforms]);
  module = withUniformDecls(module, uniformNamespaces, allLoose);
  module = withSamplerDecls(module, samplerCaptures);
  module = withStorageTextureDecls(module, storageTextureCaptures);
  module = withStorageBufferDecls(module, storageBufferCaptures);

  // Walk the parsed entry body for the synthetic return-record carrier
  // and synthesise outputs from its keys + expression types. Then run
  // `liftReturns` so the lift happens before JSON serialisation —
  // the carrier's `_record` is a non-enumerable property and would
  // otherwise be stripped at stringify time, leaving the runtime with
  // a body that returns `Const(Null)` and writes nothing.
  module = withDerivedOutputs(module, fnName, marker);
  module = liftReturns(module);

  // Emit replacement.
  //
  //   __wombat_stage(
  //     <jsonModule>,
  //     <holes>,         // closure captures (specialized as constants)
  //     "<id>",          // build-time stable hash of the IR template
  //     <avalHoles>,     // aval-typed captures (uniform bindings)
  //   )
  //
  // The runtime hands `avalHoles` to the future rendering backend so
  // it can subscribe to each aval and write the GPU buffer slot when
  // the underlying value changes; here we just pass them through.
  const closureNames = [...closureTypes.keys()];
  const bindingNames = [
    ...avalUniforms.keys(), ...samplerCaptures.keys(),
    ...storageBufferCaptures.keys(), ...storageTextureCaptures.keys(),
  ];
  const holesObj = closureNames.length === 0
    ? "{}"
    : `{ ${closureNames.map((n) => `${n}: () => ${n}`).join(", ")} }`;
  const avalObj = bindingNames.length === 0
    ? "{}"
    : `{ ${bindingNames.map((n) => `${n}: () => ${n}`).join(", ")} }`;
  const id = hashModule(module);
  const moduleJson = JSON.stringify(module);
  const fn = marker === "compute" ? "__wombat_compute" : "__wombat_stage";
  return `${fn}(${moduleJson}, ${holesObj}, ${JSON.stringify(id)}, ${avalObj})`;
}

// ─────────────────────────────────────────────────────────────────────
// Free-identifier classification (uniform vs closure)
// ─────────────────────────────────────────────────────────────────────

type Classification =
  | { kind: "uniform-namespace"; fields: Map<string, Type> }
  | { kind: "uniform-loose"; type: Type }
  | { kind: "uniform-aval"; type: Type }
  | { kind: "uniform-sampler"; type: Type }
  | { kind: "storage-buffer"; layout: Type; access: "read" | "read_write" }
  | { kind: "storage-texture"; type: Type }
  | { kind: "closure"; type: Type }
  | { kind: "unresolved" };

function classifyFreeIdentifier(
  useNode: ts.Identifier,
  name: string,
  checker: ts.TypeChecker | undefined,
  moduleConsts: ReadonlyMap<string, ModuleConst>,
): Classification {
  if (checker) {
    const sym = checker.getSymbolAtLocation(useNode);
    if (sym) {
      const tsType = checker.getTypeOfSymbolAtLocation(sym, useNode);
      // `Storage<T, A>` capture → storage-buffer binding. The
      // `inferStorageAccess` pass narrows access mode based on body
      // usage; the value here is the user's declared bound (or the
      // `read_write` default).
      if (isStorageBufferType(tsType, sym)) {
        const sb = unwrapStorageBuffer(tsType, checker, sym);
        if (sb) return { kind: "storage-buffer", layout: sb.layout, access: sb.access };
      }
      // `StorageTexture*<F, A>` capture → storage-texture binding.
      if (isStorageTextureType(tsType, sym)) {
        const st = unwrapStorageTexture(tsType, checker, sym);
        if (st) return { kind: "storage-texture", type: st };
      }
      // `aval<T>` capture → uniform binding, regardless of where the
      // declaration lives. The runtime preserves the aval handle on
      // the Effect for the future rendering backend to subscribe to.
      if (isAvalType(tsType, checker)) {
        const inner = tsTypeToIR(tsType, checker);
        if (inner) return { kind: "uniform-aval", type: inner };
        return { kind: "unresolved" };
      }
      const ambient = isAmbientDeclaration(sym);
      if (ambient) {
        // Object-typed ambient → uniform namespace.
        const ir = tsTypeToIR(tsType, checker);
        if (!ir) {
          // Non-mappable type → walk fields, treat as namespace.
          const fields = objectFieldsToIR(tsType, checker);
          if (fields.length > 0) {
            const map = new Map<string, Type>();
            for (const f of fields) map.set(f.name, f.type);
            return { kind: "uniform-namespace", fields: map };
          }
          return { kind: "unresolved" };
        }
        // Scalar/vector/matrix-typed ambient → loose uniform.
        return { kind: "uniform-loose", type: ir };
      }
      // Runtime declaration. Sampler captures can't specialize as
      // constants — they're opaque texture handles. Route them to
      // the binding path same as avals, but emit a Sampler ValueDef
      // (not Uniform) so the emitter splits texture+sampler at the
      // legalise pass for WGSL.
      const ir = tsTypeToIR(tsType, checker);
      if (ir && ir.kind === "Sampler") return { kind: "uniform-sampler", type: ir };
      if (ir) return { kind: "closure", type: ir };
      // Type checker returned an unmappable type — fall through to
      // the heuristic for closures.
    }
  }
  // Fallback heuristic: consult moduleConsts (annotated decl or
  // `new V*f(...)` initializer in this file).
  const t = moduleConsts.get(name)?.type;
  if (t) return { kind: "closure", type: t };
  return { kind: "unresolved" };
}

function uniqueNames(nodes: readonly ts.Identifier[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const n of nodes) {
    if (seen.has(n.text)) continue;
    seen.add(n.text);
    out.push(n.text);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Uniform decl emission
// ─────────────────────────────────────────────────────────────────────

function withUniformDecls(
  module: Module,
  namespaces: Map<string, Map<string, Type>>,
  loose: Map<string, Type>,
): Module {
  const uniformDecls: UniformDecl[] = [];
  // Loose first so they precede grouped buffers.
  for (const [name, type] of loose) {
    uniformDecls.push({ name, type });
  }
  for (const [bufferName, fields] of namespaces) {
    for (const [name, type] of fields) {
      uniformDecls.push({ name, type, buffer: bufferName });
    }
  }
  if (uniformDecls.length === 0) return module;
  const uniformValue: ValueDef = { kind: "Uniform", uniforms: uniformDecls };
  return { ...module, values: [uniformValue, ...module.values] };
}

function withSamplerDecls(
  module: Module,
  samplers: Map<string, Type>,
): Module {
  if (samplers.size === 0) return module;
  // Assign sequential bindings starting from group 0, slot 0. The
  // user's draw layer maps `name` → bound texture/sampler at draw
  // time so absolute binding indices don't matter to the user.
  let slot = 0;
  const decls: ValueDef[] = [];
  for (const [name, type] of samplers) {
    decls.push({
      kind: "Sampler",
      binding: { group: 0, slot: slot++ },
      name,
      type,
    });
  }
  return { ...module, values: [...decls, ...module.values] };
}

function withStorageTextureDecls(
  module: Module,
  textures: Map<string, Type>,
): Module {
  if (textures.size === 0) return module;
  // Storage textures share the `Sampler` ValueDef shape — the type
  // discriminates StorageTexture from sampled Texture. Each gets
  // one binding slot.
  let slot = 0;
  const decls: ValueDef[] = [];
  for (const [name, type] of textures) {
    decls.push({
      kind: "Sampler",
      binding: { group: 0, slot: slot++ },
      name,
      type,
    });
  }
  return { ...module, values: [...decls, ...module.values] };
}

function withStorageBufferDecls(
  module: Module,
  buffers: Map<string, { layout: Type; access: "read" | "read_write" }>,
): Module {
  if (buffers.size === 0) return module;
  let slot = 0;
  const decls: ValueDef[] = [];
  for (const [name, { layout, access }] of buffers) {
    decls.push({
      kind: "StorageBuffer",
      binding: { group: 0, slot: slot++ },
      name,
      layout,
      access,
    });
  }
  return { ...module, values: [...decls, ...module.values] };
}

// ─────────────────────────────────────────────────────────────────────
// Derive entry outputs from the return-record carrier
// ─────────────────────────────────────────────────────────────────────

function withDerivedOutputs(module: Module, entryName: string, marker: MarkerName): Module {
  const values = module.values.map((v): ValueDef => {
    if (v.kind !== "Entry" || v.entry.name !== entryName) return v;
    const carrier = findReturnCarrier(v.entry.body);
    if (carrier !== undefined) {
      const outputs = synthesiseOutputs(carrier, marker);
      if (outputs.length === 0) return v;
      const entry: EntryDef = { ...v.entry, outputs };
      return { kind: "Entry", entry };
    }
    // Bare-value return: a `V4f` literal without a `{ … }` wrapper.
    // Vertex → gl_Position (@builtin position); fragment → outColor
    // (@location 0). Compute has no shorthand.
    const bareReturn = findBareReturnType(v.entry.body);
    if (bareReturn !== undefined && isVec4f(bareReturn)) {
      const fallback = bareV4fOutputFor(marker, bareReturn);
      if (fallback !== undefined) {
        return { kind: "Entry", entry: { ...v.entry, outputs: [fallback] } };
      }
    }
    return v;
  });
  return values === module.values ? module : { ...module, values };
}

function findBareReturnType(s: Stmt): Type | undefined {
  switch (s.kind) {
    case "ReturnValue": return s.value.type;
    case "Sequential":
    case "Isolated":
      for (const c of s.body) {
        const r = findBareReturnType(c);
        if (r !== undefined) return r;
      }
      return undefined;
    case "If":
      return findBareReturnType(s.then) ?? (s.else ? findBareReturnType(s.else) : undefined);
    default:
      return undefined;
  }
}

function isVec4f(t: Type): boolean {
  return t.kind === "Vector" && t.dim === 4 && t.element.kind === "Float";
}

/** The canonical default-output name for a bare-V4f return per stage. */
function bareV4fOutputFor(marker: MarkerName, type: Type): EntryParameter | undefined {
  if (marker === "vertex") {
    return {
      name: "gl_Position",
      type,
      semantic: "Position",
      decorations: [{ kind: "Builtin", value: "position" }],
    };
  }
  if (marker === "fragment") {
    return {
      name: "outColor",
      type,
      semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    };
  }
  return undefined;
}

function findReturnCarrier(s: Stmt): ReadonlyMap<string, Expr> | undefined {
  switch (s.kind) {
    case "ReturnValue": {
      const fields = (s.value as { _record?: ReadonlyMap<string, Expr> })._record;
      return fields;
    }
    case "Sequential":
    case "Isolated":
      for (const c of s.body) {
        const r = findReturnCarrier(c);
        if (r) return r;
      }
      return undefined;
    case "If":
      return findReturnCarrier(s.then) ?? (s.else ? findReturnCarrier(s.else) : undefined);
    default:
      return undefined;
  }
}

function synthesiseOutputs(
  fields: ReadonlyMap<string, Expr>,
  marker: MarkerName,
): EntryParameter[] {
  const out: EntryParameter[] = [];
  let nextLoc = 0;
  for (const [name, expr] of fields) {
    const decorations: ParamDecoration[] = [];
    const builtin = builtinFor(name, marker);
    if (builtin) {
      decorations.push({ kind: "Builtin", value: builtin });
    } else {
      decorations.push({ kind: "Location", value: nextLoc++ });
    }
    out.push({
      name,
      type: expr.type,
      semantic: capitalise(stripInterpolantPrefix(name)),
      decorations,
    });
  }
  return out;
}

function builtinFor(name: string, marker: MarkerName):
  | "position" | "frag_depth" | undefined {
  if (marker === "vertex" && (name === "gl_Position" || name === "position")) return "position";
  if (marker === "fragment" && name === "fragDepth") return "frag_depth";
  return undefined;
}

function stripInterpolantPrefix(name: string): string {
  if (name.startsWith("v_")) return name.slice(2);
  if (name.startsWith("a_")) return name.slice(2);
  return name;
}

function capitalise(s: string): string {
  return s.length > 0 ? s[0]!.toUpperCase() + s.slice(1) : s;
}

// ─────────────────────────────────────────────────────────────────────
// Free-identifier collection
// ─────────────────────────────────────────────────────────────────────

function collectCaptureUses(
  arrow: ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration,
  sf: ts.SourceFile,
): ts.Identifier[] {
  const bound = new Set<string>();
  // Parameters bind names.
  for (const p of arrow.parameters) {
    if (ts.isIdentifier(p.name)) bound.add(p.name.text);
  }

  const seen = new Set<string>();
  const captures: ts.Identifier[] = [];

  function visit(node: ts.Node, scopeBound: Set<string>): void {
    if (ts.isVariableStatement(node)) {
      const newBound = new Set(scopeBound);
      for (const d of node.declarationList.declarations) {
        if (ts.isIdentifier(d.name)) newBound.add(d.name.text);
        if (d.initializer) visit(d.initializer, scopeBound);
      }
      // Subsequent siblings in this scope see these names — but
      // forEachChild iterates in source order so the new bindings
      // need to flow forward. We don't have to handle the binding
      // order rigorously for our use case; treating all locals as
      // bound is enough.
      scopeBound = newBound;
      return;
    }
    if (ts.isParameter(node) && ts.isIdentifier(node.name)) {
      scopeBound.add(node.name.text);
    }
    if (ts.isIdentifier(node)) {
      // Skip if it's the property name of a property access (i.e.
      // `obj.prop` — `prop` doesn't bind anything).
      const parent = node.parent;
      if (parent && ts.isPropertyAccessExpression(parent) && parent.name === node) return;
      if (parent && ts.isPropertyAssignment(parent) && parent.name === node) return;
      if (parent && ts.isShorthandPropertyAssignment(parent) && parent.name === node) {
        // Shorthand: `{ x }` ≡ `{ x: x }` — the identifier IS a
        // value reference; fall through.
      }
      if (parent && ts.isMethodDeclaration(parent) && parent.name === node) return;

      const name = node.text;
      if (scopeBound.has(name)) return;
      if (SHIPPED_TYPE_NAMES.has(name)) return;
      if (SHIPPED_INTRINSIC_NAMES.has(name)) return;
      if (STAGE_BUILTIN_RECORDS.has(name)) return;
      if (GLOBAL_NAMES.has(name)) return;
      if (MARKER_NAMES.has(name as MarkerName)) return;
      if (seen.has(name)) return;
      seen.add(name);
      captures.push(node);
      return;
    }
    if (ts.isArrowFunction(node) || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node)) {
      const inner = new Set(scopeBound);
      for (const p of node.parameters) {
        if (ts.isIdentifier(p.name)) inner.add(p.name.text);
      }
      if (node.body) visit(node.body, inner);
      return;
    }
    ts.forEachChild(node, (c) => visit(c, scopeBound));
  }

  if (arrow.body) visit(arrow.body, bound);
  return captures;
}

// ─────────────────────────────────────────────────────────────────────
// Module-level const type extraction
// ─────────────────────────────────────────────────────────────────────

function collectModuleConsts(sf: ts.SourceFile): Map<string, ModuleConst> {
  const out = new Map<string, ModuleConst>();
  ts.forEachChild(sf, (node) => {
    if (!ts.isVariableStatement(node)) return;
    const isConst = (node.declarationList.flags & ts.NodeFlags.Const) !== 0
      || (node.declarationList.flags & ts.NodeFlags.Let) !== 0;
    if (!isConst) return;
    for (const d of node.declarationList.declarations) {
      if (!ts.isIdentifier(d.name)) continue;
      const name = d.name.text;
      // Strategy 1: explicit annotation.
      if (d.type && ts.isTypeReferenceNode(d.type) && ts.isIdentifier(d.type.typeName)) {
        const t = tryResolveTypeName(d.type.typeName.text);
        if (t) {
          out.set(name, { type: t });
          continue;
        }
      }
      // Strategy 2: `new V3f(...)` initializer.
      if (d.initializer && ts.isNewExpression(d.initializer)
          && ts.isIdentifier(d.initializer.expression)) {
        const t = tryResolveTypeName(d.initializer.expression.text);
        if (t) {
          out.set(name, { type: t });
          continue;
        }
      }
      // Otherwise no resolved type — capture lookups will report an
      // error that points at the marker call site, which is more
      // useful than failing here.
      out.set(name, {});
    }
  });
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

/**
 * Use the TS type-checker to recover the arrow's first parameter
 * type and its return type. The lookup is layered so any
 * combination of marker generics, lambda annotations, and
 * body-driven inference produces the same answer:
 *
 *   1. Contextual type from the enclosing call (covers
 *      `vertex<I, O>(...)` and lambda-annotation cases).
 *   2. The arrow's own signature (covers explicit lambda
 *      annotations when no contextual call exists).
 *   3. The arrow body's type — for the return only — so
 *      `vertex<I>(input => ({...}))` (output `O` defaulted to
 *      `unknown`) still picks up the actual literal shape.
 *
 * `unknown` / `never` / `any` are treated as "no info" — those
 * are the placeholders TS produces for unconstrained generics or
 * defaulted parameters, and feeding them to the frontend is no
 * better than feeding nothing.
 */
function inferArrowSignature(
  checker: ts.TypeChecker,
  arrow: ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration,
): { input: ts.TypeNode | undefined; output: ts.TypeNode | undefined } | undefined {
  const isUseless = (t: ts.Type | undefined): boolean => {
    if (t === undefined) return true;
    return (t.flags & (ts.TypeFlags.Unknown | ts.TypeFlags.Any | ts.TypeFlags.Never)) !== 0;
  };
  let inputType: ts.Type | undefined;
  let outputType: ts.Type | undefined;
  // Layer 1 — contextual type from the enclosing call.
  if (ts.isArrowFunction(arrow) || ts.isFunctionExpression(arrow)) {
    const ctxType = checker.getContextualType(arrow);
    const sig = ctxType?.getCallSignatures()[0];
    if (sig !== undefined) {
      const param0 = sig.parameters[0];
      if (param0?.valueDeclaration !== undefined) {
        const t = checker.getTypeOfSymbolAtLocation(param0, param0.valueDeclaration);
        if (!isUseless(t)) inputType = t;
      }
      const ret = checker.getReturnTypeOfSignature(sig);
      if (!isUseless(ret)) outputType = ret;
    }
  }
  // Layer 2 — the lambda's own declaration (explicit annotations).
  if (inputType === undefined || outputType === undefined) {
    const sig = checker.getSignatureFromDeclaration(arrow);
    if (sig !== undefined) {
      if (inputType === undefined) {
        const param0 = sig.parameters[0];
        if (param0?.valueDeclaration !== undefined) {
          const t = checker.getTypeOfSymbolAtLocation(param0, param0.valueDeclaration);
          if (!isUseless(t)) inputType = t;
        }
      }
      if (outputType === undefined) {
        const ret = checker.getReturnTypeOfSignature(sig);
        if (!isUseless(ret)) outputType = ret;
      }
    }
  }
  // Layer 3 — return type from the body itself. Picks up
  // `vertex<I>(input => ({...}))` where O defaulted to `unknown`.
  if (outputType === undefined && (ts.isArrowFunction(arrow) || ts.isFunctionExpression(arrow))) {
    const body = arrow.body;
    let bodyExpr: ts.Expression | undefined;
    if (ts.isExpression(body)) bodyExpr = body;
    else if (ts.isBlock(body)) {
      for (const st of body.statements) {
        if (ts.isReturnStatement(st) && st.expression) { bodyExpr = st.expression; break; }
      }
    }
    if (bodyExpr !== undefined) {
      const t = checker.getTypeAtLocation(bodyExpr);
      if (!isUseless(t)) outputType = t;
    }
  }
  if (inputType === undefined && outputType === undefined) return undefined;
  // Convert types back to syntactic TypeNodes so `deriveParamShape`
  // and `withDerivedOutputs` can iterate their members.
  const flags = ts.NodeBuilderFlags.NoTruncation | ts.NodeBuilderFlags.WriteArrayAsGenericType;
  const input = inputType !== undefined
    ? checker.typeToTypeNode(inputType, arrow, flags) ?? undefined
    : undefined;
  const output = outputType !== undefined
    ? checker.typeToTypeNode(outputType, arrow, flags) ?? undefined
    : undefined;
  return { input, output };
}

function synthesiseFunctionSource(
  arrow: ts.ArrowFunction | ts.FunctionExpression | ts.FunctionDeclaration,
  fnName: string,
  sf: ts.SourceFile,
  fallbackInputType?: ts.TypeNode,
  fallbackReturnType?: ts.TypeNode,
): string {
  const printer = ts.createPrinter({ removeComments: true });
  const params = arrow.parameters.map((p, i) => {
    // Use the parameter's own annotation when present; otherwise
    // fall back to the marker's first generic type argument for
    // the first parameter.
    const type = p.type ?? (i === 0 ? fallbackInputType : undefined);
    return ts.factory.createParameterDeclaration(
      undefined, undefined, p.name, p.questionToken, type, p.initializer,
    );
  });
  const arrowOrFn = arrow as { body?: ts.ConciseBody | ts.Block };
  const rawBody = arrowOrFn.body;
  if (!rawBody) {
    throw new Error("wombat.shader: function reference has no body");
  }
  const body = ts.isBlock(rawBody)
    ? rawBody
    : ts.factory.createBlock([ts.factory.createReturnStatement(rawBody)], true);
  // Same fallback for the return type — generic arg #1.
  const returnType = arrow.type ?? fallbackReturnType;
  const fn = ts.factory.createFunctionDeclaration(
    undefined,
    undefined,
    fnName,
    undefined,
    params,
    returnType,
    body,
  );
  // The printer needs the node's *own* source file (not the marker's
  // file) to recover original formatting / token positions when the
  // node lives in a different file. Cross-file helpers come from
  // shaders-utils.ts (or any imported file); using `sf` for them
  // produces garbled output (`new V3f` becomes `new ` because text
  // positions don't line up).
  const ownSf = arrow.getSourceFile?.() ?? sf;
  return printer.printNode(ts.EmitHint.Unspecified, fn, ownSf);
}

// ─────────────────────────────────────────────────────────────────────
// Helper discovery (transitively walk call sites)
// ─────────────────────────────────────────────────────────────────────

type HelperLike = ts.FunctionDeclaration | ts.ArrowFunction | ts.FunctionExpression;

/**
 * Walk the call graph reachable from `start`'s body. Every call to
 * an Identifier that resolves to a top-level function declaration
 * (or `const f = (…) =>`) in the same file is treated as a shader
 * helper. The result is in dependency-respecting order (callees
 * before callers) so the synthesised source compiles even before
 * any Function ValueDef hoist.
 */
function discoverHelpers(
  start: HelperLike,
  sf: ts.SourceFile,
  checker: ts.TypeChecker | undefined,
): HelperLike[] {
  const found = new Map<string, HelperLike>();
  const seen = new Set<HelperLike>();
  const queue: HelperLike[] = [start];
  // Track recursion: a helper's body may not call itself. We don't
  // explicitly check (deferred), but the seen-set bounds the walk.
  while (queue.length > 0) {
    const fn = queue.shift()!;
    if (seen.has(fn)) continue;
    seen.add(fn);
    visit(getBody(fn));
  }
  function visit(node: ts.Node | undefined): void {
    if (!node) return;
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      const name = node.expression.text;
      // Skip shipped intrinsics / constructors / the marker calls
      // themselves — those don't need helper translation.
      if (
        SHIPPED_INTRINSIC_NAMES.has(name) ||
        SHIPPED_TYPE_NAMES.has(name) ||
        MARKER_NAMES.has(name as MarkerName)
      ) {
        for (const a of node.arguments) visit(a);
        return;
      }
      if (!found.has(name)) {
        // Same-file lookup first (cheap, no type-checker hop).
        let decl = findTopLevelFunction(sf, name);
        // Cross-file fallback via the TS checker: follow the
        // identifier's symbol (through `import` aliases) to its
        // declaring node. Works for top-level `function` and
        // `const fn = (…) => …` in any project file.
        if (!decl && checker) {
          decl = findCrossFileHelper(node.expression, checker);
        }
        if (decl) {
          found.set(name, decl);
          queue.push(decl);
        }
      }
      for (const a of node.arguments) visit(a);
      return;
    }
    ts.forEachChild(node, visit);
  }
  // Output order: helpers in the order we discovered them. Both
  // GLSL ES 3.00 and WGSL allow any function-decl order with
  // forward calls, so we don't need topological sort.
  return [...found.values()];
}

function findCrossFileHelper(
  identNode: ts.Identifier,
  checker: ts.TypeChecker,
): HelperLike | undefined {
  let sym = checker.getSymbolAtLocation(identNode);
  if (!sym) return undefined;
  // Follow `import` aliases to the original declaration.
  if (sym.flags & ts.SymbolFlags.Alias) {
    sym = checker.getAliasedSymbol(sym);
  }
  for (const d of sym.declarations ?? []) {
    if (ts.isFunctionDeclaration(d) && d.body) return d;
    if (
      ts.isVariableDeclaration(d) && d.initializer &&
      (ts.isArrowFunction(d.initializer) || ts.isFunctionExpression(d.initializer))
    ) {
      return d.initializer;
    }
  }
  return undefined;
}

function helperName(fn: HelperLike): string {
  if (ts.isFunctionDeclaration(fn) && fn.name) return fn.name.text;
  // For const-arrow / function-expression helpers, the name lives on
  // the enclosing VariableDeclaration. Walk up to find it.
  let parent: ts.Node | undefined = fn.parent;
  while (parent && !ts.isVariableDeclaration(parent)) parent = parent.parent;
  if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) {
    return parent.name.text;
  }
  throw new Error("wombat.shader: helper has no resolvable name");
}

function getBody(fn: HelperLike): ts.Node | undefined {
  if (ts.isFunctionDeclaration(fn) || ts.isFunctionExpression(fn)) return fn.body;
  // ArrowFunction body is either a block or a concise expression body.
  return fn.body;
}

function synthesiseHelperSource(
  fn: HelperLike,
  name: string,
  sf: ts.SourceFile,
): string {
  // Reuse synthesiseFunctionSource for all three shapes — it
  // handles each by accessing common parameter/body fields.
  return synthesiseFunctionSource(fn, name, sf);
}

function findTopLevelFunction(
  sf: ts.SourceFile,
  name: string,
): ts.FunctionDeclaration | ts.ArrowFunction | ts.FunctionExpression | undefined {
  for (const node of sf.statements) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) {
      return node;
    }
    if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations) {
        if (!ts.isIdentifier(d.name) || d.name.text !== name) continue;
        const init = d.initializer;
        if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
          return init;
        }
      }
    }
  }
  return undefined;
}

function sourceImportsRuntime(source: string): boolean {
  return /from\s+["']@aardworx\/wombat\.shader["']/.test(source);
}


function error(sf: ts.SourceFile, node: ts.Node, message: string): Error {
  const { line, character } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
  return new Error(`${sf.fileName}:${line + 1}:${character + 1} — ${message}`);
}
