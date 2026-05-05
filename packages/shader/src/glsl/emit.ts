// IR → GLSL ES 3.00 emitter (WebGL2 target).
//
// Consumes a fully-legalised IR Module (passes upstream of this should have
// done inlining, constant folding, DCE, cross-stage pruning, and target
// legalisation). The emitter walks the tree and produces a string + a
// minimal binding map.

import type {
  EntryDef,
  EntryParameter,
  Expr,
  LExpr,
  Literal,
  Module,
  RExpr,
  Stmt,
  SwitchCase,
  Type,
  UniformDecl,
  Var,
} from "../ir/index.js";

export interface EmitResult {
  readonly source: string;
  readonly bindings: BindingMap;
  readonly meta: BackendMeta;
  /** Per-emitted-line spans for source-map construction. */
  readonly lineSpans: ReadonlyArray<import("../ir/index.js").Span | undefined>;
}

export interface BindingMap {
  readonly inputs: readonly LocationBinding[];
  readonly outputs: readonly LocationBinding[];
  readonly uniforms: readonly UniformBinding[];
  readonly samplers: readonly SamplerBinding[];
}

export interface LocationBinding {
  readonly name: string;
  readonly location: number;
  readonly type: Type;
}

export interface UniformBinding {
  readonly name: string;
  readonly type: Type;
  readonly buffer?: string;
}

export interface SamplerBinding {
  readonly name: string;
  readonly type: Type;
}

export interface BackendMeta {
  readonly stage: "vertex" | "fragment" | "compute";
  readonly version: "300 es";
  readonly workgroupSize?: readonly [number, number, number] | undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────

export function emitGlsl(module: Module, entryName?: string): EmitResult {
  const entry = pickEntry(module, entryName);
  const w = new Writer();
  const ctx: Ctx = {
    out: w,
    structs: new Set(),
    bindings: {
      inputs: [],
      outputs: [],
      uniforms: [],
      samplers: [],
    },
  };

  w.line("#version 300 es");
  if (entry.stage === "fragment") w.line("precision highp float;");
  if (entry.stage === "vertex") w.line("precision highp float;");
  w.blank();

  // Struct types referenced by anything we'll emit.
  for (const t of module.types) {
    if (t.kind === "Struct") {
      emitStructDecl(ctx, t);
    }
  }
  if (module.types.length > 0) w.blank();

  // Uniforms / samplers — only for the chosen entry's surface area.
  for (const v of module.values) {
    if (v.kind === "Uniform") emitUniformGroup(ctx, v.uniforms);
    else if (v.kind === "Sampler") emitSampler(ctx, v.name, v.type);
  }
  if (module.values.some((v) => v.kind === "Uniform" || v.kind === "Sampler")) w.blank();

  // Entry inputs / outputs as `in`/`out` globals.
  emitEntryIO(ctx, entry);

  // User functions — emit any FunctionDef preceding `main`.
  for (const v of module.values) {
    if (v.kind === "Function") emitFunction(ctx, v.signature, v.body);
  }

  // Emit `void main()` from the entry's body.
  emitEntryMain(ctx, entry);

  return {
    source: w.toString(),
    bindings: ctx.bindings,
    meta: {
      stage: entry.stage,
      version: "300 es",
      workgroupSize: undefined,
    },
    lineSpans: w.lineSpans,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────

interface MutableBindings {
  inputs: LocationBinding[];
  outputs: LocationBinding[];
  uniforms: UniformBinding[];
  samplers: SamplerBinding[];
}

interface Ctx {
  readonly out: Writer;
  readonly structs: Set<string>;
  readonly bindings: MutableBindings;
}

function pickEntry(module: Module, entryName?: string): EntryDef {
  const entries = module.values.flatMap((v) => v.kind === "Entry" ? [v.entry] : []);
  if (entries.length === 0) throw new Error("emitGlsl: module has no entry point");
  if (entryName) {
    const e = entries.find((x) => x.name === entryName);
    if (!e) throw new Error(`emitGlsl: entry "${entryName}" not found`);
    return e;
  }
  if (entries.length > 1) {
    throw new Error("emitGlsl: multiple entries; pass entryName to select one");
  }
  return entries[0]!;
}

class Writer {
  private readonly parts: string[] = [];
  private indent = 0;
  private currentSpan: import("../ir/index.js").Span | undefined;
  readonly lineSpans: (import("../ir/index.js").Span | undefined)[] = [];

  line(s: string): void {
    this.parts.push("    ".repeat(this.indent) + s + "\n");
    this.lineSpans.push(this.currentSpan);
  }
  blank(): void {
    this.parts.push("\n");
    this.lineSpans.push(undefined);
  }
  setSpan(s: import("../ir/index.js").Span | undefined): void {
    this.currentSpan = s;
  }
  push(s: string): void {
    this.parts.push(s);
  }
  inc(): void { this.indent++; }
  dec(): void { this.indent = Math.max(0, this.indent - 1); }
  toString(): string {
    return this.parts.join("");
  }
}

function emitStructDecl(ctx: Ctx, t: { kind: "Struct"; name: string; fields: readonly { type: Type; name: string }[] }): void {
  if (ctx.structs.has(t.name)) return;
  ctx.structs.add(t.name);
  ctx.out.line(`struct ${t.name} {`);
  ctx.out.inc();
  for (const f of t.fields) {
    ctx.out.line(`${typeStr(f.type)} ${f.name};`);
  }
  ctx.out.dec();
  ctx.out.line("};");
}

function emitUniformGroup(ctx: Ctx, uniforms: readonly UniformDecl[]): void {
  // Group by buffer name (default = each uniform its own block).
  const buckets = new Map<string | undefined, UniformDecl[]>();
  for (const u of uniforms) {
    const key = u.buffer;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(u);
    else buckets.set(key, [u]);
  }
  for (const [buffer, group] of buckets) {
    if (buffer) {
      ctx.out.line(`layout(std140) uniform ${buffer} {`);
      ctx.out.inc();
      for (const u of group) {
        ctx.out.line(`${typeStr(u.type)} ${u.name};`);
        ctx.bindings.uniforms.push({ name: u.name, type: u.type, buffer });
      }
      ctx.out.dec();
      ctx.out.line("};");
    } else {
      for (const u of group) {
        ctx.out.line(`uniform ${typeStr(u.type)} ${u.name};`);
        ctx.bindings.uniforms.push({ name: u.name, type: u.type });
      }
    }
  }
}

function emitSampler(ctx: Ctx, name: string, type: Type): void {
  ctx.out.line(`uniform ${typeStr(type)} ${name};`);
  ctx.bindings.samplers.push({ name, type });
}

function emitEntryIO(ctx: Ctx, e: EntryDef): void {
  // GLSL ES 3.00 layout-location rules:
  //  - vertex inputs (attributes)     : location IS allowed
  //  - vertex outputs (interpolants)  : location NOT allowed (name-matched)
  //  - fragment inputs (interpolants) : location NOT allowed
  //  - fragment outputs (color out)   : location IS allowed
  //  - compute: no in/out
  const allowInputLocation = e.stage === "vertex";
  const allowOutputLocation = e.stage === "fragment";

  let nextLoc = 0;
  for (const p of e.inputs) {
    if (isBuiltin(p)) continue;
    const loc = locationOf(p) ?? nextLoc++;
    const interp = interpolation(p);
    const layout = allowInputLocation ? `layout(location = ${loc}) ` : "";
    const prefix = (interp ? interp + " " : "") + `${layout}in`;
    ctx.out.line(`${prefix} ${typeStr(p.type)} ${p.name};`);
    ctx.bindings.inputs.push({ name: p.name, location: loc, type: p.type });
  }
  nextLoc = 0;
  for (const p of e.outputs) {
    if (isBuiltin(p)) continue;
    const loc = locationOf(p) ?? nextLoc++;
    const interp = interpolation(p);
    const layout = allowOutputLocation ? `layout(location = ${loc}) ` : "";
    const prefix = (interp ? interp + " " : "") + `${layout}out`;
    ctx.out.line(`${prefix} ${typeStr(p.type)} ${p.name};`);
    ctx.bindings.outputs.push({ name: p.name, location: loc, type: p.type });
  }
  if (e.inputs.length > 0 || e.outputs.length > 0) ctx.out.blank();
}

function isBuiltin(p: EntryParameter): boolean {
  return p.decorations.some((d) => d.kind === "Builtin");
}

function locationOf(p: EntryParameter): number | undefined {
  const d = p.decorations.find((x) => x.kind === "Location");
  return d && d.kind === "Location" ? d.value : undefined;
}

function interpolation(p: EntryParameter): string | undefined {
  const d = p.decorations.find((x) => x.kind === "Interpolation");
  if (!d || d.kind !== "Interpolation") return undefined;
  switch (d.mode) {
    case "smooth": return "smooth";
    case "flat": return "flat";
    case "centroid": return "centroid";
    case "sample": return "sample";
    case "no-perspective": return undefined; // not in GLSL ES 3.00
  }
}

function emitFunction(
  ctx: Ctx,
  sig: { name: string; returnType: Type; parameters: readonly { name: string; type: Type; modifier: "in" | "inout" }[] },
  body: Stmt,
): void {
  const params = sig.parameters
    .map((p) => `${p.modifier === "inout" ? "inout " : ""}${typeStr(p.type)} ${p.name}`)
    .join(", ");
  ctx.out.line(`${typeStr(sig.returnType)} ${sig.name}(${params}) {`);
  ctx.out.inc();
  emitStmt(ctx, body);
  ctx.out.dec();
  ctx.out.line("}");
  ctx.out.blank();
}

function emitEntryMain(ctx: Ctx, e: EntryDef): void {
  ctx.out.line("void main() {");
  ctx.out.inc();
  emitStmt(ctx, e.body);
  ctx.out.dec();
  ctx.out.line("}");
}

// ─────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────

function emitStmt(ctx: Ctx, s: Stmt): void {
  if (s.span) ctx.out.setSpan(s.span);
  switch (s.kind) {
    case "Nop":
      return;
    case "Expression":
      ctx.out.line(`${expr(s.value)};`);
      return;
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      ctx.out.line(`${typeStr(s.var.type)} ${s.var.name}${init};`);
      return;
    }
    case "Write":
      ctx.out.line(`${lexpr(s.target)} = ${expr(s.value)};`);
      return;
    case "WriteOutput": {
      const idx = s.index ? `[${expr(s.index)}]` : "";
      ctx.out.line(`${s.name}${idx} = ${rexpr(s.value)};`);
      return;
    }
    case "Increment":
      ctx.out.line(s.prefix ? `++${lexpr(s.target)};` : `${lexpr(s.target)}++;`);
      return;
    case "Decrement":
      ctx.out.line(s.prefix ? `--${lexpr(s.target)};` : `${lexpr(s.target)}--;`);
      return;
    case "Sequential":
    case "Isolated":
      for (const c of s.body) emitStmt(ctx, c);
      return;
    case "Return":
      ctx.out.line("return;");
      return;
    case "ReturnValue":
      ctx.out.line(`return ${expr(s.value)};`);
      return;
    case "Break":
      ctx.out.line("break;");
      return;
    case "Continue":
      ctx.out.line("continue;");
      return;
    case "Discard":
      ctx.out.line("discard;");
      return;
    case "Barrier":
      // GLSL ES 3.00 has memoryBarrier()/barrier() in compute (300 es supports
      // compute via WebGL2 only with the compute extension; we emit the
      // canonical name and let the link-time check catch unsupported targets).
      ctx.out.line("barrier();");
      return;
    case "If":
      ctx.out.line(`if (${expr(s.cond)}) {`);
      ctx.out.inc(); emitStmt(ctx, s.then); ctx.out.dec();
      if (s.else) {
        ctx.out.line("} else {");
        ctx.out.inc(); emitStmt(ctx, s.else); ctx.out.dec();
      }
      ctx.out.line("}");
      return;
    case "For": {
      // For-init is a single Stmt; emit it inline as a declaration or expr.
      const initText = forStmtText(s.init);
      const stepText = forStmtText(s.step).replace(/;$/, "");
      ctx.out.line(`for (${initText} ${expr(s.cond)}; ${stepText}) {`);
      ctx.out.inc(); emitStmt(ctx, s.body); ctx.out.dec();
      ctx.out.line("}");
      return;
    }
    case "While":
      ctx.out.line(`while (${expr(s.cond)}) {`);
      ctx.out.inc(); emitStmt(ctx, s.body); ctx.out.dec();
      ctx.out.line("}");
      return;
    case "DoWhile":
      ctx.out.line("do {");
      ctx.out.inc(); emitStmt(ctx, s.body); ctx.out.dec();
      ctx.out.line(`} while (${expr(s.cond)});`);
      return;
    case "Loop":
      // GLSL has no `loop {}` keyword; lower to `while (true) {}`.
      ctx.out.line("while (true) {");
      ctx.out.inc(); emitStmt(ctx, s.body); ctx.out.dec();
      ctx.out.line("}");
      return;
    case "Switch":
      emitSwitch(ctx, s.value, s.cases, s.default);
      return;
  }
}

function forStmtText(s: Stmt): string {
  switch (s.kind) {
    case "Nop":
      return ";";
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      return `${typeStr(s.var.type)} ${s.var.name}${init};`;
    }
    case "Expression":
      return `${expr(s.value)};`;
    case "Write":
      return `${lexpr(s.target)} = ${expr(s.value)};`;
    case "Increment":
      return s.prefix ? `++${lexpr(s.target)};` : `${lexpr(s.target)}++;`;
    case "Decrement":
      return s.prefix ? `--${lexpr(s.target)};` : `${lexpr(s.target)}--;`;
    default:
      // Other statement kinds aren't valid in for-init/step; collapse to nop.
      return ";";
  }
}

function emitSwitch(
  ctx: Ctx,
  value: Expr,
  cases: readonly SwitchCase[],
  defaultStmt: Stmt | undefined,
): void {
  ctx.out.line(`switch (${expr(value)}) {`);
  ctx.out.inc();
  for (const c of cases) {
    ctx.out.line(`case ${literal(c.literal)}:`);
    ctx.out.inc(); emitStmt(ctx, c.body); ctx.out.dec();
  }
  if (defaultStmt) {
    ctx.out.line("default:");
    ctx.out.inc(); emitStmt(ctx, defaultStmt); ctx.out.dec();
  }
  ctx.out.dec();
  ctx.out.line("}");
}

// ─────────────────────────────────────────────────────────────────────
// Expression printer (returns a string; statements call into this)
// ─────────────────────────────────────────────────────────────────────

const BIN_OP: Partial<Record<Expr["kind"], string>> = {
  Add: "+", Sub: "-", Mul: "*", Div: "/", Mod: "%",
  And: "&&", Or: "||",
  BitAnd: "&", BitOr: "|", BitXor: "^",
  ShiftLeft: "<<", ShiftRight: ">>",
  Eq: "==", Neq: "!=", Lt: "<", Le: "<=", Gt: ">", Ge: ">=",
};

const VEC_BIN_FN: Partial<Record<Expr["kind"], string>> = {
  VecLt: "lessThan", VecLe: "lessThanEqual",
  VecGt: "greaterThan", VecGe: "greaterThanEqual",
  VecEq: "equal", VecNeq: "notEqual",
};

export function expr(e: Expr): string {
  switch (e.kind) {
    case "Var":
      return varRef(e.var);
    case "Const":
      return literal(e.value);
    case "ReadInput":
      return e.index !== undefined ? `${e.name}[${expr(e.index)}]` : e.name;
    case "Call":
      return `${e.fn.signature.name}(${e.args.map(expr).join(", ")})`;
    case "CallIntrinsic":
      if (e.op.atomic) {
        throw new Error(
          `emitGlsl: atomic intrinsic "${e.op.name}" is not supported on WebGL2 (GLSL ES 3.00). ` +
          `Atomics require WGSL/WebGPU or GLSL ES 3.10+.`,
        );
      }
      return `${e.op.emit.glsl}(${e.args.map(expr).join(", ")})`;
    case "Conditional":
      return `((${expr(e.cond)}) ? (${expr(e.ifTrue)}) : (${expr(e.ifFalse)}))`;
    case "Neg":
      return `(-${expr(e.value)})`;
    case "Not":
      return `(!${expr(e.value)})`;
    case "BitNot":
      return `(~${expr(e.value)})`;
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Mod":
    case "MulMatMat":
    case "MulMatVec":
    case "MulVecMat":
    case "And":
    case "Or":
    case "BitAnd":
    case "BitOr":
    case "BitXor":
    case "ShiftLeft":
    case "ShiftRight":
    case "Eq":
    case "Neq":
    case "Lt":
    case "Le":
    case "Gt":
    case "Ge": {
      // Matrix multiplications use `*` in GLSL, same as scalar/vector — driver dispatches by type.
      const op = BIN_OP[e.kind] ?? "*";
      return `(${expr(e.lhs)} ${op} ${expr(e.rhs)})`;
    }
    case "Transpose":
      return `transpose(${expr(e.value)})`;
    case "Inverse":
      return `inverse(${expr(e.value)})`;
    case "Determinant":
      return `determinant(${expr(e.value)})`;
    case "Dot":
      return `dot(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Cross":
      return `cross(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Length":
      return `length(${expr(e.value)})`;
    case "VecSwizzle":
      return `${expr(e.value)}.${e.comps.join("")}`;
    case "VecItem":
      return `${expr(e.value)}[${expr(e.index)}]`;
    case "MatrixElement":
      return `${expr(e.matrix)}[${expr(e.col)}][${expr(e.row)}]`;
    case "MatrixRow":
      // GLSL has no row accessor; transpose-based pattern is verbose. The
      // frontend should lower this to MatrixElement before we get here. We
      // emit a defensive comment and hope upstream legalisation has done it.
      return `/* MatrixRow not directly representable in GLSL */ ${expr(e.matrix)}[${expr(e.row)}]`;
    case "MatrixCol":
      return `${expr(e.matrix)}[${expr(e.col)}]`;
    case "NewVector":
      return `${typeStr(e.type)}(${e.components.map(expr).join(", ")})`;
    case "NewMatrix":
      return `${typeStr(e.type)}(${e.elements.map(expr).join(", ")})`;
    case "MatrixFromRows":
      // GLSL constructors take columns. Frontend should lower MatrixFromRows
      // to a transpose of MatrixFromCols.
      return `transpose(${typeStr(e.type)}(${e.rows.map(expr).join(", ")}))`;
    case "MatrixFromCols":
      return `${typeStr(e.type)}(${e.cols.map(expr).join(", ")})`;
    case "ConvertMatrix":
    case "Convert":
      return `${typeStr(e.type)}(${expr(e.value)})`;
    case "VecAny":
      return `any(${expr(e.value)})`;
    case "VecAll":
      return `all(${expr(e.value)})`;
    case "VecLt":
    case "VecLe":
    case "VecGt":
    case "VecGe":
    case "VecEq":
    case "VecNeq":
      return `${VEC_BIN_FN[e.kind]}(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Field":
      return `${expr(e.target)}.${e.name}`;
    case "Item":
      return `${expr(e.target)}[${expr(e.index)}]`;
    case "DebugPrintf":
      // GLSL has no printf; lower to a noop comment.
      return `/* debugPrintf */ 0`;
  }
}

export function lexpr(l: LExpr): string {
  switch (l.kind) {
    case "LVar":
      return varRef(l.var);
    case "LField":
      return `${lexpr(l.target)}.${l.name}`;
    case "LItem":
      return `${lexpr(l.target)}[${expr(l.index)}]`;
    case "LSwizzle":
      return `${lexpr(l.target)}.${l.comps.join("")}`;
    case "LMatrixElement":
      return `${lexpr(l.matrix)}[${expr(l.col)}][${expr(l.row)}]`;
    case "LInput":
      return l.index !== undefined ? `${l.name}[${expr(l.index)}]` : l.name;
  }
}

function rexpr(r: RExpr): string {
  if (r.kind === "Expr") return expr(r.value);
  return `${typeStr(r.arrayType)}(${r.values.map(expr).join(", ")})`;
}

function varRef(v: Var): string {
  return v.name;
}

function literal(l: Literal): string {
  switch (l.kind) {
    case "Bool":
      return l.value ? "true" : "false";
    case "Int":
      return l.signed ? `${l.value | 0}` : `${l.value >>> 0}u`;
    case "Float":
      return formatFloat(l.value);
    case "Null":
      return "0"; // sampler defaults — should be replaced by upstream legalisation
  }
}

function formatFloat(n: number): string {
  if (!Number.isFinite(n)) {
    return n === Infinity ? "(1.0/0.0)" : n === -Infinity ? "(-1.0/0.0)" : "(0.0/0.0)";
  }
  // Always include a decimal point so GLSL parses as float.
  if (Number.isInteger(n)) return `${n}.0`;
  return `${n}`;
}

// ─────────────────────────────────────────────────────────────────────
// Type printer
// ─────────────────────────────────────────────────────────────────────

function typeStr(t: Type): string {
  switch (t.kind) {
    case "Void": return "void";
    case "Bool": return "bool";
    case "Int":  return t.signed ? "int" : "uint";
    case "Float": return "float";
    case "Vector": {
      const dim = t.dim;
      switch (t.element.kind) {
        case "Bool":  return `bvec${dim}`;
        case "Int":   return t.element.signed ? `ivec${dim}` : `uvec${dim}`;
        case "Float": return `vec${dim}`;
        default:      throw new Error(`unsupported vector element ${t.element.kind}`);
      }
    }
    case "Matrix": {
      // GLSL matN means matNxN; matRxC has rows R cols C — note GLSL convention is matCxR (cols×rows).
      // We use mat<cols>x<rows> for non-square and matN for square.
      if (t.rows === t.cols) return `mat${t.rows}`;
      return `mat${t.cols}x${t.rows}`;
    }
    case "Array":
      return t.length === "runtime"
        ? `${typeStr(t.element)}[]`
        : `${typeStr(t.element)}[${t.length}]`;
    case "Struct":
      return t.name;
    case "Sampler":
      return samplerTypeStr(t);
    case "Texture":
      // GLSL combines sampler+texture into one symbol — Texture without a
      // companion sampler isn't a valid GLSL type. Upstream legalisation
      // should have folded textures into Samplers before we got here.
      return samplerTypeStr({
        target: t.target,
        sampled: t.sampled,
        comparison: false,
      });
    case "StorageTexture":
      throw new Error(
        `emitGlsl: storage textures (texture_storage_*) are not supported on ` +
        `WebGL2 (GLSL ES 3.00). Use the WGSL/WebGPU target.`,
      );
    case "AtomicI32": return "int";
    case "AtomicU32": return "uint";
    case "Intrinsic":
      throw new Error(`emitGlsl: unresolved intrinsic type "${t.name}"`);
  }
}

function samplerTypeStr(t: { target: string; sampled: { kind: "Float" } | { kind: "Int"; signed: boolean }; comparison: boolean }): string {
  if (t.target === "2DMS" || t.target === "2DMSArray") {
    throw new Error(
      `emitGlsl: multisampled samplers (target=${t.target}) are not supported on ` +
      `WebGL2 (GLSL ES 3.00). Use the WGSL/WebGPU target.`,
    );
  }
  const prefix = t.sampled.kind === "Float"
    ? ""
    : t.sampled.signed ? "i" : "u";
  const targetStr =
    t.target === "1D" ? "1D" :
    t.target === "2D" ? "2D" :
    t.target === "3D" ? "3D" :
    t.target === "Cube" ? "Cube" :
    t.target === "2DArray" ? "2DArray" :
    t.target === "CubeArray" ? "CubeArray" :
    t.target;
  return `${prefix}sampler${targetStr}${t.comparison ? "Shadow" : ""}`;
}

// Used by tests / debugging.
export const _internal = { typeStr, expr, lexpr, literal };
