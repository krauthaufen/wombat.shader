// IR → WGSL emitter (WebGPU target).
//
// Conventions vs. the GLSL emitter:
//
//  - Address spaces are explicit. Uniforms go in `var<uniform>`, storage
//    buffers in `var<storage>`, samplers/textures are flat globals.
//  - Entry inputs/outputs are wrapped in a struct that carries the
//    `@location(n)` and `@builtin(...)` attributes. The struct is
//    generated per entry, with names derived from the entry name.
//  - Sampler+texture are separate globals (WGSL doesn't combine them).
//    Upstream legalisation should have split combined GLSL samplers
//    into `Sampler` + `Texture` pairs before we emit.
//  - Function-local `let` (immutable) vs. `var` (mutable). We always emit
//    `var` for IR `Declare` since the IR doesn't distinguish let/var
//    explicitly; passes can lower to `let` later for read-only locals.

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
} from "@aardworx/wombat.shader-ir";

export interface EmitResult {
  readonly source: string;
  readonly bindings: BindingMap;
  readonly meta: BackendMeta;
}

export interface BindingMap {
  readonly inputs: readonly LocationBinding[];
  readonly outputs: readonly LocationBinding[];
  readonly uniforms: readonly UniformBinding[];
  readonly samplers: readonly SamplerBinding[];
  readonly storage: readonly StorageBinding[];
}

export interface LocationBinding { readonly name: string; readonly location: number; readonly type: Type }
export interface UniformBinding { readonly name: string; readonly group: number; readonly slot: number; readonly type: Type }
export interface SamplerBinding { readonly name: string; readonly group: number; readonly slot: number; readonly type: Type }
export interface StorageBinding { readonly name: string; readonly group: number; readonly slot: number; readonly type: Type; readonly access: "read" | "read_write" }

export interface BackendMeta {
  readonly stage: "vertex" | "fragment" | "compute";
  readonly version: "wgsl";
  readonly workgroupSize?: readonly [number, number, number] | undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────

export function emitWgsl(module: Module, entryName?: string): EmitResult {
  const entry = pickEntry(module, entryName);
  const w = new Writer();
  const ctx: Ctx = {
    out: w,
    structs: new Set(),
    bindings: { inputs: [], outputs: [], uniforms: [], samplers: [], storage: [] },
  };
  // Build name → buffer-struct map so `expr()` knows that a
  // ReadInput("Uniform", "u_camera_view") whose decl has buffer:"Camera"
  // emits as `Camera.u_camera_view`.
  bufferOf = new Map<string, string>();
  for (const v of module.values) {
    if (v.kind !== "Uniform") continue;
    for (const u of v.uniforms) {
      if (u.buffer) bufferOf.set(u.name, u.buffer);
    }
  }

  // Type definitions.
  for (const t of module.types) {
    if (t.kind === "Struct") emitStructDecl(ctx, t);
  }
  if (module.types.length > 0) w.blank();

  // Module-level decls: uniforms, samplers, storage buffers.
  let uniformAutoGroup = 0;
  for (const v of module.values) {
    if (v.kind === "Uniform") {
      emitUniformGroup(ctx, v.uniforms, uniformAutoGroup);
      uniformAutoGroup++;
    } else if (v.kind === "Sampler") {
      emitSampler(ctx, v.name, v.binding.group, v.binding.slot, v.type);
    } else if (v.kind === "StorageBuffer") {
      emitStorageBuffer(ctx, v.name, v.binding.group, v.binding.slot, v.layout, v.access);
    }
  }

  // Synthetic input/output structs for the chosen entry.
  const ioInfo = emitEntryStructs(ctx, entry);

  // User functions.
  for (const v of module.values) {
    if (v.kind === "Function") emitFunction(ctx, v.signature, v.body);
  }

  // Stage entry function.
  emitEntryFunction(ctx, entry, ioInfo);

  return {
    source: w.toString(),
    bindings: ctx.bindings,
    meta: {
      stage: entry.stage,
      version: "wgsl",
      workgroupSize: workgroupSize(entry),
    },
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
  storage: StorageBinding[];
}

interface Ctx {
  readonly out: Writer;
  readonly structs: Set<string>;
  readonly bindings: MutableBindings;
}

interface EntryIO {
  readonly inputStructName?: string | undefined;
  readonly outputStructName?: string | undefined;
  readonly hasOutputStruct: boolean;
}

function pickEntry(module: Module, entryName?: string): EntryDef {
  const entries = module.values.flatMap((v) => v.kind === "Entry" ? [v.entry] : []);
  if (entries.length === 0) throw new Error("emitWgsl: module has no entry point");
  if (entryName) {
    const e = entries.find((x) => x.name === entryName);
    if (!e) throw new Error(`emitWgsl: entry "${entryName}" not found`);
    return e;
  }
  if (entries.length > 1) {
    throw new Error("emitWgsl: multiple entries; pass entryName to select one");
  }
  return entries[0]!;
}

class Writer {
  private readonly parts: string[] = [];
  private indent = 0;

  line(s: string): void { this.parts.push("    ".repeat(this.indent) + s + "\n"); }
  blank(): void { this.parts.push("\n"); }
  inc(): void { this.indent++; }
  dec(): void { this.indent = Math.max(0, this.indent - 1); }
  toString(): string { return this.parts.join(""); }
}

function emitStructDecl(ctx: Ctx, t: { kind: "Struct"; name: string; fields: readonly { type: Type; name: string }[] }): void {
  if (ctx.structs.has(t.name)) return;
  ctx.structs.add(t.name);
  ctx.out.line(`struct ${t.name} {`);
  ctx.out.inc();
  for (const f of t.fields) {
    ctx.out.line(`${f.name}: ${typeStr(f.type)},`);
  }
  ctx.out.dec();
  ctx.out.line("};");
}

function emitUniformGroup(ctx: Ctx, uniforms: readonly UniformDecl[], autoGroup: number): void {
  // Group by buffer name (default = each uniform its own var<uniform>).
  const buckets = new Map<string | undefined, UniformDecl[]>();
  for (const u of uniforms) {
    const key = u.buffer;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(u);
    else buckets.set(key, [u]);
  }
  let nextSlot = 0;
  for (const [buffer, group] of buckets) {
    if (buffer) {
      // Generate a struct for the buffer.
      const structName = `_UB_${buffer}`;
      ctx.out.line(`struct ${structName} {`);
      ctx.out.inc();
      for (const u of group) ctx.out.line(`${u.name}: ${typeStr(u.type)},`);
      ctx.out.dec();
      ctx.out.line("};");
      const grp = group[0]?.group ?? autoGroup;
      const slot = group[0]?.slot ?? nextSlot++;
      ctx.out.line(`@group(${grp}) @binding(${slot}) var<uniform> ${buffer}: ${structName};`);
      for (const u of group) {
        ctx.bindings.uniforms.push({ name: `${buffer}.${u.name}`, group: grp, slot, type: u.type });
      }
    } else {
      for (const u of group) {
        const grp = u.group ?? autoGroup;
        const slot = u.slot ?? nextSlot++;
        ctx.out.line(`@group(${grp}) @binding(${slot}) var<uniform> ${u.name}: ${typeStr(u.type)};`);
        ctx.bindings.uniforms.push({ name: u.name, group: grp, slot, type: u.type });
      }
    }
  }
}

function emitSampler(ctx: Ctx, name: string, group: number, slot: number, type: Type): void {
  ctx.out.line(`@group(${group}) @binding(${slot}) var ${name}: ${typeStr(type)};`);
  ctx.bindings.samplers.push({ name, group, slot, type });
}

function emitStorageBuffer(
  ctx: Ctx, name: string, group: number, slot: number, layout: Type, access: "read" | "read_write",
): void {
  ctx.out.line(`@group(${group}) @binding(${slot}) var<storage, ${access}> ${name}: ${typeStr(layout)};`);
  ctx.bindings.storage.push({ name, group, slot, type: layout, access });
}

function emitEntryStructs(ctx: Ctx, e: EntryDef): EntryIO {
  let inputStructName: string | undefined;
  let outputStructName: string | undefined;

  // Collect non-builtin inputs into a struct.
  const inputs = e.inputs.filter((p) => !isBuiltin(p));
  if (inputs.length > 0) {
    inputStructName = `${capitalise(e.name)}Input`;
    ctx.out.line(`struct ${inputStructName} {`);
    ctx.out.inc();
    let nextLoc = 0;
    for (const p of inputs) {
      const loc = locationOf(p) ?? nextLoc++;
      const interp = interpolation(p);
      const decor = (interp ? `${interp} ` : "") + `@location(${loc})`;
      ctx.out.line(`${decor} ${p.name}: ${typeStr(p.type)},`);
      ctx.bindings.inputs.push({ name: p.name, location: loc, type: p.type });
    }
    ctx.out.dec();
    ctx.out.line("};");
  }

  // Outputs: builtin outputs (e.g. position) get @builtin attributes inside
  // the same struct as @location ones. Vertex stages have a builtin position;
  // fragment stages have @location(0) etc.
  const hasOutputs = e.outputs.length > 0 && e.stage !== "compute";
  if (hasOutputs) {
    outputStructName = `${capitalise(e.name)}Output`;
    ctx.out.line(`struct ${outputStructName} {`);
    ctx.out.inc();
    let nextLoc = 0;
    for (const p of e.outputs) {
      const builtin = builtinName(p);
      if (builtin) {
        ctx.out.line(`@builtin(${builtin}) ${p.name}: ${typeStr(p.type)},`);
      } else {
        const loc = locationOf(p) ?? nextLoc++;
        const interp = interpolation(p);
        const decor = (interp ? `${interp} ` : "") + `@location(${loc})`;
        ctx.out.line(`${decor} ${p.name}: ${typeStr(p.type)},`);
        ctx.bindings.outputs.push({ name: p.name, location: loc, type: p.type });
      }
    }
    ctx.out.dec();
    ctx.out.line("};");
  }

  ctx.out.blank();
  return { inputStructName, outputStructName, hasOutputStruct: hasOutputs };
}

function isBuiltin(p: EntryParameter): boolean {
  return p.decorations.some((d) => d.kind === "Builtin");
}
function builtinName(p: EntryParameter): string | undefined {
  const d = p.decorations.find((x) => x.kind === "Builtin");
  return d && d.kind === "Builtin" ? d.value : undefined;
}
function locationOf(p: EntryParameter): number | undefined {
  const d = p.decorations.find((x) => x.kind === "Location");
  return d && d.kind === "Location" ? d.value : undefined;
}
function interpolation(p: EntryParameter): string | undefined {
  const d = p.decorations.find((x) => x.kind === "Interpolation");
  if (!d || d.kind !== "Interpolation") return undefined;
  switch (d.mode) {
    case "smooth": return undefined; // wgsl default
    case "flat": return "@interpolate(flat)";
    case "centroid": return "@interpolate(perspective, centroid)";
    case "sample": return "@interpolate(perspective, sample)";
    case "no-perspective": return "@interpolate(linear)";
  }
}
function workgroupSize(e: EntryDef): readonly [number, number, number] | undefined {
  const d = e.decorations.find((x) => x.kind === "WorkgroupSize");
  if (!d || d.kind !== "WorkgroupSize") return undefined;
  return [d.x, d.y ?? 1, d.z ?? 1];
}

function emitFunction(
  ctx: Ctx,
  sig: { name: string; returnType: Type; parameters: readonly { name: string; type: Type; modifier: "in" | "inout" }[] },
  body: Stmt,
): void {
  const params = sig.parameters
    .map((p) => `${p.name}: ${p.modifier === "inout" ? `ptr<function, ${typeStr(p.type)}>` : typeStr(p.type)}`)
    .join(", ");
  const ret = sig.returnType.kind === "Void" ? "" : ` -> ${typeStr(sig.returnType)}`;
  ctx.out.line(`fn ${sig.name}(${params})${ret} {`);
  ctx.out.inc();
  emitStmt(ctx, body);
  ctx.out.dec();
  ctx.out.line("}");
  ctx.out.blank();
}

function emitEntryFunction(ctx: Ctx, e: EntryDef, io: EntryIO): void {
  const stageAttr = e.stage === "vertex" ? "@vertex"
    : e.stage === "fragment" ? "@fragment"
    : "@compute";
  if (e.stage === "compute") {
    const ws = workgroupSize(e) ?? [1, 1, 1];
    ctx.out.line(`@compute @workgroup_size(${ws.join(", ")})`);
  } else {
    ctx.out.line(stageAttr);
  }

  const params: string[] = [];
  if (io.inputStructName) {
    params.push(`in: ${io.inputStructName}`);
  }
  // Builtin inputs come as separate parameters.
  for (const p of e.inputs) {
    const b = builtinName(p);
    if (b) params.push(`@builtin(${b}) ${p.name}: ${typeStr(p.type)}`);
  }
  // Compute stage's `arguments` array carries workgroup-style builtins as well.
  for (const p of e.arguments) {
    const b = builtinName(p);
    if (b) params.push(`@builtin(${b}) ${p.name}: ${typeStr(p.type)}`);
    else params.push(`${p.name}: ${typeStr(p.type)}`);
  }

  const ret = io.hasOutputStruct && io.outputStructName ? ` -> ${io.outputStructName}` : "";
  ctx.out.line(`fn ${e.name}(${params.join(", ")})${ret} {`);
  ctx.out.inc();

  // Declare an output struct local variable so writes can target its fields.
  if (io.hasOutputStruct && io.outputStructName) {
    ctx.out.line(`var out: ${io.outputStructName};`);
  }
  emitStmt(ctx, e.body);
  if (io.hasOutputStruct && io.outputStructName) {
    ctx.out.line("return out;");
  }
  ctx.out.dec();
  ctx.out.line("}");
}

function capitalise(s: string): string {
  return s.length > 0 ? s[0]!.toUpperCase() + s.slice(1) : s;
}

// ─────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────

function emitStmt(ctx: Ctx, s: Stmt): void {
  switch (s.kind) {
    case "Nop":
      return;
    case "Expression":
      ctx.out.line(`${expr(s.value)};`);
      return;
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      const keyword = s.var.mutable ? "var" : "let";
      ctx.out.line(`${keyword} ${s.var.name}: ${typeStr(s.var.type)}${init};`);
      return;
    }
    case "Write":
      ctx.out.line(`${lexpr(s.target)} = ${expr(s.value)};`);
      return;
    case "WriteOutput": {
      // Stage outputs in WGSL are fields on the synthetic `out` struct.
      const idx = s.index ? `[${expr(s.index)}]` : "";
      ctx.out.line(`out.${s.name}${idx} = ${rexpr(s.value)};`);
      return;
    }
    case "Increment":
      ctx.out.line(`${lexpr(s.target)} += 1;`);
      return;
    case "Decrement":
      ctx.out.line(`${lexpr(s.target)} -= 1;`);
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
      ctx.out.line(s.scope === "workgroup" ? "workgroupBarrier();"
                  : s.scope === "storage" ? "storageBarrier();"
                  : "subgroupBarrier();"); // subgroup is a draft extension
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
      // WGSL has C-style `for`.
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
      // WGSL has no do-while; lower to loop {}.
      ctx.out.line("loop {");
      ctx.out.inc();
      emitStmt(ctx, s.body);
      ctx.out.line(`if (!(${expr(s.cond)})) { break; }`);
      ctx.out.dec();
      ctx.out.line("}");
      return;
    case "Switch":
      emitSwitch(ctx, s.value, s.cases, s.default);
      return;
  }
}

function forStmtText(s: Stmt): string {
  switch (s.kind) {
    case "Nop": return ";";
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      const keyword = s.var.mutable ? "var" : "let";
      return `${keyword} ${s.var.name}: ${typeStr(s.var.type)}${init};`;
    }
    case "Expression": return `${expr(s.value)};`;
    case "Write": return `${lexpr(s.target)} = ${expr(s.value)};`;
    case "Increment": return `${lexpr(s.target)} += 1;`;
    case "Decrement": return `${lexpr(s.target)} -= 1;`;
    default: return ";";
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
    ctx.out.line(`case ${literal(c.literal)}: {`);
    ctx.out.inc(); emitStmt(ctx, c.body); ctx.out.dec();
    ctx.out.line("}");
  }
  if (defaultStmt) {
    ctx.out.line("default: {");
    ctx.out.inc(); emitStmt(ctx, defaultStmt); ctx.out.dec();
    ctx.out.line("}");
  }
  ctx.out.dec();
  ctx.out.line("}");
}

// ─────────────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────────────

const BIN_OP: Partial<Record<Expr["kind"], string>> = {
  Add: "+", Sub: "-", Mul: "*", Div: "/", Mod: "%",
  And: "&&", Or: "||",
  BitAnd: "&", BitOr: "|", BitXor: "^",
  ShiftLeft: "<<", ShiftRight: ">>",
  Eq: "==", Neq: "!=", Lt: "<", Le: "<=", Gt: ">", Ge: ">=",
};

// Set at the start of each emitWgsl call; lets `expr` resolve uniform
// names that live inside a uniform-block struct.
let bufferOf: Map<string, string> = new Map();

export function expr(e: Expr): string {
  switch (e.kind) {
    case "Var":
      return e.var.name;
    case "Const":
      return literal(e.value);
    case "ReadInput": {
      if (e.scope === "Input") return `in.${e.name}`;
      const buf = bufferOf.get(e.name);
      if (buf) return `${buf}.${e.name}`;
      return e.name;
    }
    case "Call":
      return `${e.fn.signature.name}(${e.args.map(expr).join(", ")})`;
    case "CallIntrinsic":
      return `${e.op.emit.wgsl}(${e.args.map(expr).join(", ")})`;
    case "Conditional":
      // WGSL has `select(false, true, cond)` — argument order!
      return `select(${expr(e.ifFalse)}, ${expr(e.ifTrue)}, ${expr(e.cond)})`;
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
      const op = BIN_OP[e.kind] ?? "*";
      return `(${expr(e.lhs)} ${op} ${expr(e.rhs)})`;
    }
    case "Transpose":
      return `transpose(${expr(e.value)})`;
    case "Inverse":
      // WGSL has no `inverse`; user must supply one. We emit a call to a
      // helper named `inverse_${type}` and rely on legalisation to inject
      // the helper if used.
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
      return `/* MatrixRow not directly representable in WGSL */ ${expr(e.matrix)}[${expr(e.row)}]`;
    case "MatrixCol":
      return `${expr(e.matrix)}[${expr(e.col)}]`;
    case "NewVector":
      return `${typeStr(e.type)}(${e.components.map(expr).join(", ")})`;
    case "NewMatrix":
      return `${typeStr(e.type)}(${e.elements.map(expr).join(", ")})`;
    case "MatrixFromRows":
      // Same caveat as GLSL: WGSL constructors are column-major, so a
      // from-rows form needs a transpose.
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
      return `(${expr(e.lhs)} < ${expr(e.rhs)})`;
    case "VecLe":
      return `(${expr(e.lhs)} <= ${expr(e.rhs)})`;
    case "VecGt":
      return `(${expr(e.lhs)} > ${expr(e.rhs)})`;
    case "VecGe":
      return `(${expr(e.lhs)} >= ${expr(e.rhs)})`;
    case "VecEq":
      return `(${expr(e.lhs)} == ${expr(e.rhs)})`;
    case "VecNeq":
      return `(${expr(e.lhs)} != ${expr(e.rhs)})`;
    case "Field":
      return `${expr(e.target)}.${e.name}`;
    case "Item":
      return `${expr(e.target)}[${expr(e.index)}]`;
    case "DebugPrintf":
      return `/* debugPrintf */ 0`;
  }
}

export function lexpr(l: LExpr): string {
  switch (l.kind) {
    case "LVar":
      return l.var.name;
    case "LField":
      return `${lexpr(l.target)}.${l.name}`;
    case "LItem":
      return `${lexpr(l.target)}[${expr(l.index)}]`;
    case "LSwizzle":
      return `${lexpr(l.target)}.${l.comps.join("")}`;
    case "LMatrixElement":
      return `${lexpr(l.matrix)}[${expr(l.col)}][${expr(l.row)}]`;
    case "LInput":
      // Stage outputs are written via the synthetic `out` struct.
      return l.scope === "Output" ? `out.${l.name}${l.index ? `[${expr(l.index)}]` : ""}` : l.name;
  }
}

function rexpr(r: RExpr): string {
  if (r.kind === "Expr") return expr(r.value);
  return `array<${typeStr(r.arrayType.kind === "Array" ? r.arrayType.element : r.arrayType)}, ${r.values.length}>(${r.values.map(expr).join(", ")})`;
}

function literal(l: Literal): string {
  switch (l.kind) {
    case "Bool": return l.value ? "true" : "false";
    case "Int": return l.signed ? `${l.value | 0}i` : `${l.value >>> 0}u`;
    case "Float": return formatFloat(l.value);
    case "Null": return "0";
  }
}

function formatFloat(n: number): string {
  if (!Number.isFinite(n)) {
    return n === Infinity ? "(1.0/0.0)" : n === -Infinity ? "(-1.0/0.0)" : "(0.0/0.0)";
  }
  // WGSL needs `f` suffix for f32 literals to disambiguate from abstract floats.
  if (Number.isInteger(n)) return `${n}.0f`;
  return `${n}f`;
}

// ─────────────────────────────────────────────────────────────────────
// Type printer
// ─────────────────────────────────────────────────────────────────────

function typeStr(t: Type): string {
  switch (t.kind) {
    case "Void": return "void";
    case "Bool": return "bool";
    case "Int": return t.signed ? "i32" : "u32";
    case "Float": return "f32";
    case "Vector": {
      const dim = t.dim;
      switch (t.element.kind) {
        case "Bool": return `vec${dim}<bool>`;
        case "Int": return `vec${dim}<${t.element.signed ? "i32" : "u32"}>`;
        case "Float": return `vec${dim}<f32>`;
        default: throw new Error(`unsupported vector element ${t.element.kind}`);
      }
    }
    case "Matrix":
      return `mat${t.cols}x${t.rows}<${typeStr(t.element)}>`;
    case "Array":
      return t.length === "runtime"
        ? `array<${typeStr(t.element)}>`
        : `array<${typeStr(t.element)}, ${t.length}>`;
    case "Struct":
      return t.name;
    case "Sampler":
      return t.comparison ? "sampler_comparison" : "sampler";
    case "Texture": {
      const arr = t.arrayed ? "_array" : "";
      const ms = t.multisampled ? "_multisampled" : "";
      const dim = t.target === "1D" ? "1d"
                : t.target === "2D" ? "2d"
                : t.target === "3D" ? "3d"
                : t.target === "Cube" ? "cube"
                : t.target === "2DArray" ? "2d"
                : t.target === "CubeArray" ? "cube"
                : t.target;
      const sampledTy = t.sampled.kind === "Float" ? "f32" : (t.sampled.signed ? "i32" : "u32");
      return `texture${ms}_${dim}${arr}<${sampledTy}>`;
    }
    case "AtomicI32": return "atomic<i32>";
    case "AtomicU32": return "atomic<u32>";
    case "Intrinsic":
      throw new Error(`emitWgsl: unresolved intrinsic type "${t.name}"`);
  }
}

export const _internal = { typeStr, expr, lexpr, literal };
