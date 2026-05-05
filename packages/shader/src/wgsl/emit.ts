// IR → WGSL emitter (WebGPU target).
//
// Conventions vs. the GLSL emitter:
//
// Reserved-word handling. WGSL has a long list of keywords +
// reserved identifiers (https://www.w3.org/TR/WGSL/#keyword-summary
// + #reserved-words). A user-supplied name like `meta`, `mat`,
// `texture`, `module`, etc. would otherwise emerge as a parse
// error ("Expected an Identifier, but got a ReservedWord"). The
// emitter routes every user name through `safeName(...)` which
// prefixes reserved tokens with `_w_`. Var identity is preserved
// across declaration / reference sites because the rename is
// purely a function of the original name.
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
} from "../ir/index.js";

export interface EmitResult {
  readonly source: string;
  readonly bindings: BindingMap;
  readonly meta: BackendMeta;
  /**
   * Per-emitted-line *segments* — a list of `(col, span)` entries
   * per line, sorted by ascending column. `[]` for unmapped lines
   * (blank lines, scaffolding). The runtime pipes this directly
   * into `buildSourceMap` to produce a v3 source map.
   */
  readonly lineSegments: ReadonlyArray<readonly { col: number; span: import("../ir/index.js").Span }[]>;

  /**
   * @deprecated Use `lineSegments`. Per-line first-segment view kept
   * for backwards compatibility; produces a strictly-coarser map.
   */
  readonly lineSpans: ReadonlyArray<import("../ir/index.js").Span | undefined>;
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

/** WGSL reserved words and keywords (per spec + commonly-clashing
 *  user identifiers). Any user name in this set is rewritten to
 *  `_w_<name>` before emission. The rewrite is total per name, so
 *  declaration and reference sites stay in sync. */
const WGSL_RESERVED: ReadonlySet<string> = new Set<string>([
  // keywords
  "alias", "break", "case", "const", "const_assert", "continue",
  "continuing", "default", "diagnostic", "discard", "else", "enable",
  "false", "fn", "for", "if", "let", "loop", "override", "requires",
  "return", "struct", "switch", "true", "var", "while",
  // type keywords
  "array", "atomic", "bool", "f32", "f16", "i32", "u32", "mat2x2",
  "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2",
  "mat4x3", "mat4x4", "ptr", "sampler", "sampler_comparison",
  "texture_1d", "texture_2d", "texture_2d_array", "texture_3d",
  "texture_cube", "texture_cube_array", "texture_multisampled_2d",
  "texture_storage_1d", "texture_storage_2d", "texture_storage_2d_array",
  "texture_storage_3d", "texture_depth_2d", "texture_depth_2d_array",
  "texture_depth_cube", "texture_depth_cube_array",
  "texture_depth_multisampled_2d", "vec2", "vec3", "vec4",
  // shorthand vec/mat
  "mat2", "mat3", "mat4",
  // reserved (per spec; subset of the most common collisions)
  "NULL", "Self", "abstract", "active", "alignas", "alignof", "as",
  "asm", "asm_fragment", "async", "attribute", "auto", "await",
  "become", "binding_array", "cast", "catch", "class", "co_await",
  "co_return", "co_yield", "coherent", "column_major", "common",
  "compile", "compile_fragment", "concept", "const_cast", "consteval",
  "constexpr", "constinit", "crate", "debugger", "decltype", "delete",
  "demote", "demote_to_helper", "do", "dynamic_cast", "enum",
  "explicit", "export", "extends", "extern", "external", "fallthrough",
  "filter", "final", "finally", "friend", "from", "fxgroup", "get",
  "goto", "groupshared", "highp", "impl", "implements", "import",
  "inline", "instanceof", "interface", "layout", "lowp", "macro",
  "macro_rules", "match", "mediump", "meta", "mod", "module", "move",
  "mut", "mutable", "namespace", "new", "nil", "noexcept", "noinline",
  "nointerpolation", "noperspective", "null", "nullptr", "of",
  "operator", "package", "packoffset", "partition", "pass", "patch",
  "pixelfragment", "precise", "precision", "premerge", "priv",
  "protected", "pub", "public", "readonly", "ref", "regardless",
  "register", "reinterpret_cast", "require", "resource", "restrict",
  "self", "set", "shared", "sizeof", "smooth", "snorm", "static",
  "static_assert", "static_cast", "std", "subroutine", "super",
  "target", "template", "this", "thread_local", "throw", "trait",
  "try", "type", "typedef", "typeid", "typename", "typeof", "union",
  "unless", "unorm", "unsafe", "unsized", "use", "using", "varying",
  "virtual", "volatile", "wgsl", "where", "with", "writeonly", "yield",
]);

/** Rewrite a user identifier into something WGSL won't reject. */
export function safeName(name: string): string {
  return WGSL_RESERVED.has(name) ? `_w_${name}` : name;
}

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
    lineSegments: w.lineSegments,
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

type Span = import("../ir/index.js").Span;

interface LineSegment {
  /** 0-based generated column where this segment starts. */
  readonly col: number;
  readonly span: Span;
}

/**
 * Writer that builds an emitted text buffer plus a per-line list of
 * source-map segments. Emit code uses two patterns:
 *
 *   1. Whole-line: `setSpan(stmt.span); line("text")` — registers
 *      one segment at column 0 of the line.
 *   2. Piecewise: `setSpan(stmt.span); write("a = "); writeSpan(rhs.span); write(expr(rhs)); endLine(";")` —
 *      registers one segment per `setSpan`/`writeSpan` call at the
 *      column the next character would land at.
 *
 * Segments accumulate in `lineSegments[generatedLine]`; whole-line
 * setSpan always places the line's first segment at col 0.
 */
class Writer {
  private readonly parts: string[] = [];
  private indent = 0;
  private currentSpan: Span | undefined;
  /** Pending content for the current line (not yet committed). */
  private pendingText = "";
  /** Pending segments for the current line (not yet committed). */
  private pendingSegments: LineSegment[] = [];
  /** Per-line segment lists. Index 0 = first emitted line. */
  readonly lineSegments: LineSegment[][] = [];

  line(s: string): void {
    if (this.pendingText.length === 0 && this.pendingSegments.length === 0) {
      // Fast path: whole-line emit. Register one segment at col 0
      // if a span has been set.
      const segments: LineSegment[] = [];
      if (this.currentSpan !== undefined) {
        segments.push({ col: 0, span: this.currentSpan });
      }
      this.parts.push("    ".repeat(this.indent) + s + "\n");
      this.lineSegments.push(segments);
      return;
    }
    // Piecewise path: flush pending then append.
    this.write(s);
    this.endLine();
  }

  /** Append `s` to the current line without finalising it. */
  write(s: string): void {
    if (this.pendingText.length === 0) {
      this.pendingText = "    ".repeat(this.indent);
      // Default-segment for current span at the start of this line.
      if (this.currentSpan !== undefined) {
        this.pendingSegments.push({ col: 0, span: this.currentSpan });
      }
    }
    this.pendingText += s;
  }

  /** Register a sub-line segment at the current cursor for `span`. */
  writeSpan(span: Span | undefined): void {
    if (span === undefined) return;
    const col = this.pendingText.length === 0
      ? "    ".repeat(this.indent).length
      : this.pendingText.length;
    // De-dupe consecutive segments referring to the same span.
    const last = this.pendingSegments[this.pendingSegments.length - 1];
    if (last && last.span === span && last.col === col) return;
    this.pendingSegments.push({ col, span });
  }

  /** Finalise the current line, optionally appending `s` first. */
  endLine(s: string = ""): void {
    if (s.length > 0) this.write(s);
    if (this.pendingText.length === 0) {
      this.parts.push("\n");
      this.lineSegments.push([]);
    } else {
      this.parts.push(this.pendingText + "\n");
      this.lineSegments.push(this.pendingSegments);
    }
    this.pendingText = "";
    this.pendingSegments = [];
  }

  blank(): void {
    if (this.pendingText.length > 0 || this.pendingSegments.length > 0) {
      this.endLine();
    }
    this.parts.push("\n");
    this.lineSegments.push([]);
  }
  inc(): void { this.indent++; }
  dec(): void { this.indent = Math.max(0, this.indent - 1); }
  setSpan(s: Span | undefined): void {
    this.currentSpan = s;
  }

  /**
   * Backwards-compatible view: one span per line (the line's first
   * segment, or `undefined` if the line is unmapped). Older
   * source-map builders use this; new builders consume
   * `lineSegments` directly.
   */
  get lineSpans(): (Span | undefined)[] {
    return this.lineSegments.map(ss => ss[0]?.span);
  }

  toString(): string { return this.parts.join(""); }
}

function emitStructDecl(ctx: Ctx, t: { kind: "Struct"; name: string; fields: readonly { type: Type; name: string }[] }): void {
  if (ctx.structs.has(t.name)) return;
  ctx.structs.add(t.name);
  ctx.out.line(`struct ${t.name} {`);
  ctx.out.inc();
  for (const f of t.fields) {
    ctx.out.line(`${safeName(f.name)}: ${typeStr(f.type)},`);
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
      for (const u of group) ctx.out.line(`${safeName(u.name)}: ${typeStr(u.type)},`);
      ctx.out.dec();
      ctx.out.line("};");
      const grp = group[0]?.group ?? autoGroup;
      const slot = group[0]?.slot ?? nextSlot++;
      ctx.out.line(`@group(${grp}) @binding(${slot}) var<uniform> ${safeName(buffer)}: ${structName};`);
      for (const u of group) {
        ctx.bindings.uniforms.push({ name: `${buffer}.${u.name}`, group: grp, slot, type: u.type });
      }
    } else {
      for (const u of group) {
        const grp = u.group ?? autoGroup;
        const slot = u.slot ?? nextSlot++;
        ctx.out.line(`@group(${grp}) @binding(${slot}) var<uniform> ${safeName(u.name)}: ${typeStr(u.type)};`);
        ctx.bindings.uniforms.push({ name: u.name, group: grp, slot, type: u.type });
      }
    }
  }
}

function emitSampler(ctx: Ctx, name: string, group: number, slot: number, type: Type): void {
  ctx.out.line(`@group(${group}) @binding(${slot}) var ${safeName(name)}: ${typeStr(type)};`);
  ctx.bindings.samplers.push({ name, group, slot, type });
}

function emitStorageBuffer(
  ctx: Ctx, name: string, group: number, slot: number, layout: Type, access: "read" | "read_write",
): void {
  ctx.out.line(`@group(${group}) @binding(${slot}) var<storage, ${access}> ${safeName(name)}: ${typeStr(layout)};`);
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
      ctx.out.line(`${decor} ${safeName(p.name)}: ${typeStr(p.type)},`);
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
        ctx.out.line(`@builtin(${builtin}) ${safeName(p.name)}: ${typeStr(p.type)},`);
      } else {
        const loc = locationOf(p) ?? nextLoc++;
        const interp = interpolation(p);
        const decor = (interp ? `${interp} ` : "") + `@location(${loc})`;
        ctx.out.line(`${decor} ${safeName(p.name)}: ${typeStr(p.type)},`);
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
    .map((p) => `${safeName(p.name)}: ${p.modifier === "inout" ? `ptr<function, ${typeStr(p.type)}>` : typeStr(p.type)}`)
    .join(", ");
  const ret = sig.returnType.kind === "Void" ? "" : ` -> ${typeStr(sig.returnType)}`;
  ctx.out.line(`fn ${safeName(sig.name)}(${params})${ret} {`);
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
  // Builtin inputs come as separate parameters. Use the semantic
  // (`position`, `front_facing`, ...) as the parameter NAME so the
  // body's `ReadInput("Builtin", semantic)` reads (which the emitter
  // lowers as the bare semantic name) line up. This mirrors how the
  // compute-stage builtin args are emitted further down.
  for (const p of e.inputs) {
    const b = builtinName(p);
    if (b) params.push(`@builtin(${b}) ${b}: ${typeStr(p.type)}`);
  }
  // Compute stage's `arguments` array carries workgroup-style builtins
  // as well. The body's `ReadInput("Builtin", semantic)` references the
  // semantic name (e.g. `global_invocation_id`) — emit the parameter
  // name as the semantic so signature and body line up.
  for (const p of e.arguments) {
    const b = builtinName(p);
    if (b) params.push(`@builtin(${b}) ${b}: ${typeStr(p.type)}`);
    else params.push(`${safeName(p.name)}: ${typeStr(p.type)}`);
  }

  const ret = io.hasOutputStruct && io.outputStructName ? ` -> ${io.outputStructName}` : "";
  ctx.out.line(`fn ${safeName(e.name)}(${params.join(", ")})${ret} {`);
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
  if (s.span) ctx.out.setSpan(s.span);
  switch (s.kind) {
    case "Nop":
      return;
    case "Expression": {
      // Per-expression span: emit `expr;` with a segment registered
      // at the expression's own span position.
      const v = s.value as { span?: Span };
      ctx.out.write("");
      if (v.span) ctx.out.writeSpan(v.span);
      ctx.out.endLine(`${expr(s.value)};`);
      return;
    }
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      const keyword = s.var.mutable ? "var" : "let";
      ctx.out.line(`${keyword} ${safeName(s.var.name)}: ${typeStr(s.var.type)}${init};`);
      return;
    }
    case "Write": {
      // `target = value;` — register a segment at the RHS so that
      // diagnostics on the value side jump to the right TS column.
      ctx.out.write(`${lexpr(s.target)} = `);
      const v = s.value as { span?: Span };
      if (v.span) ctx.out.writeSpan(v.span);
      ctx.out.endLine(`${expr(s.value)};`);
      return;
    }
    case "WriteOutput": {
      // Stage outputs in WGSL are fields on the synthetic `out` struct.
      const idx = s.index ? `[${expr(s.index)}]` : "";
      ctx.out.write(`out.${safeName(s.name)}${idx} = `);
      const v = s.value as { span?: Span };
      if (v.span) ctx.out.writeSpan(v.span);
      ctx.out.endLine(`${rexpr(s.value)};`);
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
    case "Loop":
      ctx.out.line("loop {");
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
    case "Nop": return ";";
    case "Declare": {
      const init = s.init ? ` = ${rexpr(s.init)}` : "";
      const keyword = s.var.mutable ? "var" : "let";
      return `${keyword} ${safeName(s.var.name)}: ${typeStr(s.var.type)}${init};`;
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
      return safeName(e.var.name);
    case "Const":
      return literal(e.value);
    case "ReadInput": {
      if (e.scope === "Input") return `in.${safeName(e.name)}`;
      const buf = bufferOf.get(e.name);
      if (buf) return `${safeName(buf)}.${safeName(e.name)}`;
      return safeName(e.name);
    }
    case "Call":
      return `${safeName(e.fn.signature.name)}(${e.args.map(expr).join(", ")})`;
    case "CallIntrinsic": {
      // Atomic ops take a pointer to the storage element as their first
      // argument: `atomicAdd(&buf[i], 1)`.
      if (e.op.atomic && e.args.length > 0) {
        const first = expr(e.args[0]!);
        const rest = e.args.slice(1).map(expr);
        return `${e.op.emit.wgsl}(&${first}${rest.length ? ", " + rest.join(", ") : ""})`;
      }
      return `${e.op.emit.wgsl}(${e.args.map(expr).join(", ")})`;
    }
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
      return `${expr(e.target)}.${safeName(e.name)}`;
    case "Item":
      return `${expr(e.target)}[${expr(e.index)}]`;
    case "DebugPrintf":
      return `/* debugPrintf */ 0`;
  }
}

export function lexpr(l: LExpr): string {
  switch (l.kind) {
    case "LVar":
      return safeName(l.var.name);
    case "LField":
      return `${lexpr(l.target)}.${safeName(l.name)}`;
    case "LItem":
      return `${lexpr(l.target)}[${expr(l.index)}]`;
    case "LSwizzle":
      return `${lexpr(l.target)}.${l.comps.join("")}`;
    case "LMatrixElement":
      return `${lexpr(l.matrix)}[${expr(l.col)}][${expr(l.row)}]`;
    case "LInput":
      // Stage outputs are written via the synthetic `out` struct.
      return l.scope === "Output"
        ? `out.${safeName(l.name)}${l.index ? `[${expr(l.index)}]` : ""}`
        : safeName(l.name);
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
      const dim = t.target === "1D" ? "1d"
                : t.target === "2D" ? "2d"
                : t.target === "3D" ? "3d"
                : t.target === "Cube" ? "cube"
                : t.target === "2DArray" ? "2d"
                : t.target === "CubeArray" ? "cube"
                : t.target === "2DMS" ? "2d"
                : t.target === "2DMSArray" ? "2d"
                : t.target;
      // Depth/comparison textures: `texture_depth_*` (no element arg).
      if (t.comparison) {
        return `texture_depth_${dim}${arr}`;
      }
      // Multisampled textures: `texture_multisampled_*<T>` (no _array
      // form is exposed by WGSL today; 2DMSArray collapses to 2DMS
      // emit at this layer).
      if (t.multisampled) {
        const sampledTy = t.sampled.kind === "Float" ? "f32" : (t.sampled.signed ? "i32" : "u32");
        return `texture_multisampled_${dim}<${sampledTy}>`;
      }
      const sampledTy = t.sampled.kind === "Float" ? "f32" : (t.sampled.signed ? "i32" : "u32");
      return `texture_${dim}${arr}<${sampledTy}>`;
    }
    case "StorageTexture": {
      const arr = t.arrayed ? "_array" : "";
      const dim = t.target === "2D" ? "2d"
                : t.target === "3D" ? "3d"
                : t.target === "Cube" ? "cube"
                : t.target === "1D" ? "1d"
                : t.target;
      return `texture_storage_${dim}${arr}<${t.format}, ${t.access}>`;
    }
    case "AtomicI32": return "atomic<i32>";
    case "AtomicU32": return "atomic<u32>";
    case "Intrinsic":
      throw new Error(`emitWgsl: unresolved intrinsic type "${t.name}"`);
  }
}

export const _internal = { typeStr, expr, lexpr, literal };
