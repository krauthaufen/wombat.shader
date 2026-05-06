// `ProgramInterface` — full backend-ready signature of a compiled
// effect. Modelled after FShade's GLSLProgramInterface.
//
// Everything callers need to bind data to the GPU is here:
//   - vertex attribute names, locations, and a WebGPU-style format
//     string for vertexAttribPointer / GPUVertexBufferLayout
//   - fragment output names + locations
//   - uniform blocks with resolved field offsets and total buffer size
//     (std140 for GLSL, WGSL layout rules for WGSL)
//   - free-floating uniforms (GLSL only — WebGPU mandates blocks)
//   - samplers / textures / storage buffers with their bind groups
//   - workgroup size for compute stages
//
// No "uniform name → location?" string-keyed runtime lookups required;
// the user reads the interface, finds the field, writes its bytes at
// the offset.

import type {
  BuiltinSemantic,
  EntryDef,
  EntryParameter,
  Module,
  Stage,
  Type,
  ValueDef,
} from "../ir/index.js";
import { computeLayout, type FieldLayout, type LayoutInfo, type LayoutTarget } from "./layout.js";

export type Target = "glsl" | "wgsl";

export interface ProgramInterface {
  readonly target: Target;
  readonly stages: readonly StageInfo[];
  readonly attributes: readonly AttributeInfo[];     // vertex stage inputs
  readonly fragmentOutputs: readonly OutputInfo[];   // fragment stage outputs
  readonly uniforms: readonly LooseUniformInfo[];    // GLSL: loose `uniform` decls; WGSL: empty
  readonly uniformBlocks: readonly UniformBlockInfo[];
  readonly samplers: readonly SamplerInfo[];
  readonly textures: readonly TextureInfo[];
  readonly storageBuffers: readonly StorageBufferInfo[];
}

export interface StageInfo {
  readonly stage: Stage;
  readonly entryName: string;
  readonly source: string;
  /** Compute only: workgroup_size. */
  readonly workgroupSize?: readonly [number, number, number] | undefined;
  /**
   * Every input parameter to this stage, including @builtin ones.
   * For the vertex stage these are also surfaced as `attributes` at
   * the top of the interface; this list is the full per-stage view.
   */
  readonly inputs: readonly StageParameterInfo[];
  /**
   * Every output parameter from this stage, including @builtin ones
   * (e.g. `position` for vertex, `frag_depth` for fragment).
   */
  readonly outputs: readonly StageParameterInfo[];
  /**
   * Compute stages: workgroup-style builtin arguments
   * (global_invocation_id, etc.).
   */
  readonly arguments: readonly StageParameterInfo[];
}

export interface StageParameterInfo {
  readonly name: string;
  readonly type: Type;
  readonly semantic: string;
  /** Location for non-builtin parameters; absent for @builtin params. */
  readonly location?: number | undefined;
  /** @builtin name (`position`, `vertex_index`, …) for builtin parameters. */
  readonly builtin?: BuiltinSemantic | undefined;
  /** Interpolation mode (vertex outputs / fragment inputs). */
  readonly interpolation?: "smooth" | "flat" | "centroid" | "sample" | "no-perspective" | undefined;
  /**
   * WebGPU `GPUVertexFormat` string (vertex-stage inputs only).
   * `byteSize` reports the same value the matching `AttributeInfo`
   * would carry; here for callers who only walk the per-stage view.
   */
  readonly vertexFormat?: string | undefined;
  readonly byteSize?: number | undefined;
}

export interface AttributeInfo {
  readonly name: string;
  readonly location: number;
  readonly type: Type;
  /** WebGPU `GPUVertexFormat` string, also describes the WebGL2 type+size. */
  readonly format: string;
  /** Number of components (4 for vec4, 1 for scalar, 16 for mat4 — but mat4 attribs are uncommon). */
  readonly components: number;
  /** Total bytes occupied by one such attribute. */
  readonly byteSize: number;
}

export interface OutputInfo {
  readonly name: string;
  readonly location: number;
  readonly type: Type;
}

export interface LooseUniformInfo {
  readonly name: string;
  readonly type: Type;
}

export interface UniformBlockInfo {
  /** GLSL UBO name / WGSL synthetic struct name. */
  readonly name: string;
  /** Group/slot pair. GLSL UBOs use slot only (group always 0). */
  readonly group: number;
  readonly slot: number;
  /** Total buffer size in bytes (std140 or WGSL uniform layout). */
  readonly size: number;
  readonly fields: readonly UniformFieldInfo[];
}

export interface UniformFieldInfo {
  readonly name: string;
  readonly type: Type;
  readonly offset: number;
  readonly size: number;
  readonly align: number;
  readonly stride?: number | undefined;
}

export interface SamplerInfo {
  readonly name: string;
  readonly group: number;
  readonly slot: number;
  readonly type: Type;
}

export interface TextureInfo {
  readonly name: string;
  readonly group: number;
  readonly slot: number;
  readonly type: Type;
}

export interface StorageBufferInfo {
  readonly name: string;
  readonly group: number;
  readonly slot: number;
  readonly access: "read" | "read_write";
  readonly size: number;
  readonly fields: readonly UniformFieldInfo[];
}

// ─────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────

export interface StageSourceInfo {
  readonly stage: Stage;
  readonly entryName: string;
  readonly source: string;
  readonly workgroupSize?: readonly [number, number, number] | undefined;
}

export interface BuildInterfaceInput {
  readonly target: Target;
  readonly module: Module;
  readonly stages: readonly StageSourceInfo[];
}

export function buildInterface(input: BuildInterfaceInput): ProgramInterface {
  const layoutTarget: LayoutTarget = input.target === "glsl" ? "glsl-std140" : "wgsl-uniform";

  const stages = enrichStageInfo(input.module, input.stages);
  const attributes = collectAttributes(input.module);
  const fragmentOutputs = collectFragmentOutputs(input.module);
  // Walk the module once with one slot counter shared across uniform
  // blocks / samplers / textures / storage buffers, mirroring the WGSL
  // emitter's centralised assignment so the runtime BindGroupLayout
  // matches what `var<...>` declarations the shader actually has.
  const slot = { next: 0 };
  const { uniforms, uniformBlocks } = collectUniforms(input.module, layoutTarget, slot);
  const { samplers, textures } = collectSamplersAndTextures(input.module, slot);
  const storageBuffers = collectStorageBuffers(input.module, input.target, slot);

  return {
    target: input.target,
    stages,
    attributes,
    fragmentOutputs,
    uniforms,
    uniformBlocks,
    samplers,
    textures,
    storageBuffers,
  };
}

// ─────────────────────────────────────────────────────────────────────
// Enrich the per-stage source info with full inputs/outputs/arguments.
// ─────────────────────────────────────────────────────────────────────

function enrichStageInfo(module: Module, stages: readonly StageSourceInfo[]): StageInfo[] {
  const out: StageInfo[] = [];
  for (const s of stages) {
    const entry = entryByName(module, s.entryName);
    if (!entry) {
      // Should not happen — the stages array is built from the same Module.
      out.push({ ...s, inputs: [], outputs: [], arguments: [] });
      continue;
    }
    const isVertex = entry.stage === "vertex";
    const inputs = entry.inputs.map((p, i) => paramToInfo(p, i, isVertex));
    const outputs = entry.outputs.map((p, i) => paramToInfo(p, i, /* needsFmt */ false));
    const args = entry.arguments.map((p, i) => paramToInfo(p, i, /* needsFmt */ false));
    out.push({
      ...s,
      inputs,
      outputs,
      arguments: args,
    });
  }
  return out;
}

function paramToInfo(p: EntryParameter, fallbackLocation: number, needsVertexFormat: boolean): StageParameterInfo {
  const builtin = p.decorations.find((d) => d.kind === "Builtin");
  const locDec = p.decorations.find((d) => d.kind === "Location");
  const interp = p.decorations.find((d) => d.kind === "Interpolation");

  const result: {
    name: string;
    type: Type;
    semantic: string;
    location?: number | undefined;
    builtin?: BuiltinSemantic | undefined;
    interpolation?: StageParameterInfo["interpolation"];
    vertexFormat?: string | undefined;
    byteSize?: number | undefined;
  } = {
    name: p.name,
    type: p.type,
    semantic: p.semantic,
  };

  if (builtin && builtin.kind === "Builtin") {
    result.builtin = builtin.value;
  } else {
    result.location = locDec && locDec.kind === "Location" ? locDec.value : fallbackLocation;
  }
  if (interp && interp.kind === "Interpolation") {
    result.interpolation = interp.mode;
  }
  if (needsVertexFormat && (!builtin || builtin.kind !== "Builtin")) {
    const fmt = vertexFormat(p.type);
    result.vertexFormat = fmt.format;
    result.byteSize = fmt.byteSize;
  }
  return result as StageParameterInfo;
}

function entryByName(module: Module, name: string): EntryDef | undefined {
  for (const v of module.values) {
    if (v.kind === "Entry" && v.entry.name === name) return v.entry;
  }
  return undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Per-section collectors
// ─────────────────────────────────────────────────────────────────────

function collectAttributes(module: Module): AttributeInfo[] {
  const out: AttributeInfo[] = [];
  const vsEntries = entriesOfStage(module, "vertex");
  for (const e of vsEntries) {
    let nextLoc = 0;
    for (const p of e.inputs) {
      if (p.decorations.some((d) => d.kind === "Builtin")) continue;
      // Always auto-assign — emit.ts emits contiguous vertex-input
      // locations in declaration order, so the runtime view has to
      // mirror that.
      const location = nextLoc++;
      const fmt = vertexFormat(p.type);
      out.push({
        name: p.name,
        location,
        type: p.type,
        format: fmt.format,
        components: fmt.components,
        byteSize: fmt.byteSize,
      });
    }
  }
  return dedupeByName(out);
}

function collectFragmentOutputs(module: Module): OutputInfo[] {
  const out: OutputInfo[] = [];
  for (const e of entriesOfStage(module, "fragment")) {
    let nextLoc = 0;
    for (const p of e.outputs) {
      if (p.decorations.some((d) => d.kind === "Builtin")) continue;
      const locDec = p.decorations.find((d) => d.kind === "Location");
      const location = locDec && locDec.kind === "Location" ? locDec.value : nextLoc++;
      out.push({ name: p.name, location, type: p.type });
    }
  }
  return dedupeByName(out);
}

function collectUniforms(
  module: Module,
  target: LayoutTarget,
  slotCounter: { next: number },
): {
  uniforms: LooseUniformInfo[];
  uniformBlocks: UniformBlockInfo[];
} {
  const loose: LooseUniformInfo[] = [];
  const blocks = new Map<string, { group: number; slot: number; fields: UniformFieldInfo[]; size: number; align: number }>();

  for (const v of module.values) {
    if (v.kind !== "Uniform") continue;
    // Group uniforms by `buffer` name.
    const buckets = new Map<string | undefined, typeof v.uniforms[number][]>();
    for (const u of v.uniforms) {
      const key = u.buffer;
      const arr = buckets.get(key);
      if (arr) arr.push(u);
      else buckets.set(key, [u]);
    }
    for (const [bufferName, decls] of buckets) {
      if (bufferName === undefined) {
        // Loose uniforms — only meaningful for GLSL targets.
        for (const u of decls) loose.push({ name: u.name, type: u.type });
        continue;
      }
      const fields: UniformFieldInfo[] = [];
      let offset = 0;
      let maxAlign = 1;
      for (const u of decls) {
        const fl = computeLayout(u.type, target);
        offset = roundUp(offset, fl.align);
        fields.push({
          name: u.name,
          type: u.type,
          offset,
          size: fl.size,
          align: fl.align,
          ...(fl.stride !== undefined ? { stride: fl.stride } : {}),
        });
        offset += fl.size;
        if (fl.align > maxAlign) maxAlign = fl.align;
      }
      const size = roundUp(offset, maxAlign);
      // Always auto-assign — explicit group/slot annotations on
      // upstream UniformDecls are ignored. Multiple effects can pin
      // (group=0, slot=0) independently and collide; centralising the
      // assignment here keeps the runtime layout consistent.
      const group = 0;
      const slot  = slotCounter.next++;
      blocks.set(bufferName, { group, slot, fields, size, align: maxAlign });
    }
  }
  const uniformBlocks = [...blocks.entries()].map(([name, b]) => ({
    name,
    group: b.group,
    slot: b.slot,
    size: b.size,
    fields: b.fields,
  }));
  return { uniforms: loose, uniformBlocks };
}

function collectSamplersAndTextures(
  module: Module,
  slotCounter: { next: number },
): {
  samplers: SamplerInfo[];
  textures: TextureInfo[];
} {
  const samplers: SamplerInfo[] = [];
  const textures: TextureInfo[] = [];
  for (const v of module.values) {
    if (v.kind !== "Sampler") continue;
    const entry = {
      name: v.name,
      group: 0,
      slot: slotCounter.next++,
      type: v.type,
    };
    if (v.type.kind === "Texture") textures.push(entry);
    else samplers.push(entry);
  }
  return { samplers, textures };
}

function collectStorageBuffers(
  module: Module,
  target: Target,
  slotCounter: { next: number },
): StorageBufferInfo[] {
  const out: StorageBufferInfo[] = [];
  const layoutTarget: LayoutTarget = target === "glsl" ? "glsl-std140" : "wgsl-storage";
  for (const v of module.values) {
    if (v.kind !== "StorageBuffer") continue;
    const layout = computeLayout(v.layout, layoutTarget);
    const fields = (layout.fields ?? []).map((f) => fieldToInfo(f));
    out.push({
      name: v.name,
      group: 0,
      slot: slotCounter.next++,
      access: v.access,
      size: layout.size,
      fields,
    });
  }
  return out;
}

function fieldToInfo(f: FieldLayout): UniformFieldInfo {
  return {
    name: f.name,
    type: f.type,
    offset: f.offset,
    size: f.layout.size,
    align: f.layout.align,
    ...(f.layout.stride !== undefined ? { stride: f.layout.stride } : {}),
  };
}

// ─────────────────────────────────────────────────────────────────────
// Vertex format mapping
// ─────────────────────────────────────────────────────────────────────

interface VertexFormat {
  readonly format: string;
  readonly components: number;
  readonly byteSize: number;
}

function vertexFormat(t: Type): VertexFormat {
  if (t.kind === "Float") return { format: "float32", components: 1, byteSize: 4 };
  if (t.kind === "Int") {
    return t.signed
      ? { format: "sint32", components: 1, byteSize: 4 }
      : { format: "uint32", components: 1, byteSize: 4 };
  }
  if (t.kind === "Vector") {
    const dim = t.dim;
    if (t.element.kind === "Float") return { format: `float32x${dim}`, components: dim, byteSize: 4 * dim };
    if (t.element.kind === "Int") {
      const prefix = t.element.signed ? "sint32" : "uint32";
      return { format: `${prefix}x${dim}`, components: dim, byteSize: 4 * dim };
    }
  }
  // Fallback for other types (bool vectors etc.) — describe as float.
  return { format: "float32x4", components: 4, byteSize: 16 };
}

// ─────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────

function entriesOfStage(module: Module, stage: Stage): EntryDef[] {
  return module.values.flatMap((v) => v.kind === "Entry" && v.entry.stage === stage ? [v.entry] : []);
}

function dedupeByName<T extends { name: string }>(arr: readonly T[]): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const x of arr) {
    if (seen.has(x.name)) continue;
    seen.add(x.name);
    out.push(x);
  }
  return out;
}

function roundUp(value: number, multiple: number): number {
  return Math.ceil(value / multiple) * multiple;
}

// Re-export ValueDef for the public API consumers who want to round-trip.
export type { ValueDef };
