// Tests for composeStages, pruneCrossStage, and reduceUniforms.

import { describe, expect, it } from "vitest";
import {
  Tf32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
  type Module,
  type Stmt,
  type Type,
  type Var,
} from "@aardworx/wombat.shader/ir";
import {
  composeStages,
  linkFragmentOutputs,
  pruneCrossStage,
  reduceUniforms,
} from "@aardworx/wombat.shader/passes";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const v = (va: Var): Expr => ({ kind: "Var", var: va, type: va.type });
const readU = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Uniform", name, type });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });

function entriesOf(m: Module): EntryDef[] {
  return m.values.flatMap((x) => x.kind === "Entry" ? [x.entry] : []);
}

// ─── reduceUniforms ──────────────────────────────────────────────────

describe("reduceUniforms", () => {
  it("drops uniforms no entry references", () => {
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "u_used", type: Tf32 }, { name: "u_dead", type: Tf32 }] },
        {
          kind: "Entry", entry: {
            name: "main", stage: "fragment", inputs: [], outputs: [],
            arguments: [], returnType: Tvoid,
            body: { kind: "Expression", value: readU("u_used", Tf32) },
            decorations: [],
          },
        },
      ],
    };
    const out = reduceUniforms(m);
    const u = out.values.find((x) => x.kind === "Uniform")!;
    expect(u.kind).toBe("Uniform");
    if (u.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["u_used"]);
    }
  });

  it("drops samplers no entry references", () => {
    const samplerType: Type = { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false };
    const m: Module = {
      types: [],
      values: [
        { kind: "Sampler", binding: { group: 0, slot: 0 }, name: "u_used_tex", type: samplerType },
        { kind: "Sampler", binding: { group: 0, slot: 1 }, name: "u_dead_tex", type: samplerType },
        {
          kind: "Entry", entry: {
            name: "main", stage: "fragment", inputs: [], outputs: [],
            arguments: [], returnType: Tvoid,
            body: { kind: "Expression", value: readU("u_used_tex", samplerType) },
            decorations: [],
          },
        },
      ],
    };
    const out = reduceUniforms(m);
    const samplers = out.values.filter((x) => x.kind === "Sampler");
    expect(samplers.length).toBe(1);
    if (samplers[0]?.kind === "Sampler") {
      expect(samplers[0].name).toBe("u_used_tex");
    }
  });
});

// ─── pruneCrossStage ─────────────────────────────────────────────────

describe("pruneCrossStage", () => {
  it("drops vertex outputs the fragment doesn't read", () => {
    // Vertex writes v_color and v_normal; fragment reads only v_color.
    const a_pos: EntryDef["inputs"][number] = {
      name: "a_pos", type: Tvec3f, semantic: "Position",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const a_color: EntryDef["inputs"][number] = {
      name: "a_color", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 1 }],
    };
    const a_normal: EntryDef["inputs"][number] = {
      name: "a_normal", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 2 }],
    };
    const v_color: EntryDef["outputs"][number] = {
      name: "v_color", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const v_normal: EntryDef["outputs"][number] = {
      name: "v_normal", type: Tvec3f, semantic: "Normal",
      decorations: [{ kind: "Location", value: 1 }],
    };
    const gl_Position: EntryDef["outputs"][number] = {
      name: "gl_Position", type: Tvec4f, semantic: "Position",
      decorations: [{ kind: "Builtin", value: "position" }],
    };

    const vertex: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [a_pos, a_color, a_normal],
      outputs: [gl_Position, v_color, v_normal],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential", body: [
          {
            kind: "WriteOutput", name: "gl_Position",
            value: {
              kind: "Expr",
              value: {
                kind: "NewVector",
                components: [readI("a_pos", Tvec3f), constF(1)],
                type: Tvec4f,
              },
            },
          },
          {
            kind: "WriteOutput", name: "v_color",
            value: { kind: "Expr", value: readI("a_color", Tvec3f) },
          },
          {
            kind: "WriteOutput", name: "v_normal",
            value: { kind: "Expr", value: readI("a_normal", Tvec3f) },
          },
        ],
      },
      decorations: [],
    };

    const fragment: EntryDef = {
      name: "fsMain", stage: "fragment",
      inputs: [v_color], // only v_color — note v_normal is NOT in fragment inputs
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: {
          kind: "Expr",
          value: {
            kind: "NewVector",
            components: [readI("v_color", Tvec3f), constF(1)],
            type: Tvec4f,
          },
        },
      },
      decorations: [],
    };

    const m: Module = {
      types: [],
      values: [
        { kind: "Entry", entry: vertex },
        { kind: "Entry", entry: fragment },
      ],
    };

    const pruned = pruneCrossStage(m);
    const v2 = entriesOf(pruned).find((e) => e.stage === "vertex")!;
    expect(v2.outputs.map((o) => o.name)).toEqual(["gl_Position", "v_color"]);
    // The body should have the v_normal write removed (DCE may collapse Sequential).
    const seq = v2.body;
    const stmts = seq.kind === "Sequential" ? seq.body : [seq];
    const writeNames = stmts.flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writeNames).toContain("gl_Position");
    expect(writeNames).toContain("v_color");
    expect(writeNames).not.toContain("v_normal");
  });

  it("preserves builtin outputs even if no fragment exists", () => {
    const gl_Position: EntryDef["outputs"][number] = {
      name: "gl_Position", type: Tvec4f, semantic: "Position",
      decorations: [{ kind: "Builtin", value: "position" }],
    };
    const v_extra: EntryDef["outputs"][number] = {
      name: "v_extra", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const vertex: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [], outputs: [gl_Position, v_extra],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "Sequential", body: [
          { kind: "WriteOutput", name: "gl_Position", value: { kind: "Expr", value: { kind: "NewVector", components: [constF(0), constF(0), constF(0), constF(1)], type: Tvec4f } } },
          { kind: "WriteOutput", name: "v_extra",     value: { kind: "Expr", value: { kind: "NewVector", components: [constF(0), constF(0), constF(0)], type: Tvec3f } } },
        ],
      },
      decorations: [],
    };
    const m: Module = { types: [], values: [{ kind: "Entry", entry: vertex }] };
    const pruned = pruneCrossStage(m);
    const out = entriesOf(pruned)[0]!.outputs;
    // No fragment / compute consumes v_extra → drops; gl_Position kept.
    expect(out.map((o) => o.name)).toEqual(["gl_Position"]);
  });
});

// ─── composeStages ────────────────────────────────────────────────────

describe("composeStages", () => {
  it("two same-stage fragments fuse with carrier vars", () => {
    const out_a: EntryDef["outputs"][number] = {
      name: "fragA_color", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const fragA: EntryDef = {
      name: "fA", stage: "fragment",
      inputs: [], outputs: [out_a],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "fragA_color",
        value: { kind: "Expr", value: { kind: "NewVector", components: [constF(1), constF(0), constF(0)], type: Tvec3f } },
      },
      decorations: [],
    };
    const fragB: EntryDef = {
      name: "fB", stage: "fragment",
      // B reads what A wrote.
      inputs: [{ name: "fragA_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "outColor",
        value: {
          kind: "Expr",
          value: {
            kind: "NewVector",
            components: [readI("fragA_color", Tvec3f), constF(1)],
            type: Tvec4f,
          },
        },
      },
      decorations: [],
    };

    const m: Module = {
      types: [],
      values: [
        { kind: "Entry", entry: fragA },
        { kind: "Entry", entry: fragB },
      ],
    };

    const composed = composeStages(m);
    const entries = entriesOf(composed);
    expect(entries.length).toBe(1);
    const merged = entries[0]!;
    expect(merged.stage).toBe("fragment");
    // Inputs of A (none) plus inputs of B that aren't piped (none).
    expect(merged.inputs.length).toBe(0);
    // Outputs: only `outColor`. `fragA_color` is internalised as a
    // carrier because `fragB` reads it.
    expect(merged.outputs.map((o) => o.name)).toEqual(["outColor"]);
    // Body should declare a `_pipe_fragA_color` carrier.
    const seq = merged.body.kind === "Sequential" ? merged.body.body : [merged.body];
    const decls = seq.flatMap((s) => s.kind === "Declare" ? [s.var.name] : []);
    expect(decls).toContain("_pipe_fragA_color");
  });

  it("vertex + fragment are not fused but coexist as a pipeline", () => {
    const vertex: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [], outputs: [],
      arguments: [], returnType: Tvoid,
      body: { kind: "Nop" },
      decorations: [],
    };
    const fragment: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [], outputs: [],
      arguments: [], returnType: Tvoid,
      body: { kind: "Nop" },
      decorations: [],
    };
    const m: Module = { types: [], values: [
      { kind: "Entry", entry: vertex },
      { kind: "Entry", entry: fragment },
    ]};
    const composed = composeStages(m);
    expect(entriesOf(composed).length).toBe(2);
  });
});

// ─── pipeline integration ────────────────────────────────────────────

describe("integration: composeStages → pruneCrossStage → reduceUniforms", () => {
  it("dead carrier from fused fragments is pruned", () => {
    // Two fragments fuse; the carrier output isn't read by anything
    // downstream. After prune, it should disappear; if the only use of
    // a uniform was inside the dropped chain, reduceUniforms drops it
    // too (we make the dropped chain explicitly mention a dead uniform).
    const out_a: EntryDef["outputs"][number] = {
      name: "tmp", type: Tvec3f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    };
    const u_dead: import("@aardworx/wombat.shader/ir").UniformDecl = {
      name: "u_dead", type: Tf32,
    };
    const u_live: import("@aardworx/wombat.shader/ir").UniformDecl = {
      name: "u_live", type: Tf32,
    };
    const fragA: EntryDef = {
      name: "fA", stage: "fragment",
      inputs: [], outputs: [out_a],
      arguments: [], returnType: Tvoid,
      body: {
        kind: "WriteOutput", name: "tmp",
        value: {
          kind: "Expr",
          value: {
            // tmp = new V3f(u_dead, u_dead, u_dead) — uses u_dead only.
            kind: "NewVector",
            components: [readU("u_dead", Tf32), readU("u_dead", Tf32), readU("u_dead", Tf32)],
            type: Tvec3f,
          },
        },
      },
      decorations: [],
    };
    const fragB: EntryDef = {
      name: "fB", stage: "fragment",
      inputs: [{ name: "tmp", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      // fB ignores its `tmp` input and uses u_live instead.
      body: {
        kind: "WriteOutput", name: "outColor",
        value: {
          kind: "Expr",
          value: {
            kind: "NewVector",
            components: [readU("u_live", Tf32), constF(0), constF(0), constF(1)],
            type: Tvec4f,
          },
        },
      },
      decorations: [],
    };
    const m: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [u_dead, u_live] },
        { kind: "Entry", entry: fragA },
        { kind: "Entry", entry: fragB },
      ],
    };

    const composed = composeStages(m);
    // FShade-style composition keeps every output, so `tmp` survives
    // (no carrier was created — fB doesn't read it). The
    // fragment-output linker drops it when the target framebuffer
    // signature only declares `outColor`. After link + reduceUniforms,
    // u_dead has nothing keeping it live.
    const merged = entriesOf(composed)[0]!;
    expect(merged.stage).toBe("fragment");
    // composeStages keeps `tmp`; it's not the linker's job yet.
    expect(merged.outputs.map((o) => o.name).sort()).toEqual(["outColor", "tmp"]);

    const linked = linkFragmentOutputs(composed, {
      locations: new Map([["outColor", 0]]),
    });
    const linkedMerged = entriesOf(linked)[0]!;
    expect(linkedMerged.outputs.map((o) => o.name)).toEqual(["outColor"]);
    // After link DCE the dropped output's u_dead reference is gone.
    expect(JSON.stringify(linkedMerged)).not.toContain("u_dead");

    // reduceUniforms now drops u_dead from the Uniform decl.
    const reduced = reduceUniforms(linked);
    const u = reduced.values.find((x) => x.kind === "Uniform");
    expect(u?.kind).toBe("Uniform");
    if (u?.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["u_live"]);
    }
  });
});
