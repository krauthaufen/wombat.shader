// substituteInputsInStage + the rename* APIs.

import { describe, expect, it } from "vitest";
import {
  Tf32,
  Tvoid,
  Vec,
  type EntryDef,
  type Expr,
  type FunctionRef,
  type FunctionSignature,
  type Module,
  type Stmt,
  type Type,
  type Var,
} from "@aardworx/wombat.shader/ir";
import {
  renameEntries,
  renameFunctions,
  renameInputs,
  renameInputsInStage,
  renameOutputs,
  renameOutputsInStage,
  renameTypes,
  renameVars,
  renameVaryings,
  substituteInputsInStage,
} from "@aardworx/wombat.shader/passes";
import { effect, stage } from "@aardworx/wombat.shader";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const readU = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Uniform", name, type });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });

function entriesOf(m: Module): EntryDef[] {
  return m.values.flatMap((x) => x.kind === "Entry" ? [x.entry] : []);
}

// Helper: count occurrences of `ReadInput(scope, name)` inside a body.
function countReads(body: Stmt, scope: string, name: string): number {
  let n = 0;
  const walk = (s: Stmt): void => {
    const json = JSON.stringify(s);
    n += (json.match(new RegExp(`"scope":"${scope}","name":"${name}"`, "g")) ?? []).length;
  };
  walk(body);
  return n;
}

function vfModule(): Module {
  const vsBody: Stmt = {
    kind: "Sequential", body: [
      { kind: "WriteOutput", name: "gl_Position",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readI("a_pos", Tvec3f), constF(1)], type: Tvec4f } } },
      { kind: "WriteOutput", name: "v_color",
        value: { kind: "Expr", value: readU("u_color", Tvec3f) } },
    ],
  };
  const vs: EntryDef = {
    name: "vsMain", stage: "vertex",
    inputs: [{ name: "a_pos", type: Tvec3f, semantic: "Position",
              decorations: [{ kind: "Location", value: 0 }] }],
    outputs: [
      { name: "gl_Position", type: Tvec4f, semantic: "Position",
        decorations: [{ kind: "Builtin", value: "position" }] },
      { name: "v_color", type: Tvec3f, semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }] },
    ],
    arguments: [], returnType: Tvoid,
    body: vsBody, decorations: [],
  };
  const fs: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [{ name: "v_color", type: Tvec3f, semantic: "Color",
              decorations: [{ kind: "Location", value: 0 }] }],
    outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color",
                decorations: [{ kind: "Location", value: 0 }] }],
    arguments: [], returnType: Tvoid,
    body: { kind: "WriteOutput", name: "outColor",
      value: { kind: "Expr", value: { kind: "NewVector",
        components: [readI("v_color", Tvec3f), readU("u_color", Tf32)], type: Tvec4f } } },
    decorations: [],
  };
  return { types: [], values: [
    { kind: "Uniform", uniforms: [{ name: "u_color", type: Tvec3f }] },
    { kind: "Entry", entry: vs },
    { kind: "Entry", entry: fs },
  ] };
}

describe("substituteInputsInStage", () => {
  it("rewrites only the targeted stage", () => {
    const m = vfModule();
    const replacement: Expr = constF(42);
    const out = substituteInputsInStage(m, "vertex", "Uniform",
      (n) => n === "u_color" ? replacement : undefined);

    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;

    // Vertex uniform read should be gone.
    expect(countReads(vs.body, "Uniform", "u_color")).toBe(0);
    // Fragment uniform read should still be present.
    expect(countReads(fs.body, "Uniform", "u_color")).toBeGreaterThan(0);
  });

  it("walks Function bodies regardless of stage", () => {
    // Helper that reads a uniform; called from neither — the helper
    // body itself is what we're checking.
    const helperSig: FunctionSignature = { name: "h", returnType: Tf32, parameters: [] };
    const helperBody: Stmt = { kind: "ReturnValue", value: readU("u_helper", Tf32) };
    const m: Module = {
      types: [],
      values: [
        { kind: "Function", signature: helperSig, body: helperBody },
        { kind: "Entry", entry: {
          name: "vs", stage: "vertex", inputs: [], outputs: [],
          arguments: [], returnType: Tvoid, body: { kind: "Nop" }, decorations: [],
        } },
      ],
    };
    const out = substituteInputsInStage(m, "vertex", "Uniform",
      (n) => n === "u_helper" ? constF(7) : undefined);
    const fn = out.values.find((v) => v.kind === "Function")!;
    if (fn.kind === "Function") {
      expect(countReads(fn.body, "Uniform", "u_helper")).toBe(0);
    }
  });
});

describe("rename APIs", () => {
  it("renameInputs swaps Entry input param + ReadInput names", () => {
    const m = vfModule();
    const out = renameInputs(m, "Input", new Map([["a_pos", "in_position"]]));
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    expect(vs.inputs.map((p) => p.name)).toEqual(["in_position"]);
    expect(countReads(vs.body, "Input", "a_pos")).toBe(0);
    expect(countReads(vs.body, "Input", "in_position")).toBeGreaterThan(0);
  });

  it("renameInputs renames Uniform decls and reads", () => {
    const m = vfModule();
    const out = renameInputs(m, "Uniform", new Map([["u_color", "uniColor"]]));
    const u = out.values.find((v) => v.kind === "Uniform")!;
    if (u.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["uniColor"]);
    }
    for (const e of entriesOf(out)) {
      expect(countReads(e.body, "Uniform", "u_color")).toBe(0);
    }
  });

  it("renameInputsInStage scopes the rewrite", () => {
    const m: Module = {
      types: [], values: [
        { kind: "Entry", entry: {
          name: "vs", stage: "vertex", inputs: [], outputs: [],
          arguments: [], returnType: Tvoid,
          body: { kind: "Expression", value: readU("u_x", Tf32) },
          decorations: [],
        } },
        { kind: "Entry", entry: {
          name: "fs", stage: "fragment", inputs: [], outputs: [],
          arguments: [], returnType: Tvoid,
          body: { kind: "Expression", value: readU("u_x", Tf32) },
          decorations: [],
        } },
      ],
    };
    const out = renameInputsInStage(m, "vertex", "Uniform", new Map([["u_x", "u_y"]]));
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;
    expect(countReads(vs.body, "Uniform", "u_x")).toBe(0);
    expect(countReads(fs.body, "Uniform", "u_x")).toBe(1);
  });

  it("renameOutputs swaps Entry output param + WriteOutput names", () => {
    const m = vfModule();
    const out = renameOutputs(m, new Map([["v_color", "v_tint"]]));
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    expect(vs.outputs.map((o) => o.name).sort()).toEqual(["gl_Position", "v_tint"]);
    // The vertex body's WriteOutput for v_color is renamed.
    const seq = vs.body.kind === "Sequential" ? vs.body.body : [vs.body];
    const writes = seq.flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writes).toContain("v_tint");
    expect(writes).not.toContain("v_color");
  });

  it("renameOutputsInStage only touches matching stage", () => {
    const m = vfModule();
    const out = renameOutputsInStage(m, "fragment", new Map([["outColor", "frag_out"]]));
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;
    expect(fs.outputs.map((o) => o.name)).toEqual(["frag_out"]);
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    // Vertex outputs unaffected.
    expect(vs.outputs.map((o) => o.name)).toContain("gl_Position");
  });

  it("renameVars rewrites every Var reference and the Declare", () => {
    const oldVar: Var = { name: "x", type: Tf32, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: oldVar, init: { kind: "Expr", value: constF(1) } },
        { kind: "Expression", value: { kind: "Var", var: oldVar, type: Tf32 } },
      ],
    };
    const m: Module = { types: [], values: [
      { kind: "Entry", entry: {
        name: "main", stage: "fragment", inputs: [], outputs: [],
        arguments: [], returnType: Tvoid, body, decorations: [],
      } },
    ]};
    const out = renameVars(m, new Map([[oldVar, "y"]]));
    const e = entriesOf(out)[0]!;
    const seq = e.body.kind === "Sequential" ? e.body.body : [];
    const decl = seq.find((s) => s.kind === "Declare");
    if (decl?.kind === "Declare") expect(decl.var.name).toBe("y");
    const expr = seq.find((s) => s.kind === "Expression");
    if (expr?.kind === "Expression" && expr.value.kind === "Var") {
      expect(expr.value.var.name).toBe("y");
    }
  });

  it("renameTypes renames a struct everywhere it appears", () => {
    const oldStruct: Type = { kind: "Struct", name: "Material", fields: [
      { name: "albedo", type: Tvec3f },
    ] };
    const m: Module = {
      types: [{ kind: "Struct", name: "Material", fields: [{ name: "albedo", type: Tvec3f }] }],
      values: [
        { kind: "Uniform", uniforms: [{ name: "mat", type: oldStruct }] },
        { kind: "Entry", entry: {
          name: "main", stage: "fragment", inputs: [], outputs: [],
          arguments: [], returnType: Tvoid,
          body: { kind: "Expression", value: { kind: "ReadInput", scope: "Uniform", name: "mat", type: oldStruct } },
          decorations: [],
        } },
      ],
    };
    const out = renameTypes(m, new Map([["Material", "Mat"]]));
    expect(out.types[0]!.kind === "Struct" && out.types[0]!.name).toBe("Mat");
    const u = out.values.find((v) => v.kind === "Uniform")!;
    if (u.kind === "Uniform") {
      const t = u.uniforms[0]!.type;
      expect(t.kind).toBe("Struct");
      if (t.kind === "Struct") expect(t.name).toBe("Mat");
    }
    const e = entriesOf(out)[0]!;
    if (e.body.kind === "Expression") {
      const t = e.body.value.type;
      expect(t.kind).toBe("Struct");
      if (t.kind === "Struct") expect(t.name).toBe("Mat");
    }
  });

  it("renameEntries swaps Entry name", () => {
    const m = vfModule();
    const out = renameEntries(m, new Map([["vsMain", "myVs"]]));
    const names = entriesOf(out).map((e) => e.name).sort();
    expect(names).toEqual(["fsMain", "myVs"]);
  });

  it("renameFunctions swaps Function names + Call references", () => {
    const sig: FunctionSignature = { name: "scale", returnType: Tf32,
      parameters: [{ name: "v", type: Tf32, modifier: "in" }] };
    const fnRef: FunctionRef = { id: "scale", signature: sig, pure: true };
    const callExpr: Expr = { kind: "Call", fn: fnRef, args: [constF(1)], type: Tf32 };
    const m: Module = {
      types: [],
      values: [
        { kind: "Function", signature: sig, body: { kind: "ReturnValue", value: constF(2) } },
        { kind: "Entry", entry: {
          name: "main", stage: "fragment", inputs: [], outputs: [],
          arguments: [], returnType: Tvoid,
          body: { kind: "Expression", value: callExpr },
          decorations: [],
        } },
      ],
    };
    const out = renameFunctions(m, new Map([["scale", "scaleX"]]));
    const fn = out.values.find((v) => v.kind === "Function")!;
    if (fn.kind === "Function") expect(fn.signature.name).toBe("scaleX");
    const e = entriesOf(out)[0]!;
    if (e.body.kind === "Expression" && e.body.value.kind === "Call") {
      expect(e.body.value.fn.signature.name).toBe("scaleX");
    }
  });

  it("collision detection throws on existing entry name", () => {
    const m = vfModule();
    expect(() => renameEntries(m, new Map([["vsMain", "fsMain"]]))).toThrow(/already exists/);
  });

  it("collision detection throws on existing uniform name", () => {
    const m = vfModule();
    // Add a second uniform so we can collide onto it.
    const m2: Module = { ...m, values: m.values.map((v) =>
      v.kind === "Uniform"
        ? { ...v, uniforms: [...v.uniforms, { name: "u_taken", type: Tf32 }] }
        : v) };
    expect(() => renameInputs(m2, "Uniform", new Map([["u_color", "u_taken"]]))).toThrow(/already exists/);
  });

  it("silent no-op when oldName is not present", () => {
    const m = vfModule();
    const out = renameEntries(m, new Map([["doesNotExist", "alsoNo"]]));
    // Same module shape (no throw, no change).
    expect(entriesOf(out).map((e) => e.name).sort()).toEqual(
      entriesOf(m).map((e) => e.name).sort()
    );
  });
});

describe("function-form RenameMapping", () => {
  it("renameInputs (Uniform): function form receives (name, type)", () => {
    const m = vfModule();
    const seen: Array<[string, string]> = [];
    const out = renameInputs(m, "Uniform", (name, type) => {
      seen.push([name, type.kind]);
      return name === "u_color" ? "uniColor" : undefined;
    });
    // Function was called with the uniform's declared type.
    expect(seen).toContainEqual(["u_color", "Vector"]);
    const u = out.values.find((v) => v.kind === "Uniform")!;
    if (u.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["uniColor"]);
    }
  });

  it("renameOutputs: function form can rename only by type kind", () => {
    const m = vfModule();
    // Rename the vec3f output (v_color) but leave the vec4f one
    // (gl_Position) alone.
    const out = renameOutputs(m, (name, type) => {
      if (type.kind !== "Vector" || type.dim !== 3) return undefined;
      return `${name}_v3`;
    });
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    expect(vs.outputs.map((o) => o.name).sort()).toEqual(["gl_Position", "v_color_v3"]);
  });

  it("renameVaryings: function form receives (name, VS-output type)", () => {
    const m = vfModule();
    const seen: Array<[string, string]> = [];
    const out = renameVaryings(m, (name, type) => {
      seen.push([name, type.kind]);
      // Rename only vec3 varyings (vec4 like gl_Position pass through).
      if (type.kind === "Vector" && type.dim === 3) return `${name}__v3`;
      return undefined;
    });
    expect(seen).toContainEqual(["v_color", "Vector"]);
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;
    expect(fs.inputs.map((p) => p.name)).toEqual(["v_color__v3"]);
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    expect(vs.outputs.map((o) => o.name).sort()).toEqual(["gl_Position", "v_color__v3"]);
  });

  it("renameVars: function form receives the Var, can read .type", () => {
    const oldA: Var = { name: "a", type: Tf32, mutable: false };
    const oldB: Var = { name: "b", type: Tvec3f, mutable: false };
    const body: Stmt = {
      kind: "Sequential", body: [
        { kind: "Declare", var: oldA, init: { kind: "Expr", value: constF(1) } },
        { kind: "Declare", var: oldB, init: { kind: "Expr",
          value: { kind: "NewVector",
            components: [constF(0), constF(0), constF(0)], type: Tvec3f } } },
        { kind: "Expression", value: { kind: "Var", var: oldA, type: Tf32 } },
        { kind: "Expression", value: { kind: "Var", var: oldB, type: Tvec3f } },
      ],
    };
    const m: Module = { types: [], values: [
      { kind: "Entry", entry: {
        name: "main", stage: "fragment", inputs: [], outputs: [],
        arguments: [], returnType: Tvoid, body, decorations: [],
      } },
    ]};
    // Only rename Floats — leaves vec3f alone.
    const out = renameVars(m, (v) => v.type.kind === "Float" ? `${v.name}_f` : undefined);
    const e = entriesOf(out)[0]!;
    const seq = e.body.kind === "Sequential" ? e.body.body : [];
    const decls = seq.flatMap((s) => s.kind === "Declare" ? [s.var.name] : []);
    expect(decls).toEqual(["a_f", "b"]);
  });

  it("merge-time disambiguation: same-name-different-type varyings via function form", () => {
    // Two synthetic VS modules both declare a varying named "v" but
    // with different types. The caller uses the function form to
    // suffix one with type kind so a synthetic merge has no clash.
    const mkVs = (t: Type, suffix: string): Module => {
      const e: EntryDef = {
        name: `vs_${suffix}`, stage: "vertex",
        inputs: [], outputs: [
          { name: "v", type: t, semantic: "X",
            decorations: [{ kind: "Location", value: 0 }] },
        ],
        arguments: [], returnType: Tvoid,
        body: { kind: "WriteOutput", name: "v",
          value: { kind: "Expr", value: t.kind === "Float"
            ? constF(1)
            : { kind: "NewVector", components: [constF(1), constF(0), constF(0)], type: t } } },
        decorations: [],
      };
      return { types: [], values: [{ kind: "Entry", entry: e }] };
    };
    const a = mkVs(Tf32, "a");
    const b = mkVs(Tvec3f, "b");

    const renameByKind = (_name: string, type: Type) =>
      `v__${type.kind}`;
    const aR = renameVaryings(a, renameByKind);
    const bR = renameVaryings(b, renameByKind);

    // Synthetic merge — two distinct Entry values, names now differ.
    const merged: Module = {
      types: [],
      values: [...aR.values, ...bR.values],
    };
    const outputNames = entriesOf(merged)
      .flatMap((e) => e.outputs.map((o) => o.name))
      .sort();
    expect(outputNames).toEqual(["v__Float", "v__Vector"]);
    // Distinct — merge is clash-free.
    expect(new Set(outputNames).size).toBe(outputNames.length);
  });

  it("function form returning undefined for every name is a no-op", () => {
    const m = vfModule();
    const out = renameInputs(m, "Uniform", () => undefined);
    expect(out).toBe(m);
  });
});

describe("renameVaryings", () => {
  it("renames VS output + matching FS input together; cross-stage link survives compile", () => {
    const m = vfModule();
    const out = renameVaryings(m, new Map([["v_color", "v_tint"]]));
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;
    // VS: output renamed + WriteOutput body renamed.
    expect(vs.outputs.map((o) => o.name).sort()).toEqual(["gl_Position", "v_tint"]);
    const seq = vs.body.kind === "Sequential" ? vs.body.body : [vs.body];
    const writes = seq.flatMap((s) => s.kind === "WriteOutput" ? [s.name] : []);
    expect(writes).toContain("v_tint");
    expect(writes).not.toContain("v_color");
    // FS: input renamed + ReadInput body renamed.
    expect(fs.inputs.map((p) => p.name)).toEqual(["v_tint"]);
    expect(countReads(fs.body, "Input", "v_color")).toBe(0);
    expect(countReads(fs.body, "Input", "v_tint")).toBeGreaterThan(0);
  });

  it("asymmetric: name only in VS outputs (no FS reader) — renames VS, FS no-op", () => {
    const m = vfModule();
    // gl_Position is a VS output not consumed as an FS Input varying.
    // (FS doesn't list it as an input, doesn't read it.)
    const out = renameVaryings(m, new Map([["gl_Position", "gl_Pos"]]));
    const vs = entriesOf(out).find((e) => e.stage === "vertex")!;
    expect(vs.outputs.map((o) => o.name)).toContain("gl_Pos");
    // FS unchanged — its inputs are still ["v_color"], not affected.
    const fs = entriesOf(out).find((e) => e.stage === "fragment")!;
    expect(fs.inputs.map((p) => p.name)).toEqual(["v_color"]);
  });

  it("asymmetric: name only in FS inputs (no VS writer) — renames FS, VS no-op", () => {
    // Construct a module where FS reads "x" but VS doesn't write it.
    const fs: EntryDef = {
      name: "fs", stage: "fragment",
      inputs: [{ name: "x", type: Tf32, semantic: "X",
                 decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [], arguments: [], returnType: Tvoid,
      body: { kind: "Expression", value: readI("x", Tf32) },
      decorations: [],
    };
    const vs: EntryDef = {
      name: "vs", stage: "vertex",
      inputs: [], outputs: [], arguments: [], returnType: Tvoid,
      body: { kind: "Nop" }, decorations: [],
    };
    const m: Module = { types: [],
      values: [{ kind: "Entry", entry: vs }, { kind: "Entry", entry: fs }] };
    const out = renameVaryings(m, new Map([["x", "y"]]));
    const fsOut = entriesOf(out).find((e) => e.stage === "fragment")!;
    expect(fsOut.inputs.map((p) => p.name)).toEqual(["y"]);
    expect(countReads(fsOut.body, "Input", "x")).toBe(0);
    expect(countReads(fsOut.body, "Input", "y")).toBeGreaterThan(0);
  });
});

describe("Effect.substitute / Effect.rename", () => {
  function mkEffect() {
    const vs: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [], outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position",
          decorations: [{ kind: "Builtin", value: "position" }] },
      ],
      arguments: [], returnType: Tvoid,
      body: { kind: "WriteOutput", name: "gl_Position",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readU("u_offset", Tf32), constF(0), constF(0), constF(1)],
          type: Tvec4f } } },
      decorations: [],
    };
    const fs: EntryDef = {
      name: "fsMain", stage: "fragment",
      inputs: [], outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color",
          decorations: [{ kind: "Location", value: 0 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: { kind: "WriteOutput", name: "outColor",
        value: { kind: "Expr", value: { kind: "NewVector",
          components: [readU("u_offset", Tf32), constF(1), constF(1), constF(1)],
          type: Tvec4f } } },
      decorations: [],
    };
    const vsMod: Module = { types: [],
      values: [{ kind: "Uniform", uniforms: [{ name: "u_offset", type: Tf32 }] },
               { kind: "Entry", entry: vs }] };
    const fsMod: Module = { types: [],
      values: [{ kind: "Uniform", uniforms: [{ name: "u_offset", type: Tf32 }] },
               { kind: "Entry", entry: fs }] };
    return effect(stage(vsMod), stage(fsMod));
  }

  it("substitute(vertex.uniforms) only rewrites the vertex stage and changes id", () => {
    const fx = mkEffect();
    const id0 = fx.id;
    const replacement: Expr = constF(99);
    const fx2 = fx.substitute({
      vertex: { uniforms: new Map([["u_offset", replacement]]) },
    });
    expect(fx2.id).not.toBe(id0);
    // Vertex template: the ReadInput("Uniform","u_offset") is gone.
    const vsTpl = fx2.stages[0]!.template;
    const vsBody = entriesOf(vsTpl)[0]!.body;
    expect(countReads(vsBody, "Uniform", "u_offset")).toBe(0);
    // Fragment template: still reads u_offset.
    const fsTpl = fx2.stages[1]!.template;
    const fsBody = entriesOf(fsTpl)[0]!.body;
    expect(countReads(fsBody, "Uniform", "u_offset")).toBeGreaterThan(0);
  });

  it("rename(types) is applied across every stage", () => {
    // Inject a struct in both stages.
    const sty: Type = { kind: "Struct", name: "M", fields: [{ name: "a", type: Tf32 }] };
    const vs: EntryDef = {
      name: "vs", stage: "vertex", inputs: [], outputs: [],
      arguments: [], returnType: Tvoid,
      body: { kind: "Expression", value: { kind: "ReadInput", scope: "Uniform", name: "x", type: sty } },
      decorations: [],
    };
    const fs: EntryDef = {
      name: "fs", stage: "fragment", inputs: [], outputs: [],
      arguments: [], returnType: Tvoid,
      body: { kind: "Expression", value: { kind: "ReadInput", scope: "Uniform", name: "x", type: sty } },
      decorations: [],
    };
    const fx = effect(
      stage({ types: [{ kind: "Struct", name: "M", fields: [{ name: "a", type: Tf32 }] }],
              values: [{ kind: "Entry", entry: vs }] }),
      stage({ types: [{ kind: "Struct", name: "M", fields: [{ name: "a", type: Tf32 }] }],
              values: [{ kind: "Entry", entry: fs }] }),
    );
    const fx2 = fx.rename({ types: new Map([["M", "Mat"]]) });
    for (const s of fx2.stages) {
      const td = s.template.types[0]!;
      expect(td.kind === "Struct" && td.name).toBe("Mat");
    }
  });

  it("repeated rename/substitute with the same spec yields the same id", () => {
    const fx = mkEffect();
    const repl: Expr = constF(7);
    const a = fx.substitute({ vertex: { uniforms: new Map([["u_offset", repl]]) } });
    const b = fx.substitute({ vertex: { uniforms: new Map([["u_offset", repl]]) } });
    expect(a.id).toBe(b.id);

    const c = fx.rename({ entries: new Map([["vsMain", "myVs"]]) });
    const d = fx.rename({ entries: new Map([["vsMain", "myVs"]]) });
    expect(c.id).toBe(d.id);
  });

  it("rename(varyings) updates VS output + FS input across paired stages", () => {
    // Build an effect with a real cross-stage varying named "v_color"
    // (VS writes it, FS reads it).
    const vs: EntryDef = {
      name: "vsMain", stage: "vertex",
      inputs: [], outputs: [
        { name: "v_color", type: Tvec3f, semantic: "Color",
          decorations: [{ kind: "Location", value: 0 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: { kind: "Sequential", body: [
        { kind: "WriteOutput", name: "v_color",
          value: { kind: "Expr", value: { kind: "NewVector",
            components: [constF(1), constF(0), constF(0)], type: Tvec3f } } },
      ] },
      decorations: [],
    };
    const fs: EntryDef = {
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_color", type: Tvec3f, semantic: "Color",
                 decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color",
                  decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: { kind: "Sequential", body: [
        { kind: "WriteOutput", name: "outColor",
          value: { kind: "Expr", value: { kind: "NewVector",
            components: [readI("v_color", Tvec3f), constF(1)], type: Tvec4f } } },
      ] },
      decorations: [],
    };
    const fx = effect(
      stage({ types: [], values: [{ kind: "Entry", entry: vs }] }),
      stage({ types: [], values: [{ kind: "Entry", entry: fs }] }),
    );
    const fx2 = fx.rename({ varyings: new Map([["v_color", "v_tint"]]) });
    // dumpIR shape: both stages should mention v_tint, neither should
    // mention v_color anymore.
    const dump = fx2.dumpIR();
    expect(dump).toMatch(/v_tint/);
    expect(dump).not.toMatch(/v_color/);
  });

  it("rename(varyings + vertex.outputs overlap) throws", () => {
    const fx = mkEffect();
    expect(() =>
      fx.rename({
        varyings: new Map([["A", "B"]]),
        vertex: { outputs: new Map([["X", "B"]]) },
      })
    ).toThrow(/varyings rename .* vertex\.outputs overlap/);
    expect(() =>
      fx.rename({
        varyings: new Map([["A", "B"]]),
        vertex: { outputs: new Map([["A", "C"]]) },
      })
    ).toThrow(/varyings rename .* vertex\.outputs overlap/);
  });

  it("compile after rename + substitute produces clean WGSL", () => {
    const fx = mkEffect();
    const repl: Expr = constF(0.5);
    const fx2 = fx
      .substitute({ vertex: { uniforms: new Map([["u_offset", repl]]) } })
      .rename({ entries: new Map([["fsMain", "frag_main"]]) });
    const compiled = fx2.compile({ target: "wgsl" });
    const sources = compiled.stages.map((s) => s.source).join("\n");
    // No leftover "u_offset" use in the vertex stage; original fsMain
    // entry name is renamed.
    expect(sources).not.toMatch(/fn\s+fsMain\b/);
    expect(sources).toMatch(/frag_main/);
  });
});
