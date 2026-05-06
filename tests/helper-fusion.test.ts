// Tests for helper-extraction-based same-stage fusion + cross-helper
// linker. Covers:
//   - composition correctness (helper-call wrapper shape)
//   - imperative early-return semantics (Return exits only the helper)
//   - cross-helper liveness DCE (dead State fields + dead helper writes
//     + dead uniforms get pruned)
//   - tightest wrapper inputs (only what some live helper actually reads)
//   - State-struct shrink after DCE
//   - auto pass-through (helper B reads what helper A wrote, no carrier
//     declaration needed at the user level)

import { describe, expect, it } from "vitest";
import {
  composeStages,
  cse,
  dce,
  foldConstants,
  inlinePass,
  linkFragmentOutputs,
  linkHelpers,
  liftReturns,
  pruneCrossStage,
  reduceUniforms,
} from "@aardworx/wombat.shader/passes";
import { compileShaderSource } from "@aardworx/wombat.shader";
import {
  Tf32, Tvoid, Vec, Mat,
  type EntryDef, type Expr, type Module, type Stmt, type Type, type ValueDef,
} from "@aardworx/wombat.shader/ir";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);

// ─── construction helpers ──────────────────────────────────────────

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const readU = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Uniform", name, type });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });
const newV4 = (x: Expr, y: Expr, z: Expr, w: Expr): Expr => ({
  kind: "NewVector", components: [x, y, z, w], type: Tvec4f,
});
const newV3 = (x: Expr, y: Expr, z: Expr): Expr => ({
  kind: "NewVector", components: [x, y, z], type: Tvec3f,
});
const writeOut = (name: string, e: Expr): Stmt => ({
  kind: "WriteOutput", name, value: { kind: "Expr", value: e },
});
const seq = (...body: Stmt[]): Stmt => ({ kind: "Sequential", body });
const entriesOf = (m: Module): EntryDef[] =>
  m.values.flatMap((v) => v.kind === "Entry" ? [v.entry] : []);
const helpersOf = (m: Module): (ValueDef & { kind: "Function" })[] =>
  m.values.flatMap((v) =>
    v.kind === "Function" && (v.attributes ?? []).includes("merged_state_helper")
      ? [v]
      : [],
  );

// ─── composition shape ─────────────────────────────────────────────

describe("composeStages: helper-extracted same-stage fusion", () => {
  it("two fragments fuse into one wrapper Entry plus N helper Functions", () => {
    const fragA: EntryDef = {
      name: "a", stage: "fragment",
      inputs: [], outputs: [{ name: "tmp", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("tmp", newV3(constF(0), constF(0), constF(0))),
      decorations: [],
    };
    const fragB: EntryDef = {
      name: "b", stage: "fragment",
      inputs: [{ name: "tmp", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", newV4(
        { kind: "Field", target: readI("tmp", Tvec3f), name: "x", type: Tf32 },
        { kind: "Field", target: readI("tmp", Tvec3f), name: "y", type: Tf32 },
        { kind: "Field", target: readI("tmp", Tvec3f), name: "z", type: Tf32 },
        constF(1),
      )),
      decorations: [],
    };
    const m: Module = { types: [], values: [
      { kind: "Entry", entry: fragA },
      { kind: "Entry", entry: fragB },
    ]};

    const composed = composeStages(m);
    expect(entriesOf(composed)).toHaveLength(1);
    expect(helpersOf(composed)).toHaveLength(2);

    const wrapper = entriesOf(composed)[0]!;
    expect(wrapper.stage).toBe("fragment");
    expect(wrapper.outputs.map((o) => o.name)).toEqual(["outColor"]);
    expect(wrapper.inputs).toHaveLength(0);

    // The State struct is added to module.types and contains both
    // the surfaced output (`outColor`) and the carrier (`tmp`).
    const states = composed.types.filter((t) => t.kind === "Struct");
    expect(states).toHaveLength(1);
    const stateName = states[0]!.kind === "Struct" ? states[0]!.name : "";
    if (states[0]!.kind === "Struct") {
      const fieldNames = states[0]!.fields.map((f) => f.name).sort();
      expect(fieldNames).toEqual(["outColor", "tmp"]);
    }

    // Helpers are tagged with the merged_state_helper attribute and
    // share the same single State parameter shape.
    for (const h of helpersOf(composed)) {
      expect(h.attributes).toContain("merged_state_helper");
      expect(h.signature.parameters).toHaveLength(1);
      expect(h.signature.parameters[0]!.modifier).toBe("in");
      expect(h.signature.parameters[0]!.type.kind).toBe("Struct");
      if (h.signature.parameters[0]!.type.kind === "Struct") {
        expect(h.signature.parameters[0]!.type.name).toBe(stateName);
      }
      expect(h.signature.returnType.kind).toBe("Struct");
    }
  });

  it("three-helper chain — A writes pipe X, B reads X writes Y, C reads Y writes outColor", () => {
    const A: EntryDef = {
      name: "A", stage: "fragment",
      inputs: [], outputs: [{ name: "X", type: Tvec3f, semantic: "X", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("X", newV3(constF(1), constF(2), constF(3))),
      decorations: [],
    };
    const B: EntryDef = {
      name: "B", stage: "fragment",
      inputs: [{ name: "X", type: Tvec3f, semantic: "X", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "Y", type: Tvec3f, semantic: "Y", decorations: [{ kind: "Location", value: 1 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("Y", readI("X", Tvec3f)),
      decorations: [],
    };
    const C: EntryDef = {
      name: "C", stage: "fragment",
      inputs: [{ name: "Y", type: Tvec3f, semantic: "Y", decorations: [{ kind: "Location", value: 1 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", newV4(
        { kind: "Field", target: readI("Y", Tvec3f), name: "x", type: Tf32 },
        { kind: "Field", target: readI("Y", Tvec3f), name: "y", type: Tf32 },
        { kind: "Field", target: readI("Y", Tvec3f), name: "z", type: Tf32 },
        constF(1),
      )),
      decorations: [],
    };
    const m: Module = { types: [], values: [
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B }, { kind: "Entry", entry: C },
    ]};

    const composed = composeStages(m);
    expect(helpersOf(composed)).toHaveLength(3);
    const wrapper = entriesOf(composed)[0]!;
    // Both X and Y are pipes (consumed downstream); only outColor surfaces.
    expect(wrapper.outputs.map((o) => o.name)).toEqual(["outColor"]);
    expect(wrapper.inputs).toHaveLength(0);
    // State holds every port (X, Y, outColor).
    const state = composed.types.find((t) => t.kind === "Struct");
    expect(state?.kind).toBe("Struct");
    if (state?.kind === "Struct") {
      const names = state.fields.map((f) => f.name).sort();
      expect(names).toEqual(["X", "Y", "outColor"]);
    }
  });

  it("B-wins on output collision: A writes outColor + extra; B reads & re-writes outColor", () => {
    const A: EntryDef = {
      name: "A", stage: "fragment",
      inputs: [], outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 7 }] },
        { name: "extra", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 8 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: seq(
        writeOut("outColor", newV4(constF(1), constF(0), constF(0), constF(1))),
        writeOut("extra", newV3(constF(0), constF(0), constF(0))),
      ),
      decorations: [],
    };
    const B: EntryDef = {
      name: "B", stage: "fragment",
      inputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", readI("outColor", Tvec4f)),
      decorations: [],
    };
    const composed = composeStages({ types: [], values: [
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B },
    ]});
    const wrapper = entriesOf(composed)[0]!;
    const names = wrapper.outputs.map((o) => o.name).sort();
    // outColor surfaces from B's last write; A's `extra` (no
    // downstream reader) also surfaces.
    expect(names).toEqual(["extra", "outColor"]);
    // B's Location for outColor wins (= 0, not A's 7).
    const oc = wrapper.outputs.find((o) => o.name === "outColor")!;
    const loc = oc.decorations.find((d) => d.kind === "Location");
    expect(loc?.kind === "Location" && loc.value).toBe(0);
  });
});

// ─── imperative early-return ────────────────────────────────────────

describe("composeStages + helper extraction: imperative early-return", () => {
  it("A's `if (cond) return X;` exits A's helper only — B still runs", () => {
    // A writes outColor based on a uniform threshold; if threshold < 0
    // it bails early.
    const A: EntryDef = {
      name: "A", stage: "fragment",
      inputs: [],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: seq(
        // if (u_thresh < 0) return { outColor: vec4(0,0,0,1) };
        {
          kind: "If",
          cond: { kind: "Lt", lhs: readU("u_thresh", Tf32), rhs: constF(0), type: { kind: "Bool" } },
          then: seq(
            writeOut("outColor", newV4(constF(0), constF(0), constF(0), constF(1))),
            { kind: "Return" },
          ),
        },
        writeOut("outColor", newV4(constF(0.5), constF(0.5), constF(0.5), constF(1))),
      ),
      decorations: [],
    };
    // B unconditionally adds bias to outColor — its writes should
    // ALWAYS run regardless of A's early return.
    const B: EntryDef = {
      name: "B", stage: "fragment",
      inputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", readI("outColor", Tvec4f)),
      decorations: [],
    };

    const composed = composeStages({ types: [], values: [
      { kind: "Uniform", uniforms: [{ name: "u_thresh", type: Tf32 }] },
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B },
    ]});

    // Helper A's body still has the imperative Return (the emitter
    // turns it into `return out;` inside A's helper, which exits the
    // helper only — wrapper continues to call helper B).
    const helperA = helpersOf(composed).find((h) => h.signature.name.includes("_A_"))!;
    expect(helperA).toBeDefined();
    const helperAhasReturn = (function findReturn(s: Stmt): boolean {
      if (s.kind === "Return") return true;
      if (s.kind === "Sequential") return s.body.some(findReturn);
      if (s.kind === "If") return findReturn(s.then) || (s.else !== undefined && findReturn(s.else));
      return false;
    })(helperA.body);
    expect(helperAhasReturn).toBe(true);
  });
});

// ─── cross-helper liveness DCE ──────────────────────────────────────

describe("linkHelpers: backward liveness DCE across helper calls", () => {
  it("drops a State field that no surviving wrapper output reads", () => {
    // A writes both `outColor` (live) and `dead` (will get demoted by
    // linkFragmentOutputs since the FB layout doesn't include it).
    const A: EntryDef = {
      name: "A", stage: "fragment",
      inputs: [],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
        { name: "dead", type: Tvec3f, semantic: "Dead", decorations: [{ kind: "Location", value: 1 }] },
      ],
      arguments: [], returnType: Tvoid,
      body: seq(
        writeOut("outColor", newV4(readU("u_live", Tf32), constF(0), constF(0), constF(1))),
        writeOut("dead", newV3(readU("u_dead", Tf32), readU("u_dead", Tf32), readU("u_dead", Tf32))),
      ),
      decorations: [],
    };
    // A no-op B that just preserves outColor (so we still get a fused
    // wrapper rather than a single-entry shortcut).
    const B: EntryDef = {
      name: "B", stage: "fragment",
      inputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", readI("outColor", Tvec4f)),
      decorations: [],
    };
    let m: Module = { types: [], values: [
      { kind: "Uniform", uniforms: [{ name: "u_live", type: Tf32 }, { name: "u_dead", type: Tf32 }] },
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B },
    ]};

    m = composeStages(m);
    m = linkFragmentOutputs(m, { locations: new Map([["outColor", 0]]) });
    m = linkHelpers(m);
    m = reduceUniforms(m);

    // u_dead has no live consumer left.
    const u = m.values.find((v) => v.kind === "Uniform");
    expect(u?.kind).toBe("Uniform");
    if (u?.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["u_live"]);
    }

    // The State struct shrunk — `dead` field is gone.
    const state = m.types.find((t) => t.kind === "Struct");
    expect(state?.kind).toBe("Struct");
    if (state?.kind === "Struct") {
      expect(state.fields.map((f) => f.name)).not.toContain("dead");
    }

    // Helper A no longer writes `dead`.
    const helperA = helpersOf(m).find((h) => h.signature.name.includes("_A_"))!;
    expect(JSON.stringify(helperA.body)).not.toContain("u_dead");
    expect(JSON.stringify(helperA.body)).not.toContain("\"dead\"");
  });

  it("propagates liveness across multiple helper hops (A writes X, B writes Y from X, C writes outColor from Y)", () => {
    // If outColor is live, Y must be live (C reads it), X must be live
    // (B reads it). All three helpers' writes survive.
    const make = (name: string, inputs: { name: string; loc: number; }[], output: { name: string; loc: number; type: Type }, bodyF: () => Stmt): EntryDef => ({
      name, stage: "fragment",
      inputs: inputs.map(({ name, loc }) => ({ name, type: Tvec3f, semantic: name, decorations: [{ kind: "Location" as const, value: loc }] })),
      outputs: [{ name: output.name, type: output.type, semantic: output.name, decorations: [{ kind: "Location" as const, value: output.loc }] }],
      arguments: [], returnType: Tvoid,
      body: bodyF(),
      decorations: [],
    });
    const A = make("A", [], { name: "X", loc: 0, type: Tvec3f },
      () => writeOut("X", newV3(readU("uA", Tf32), constF(0), constF(0))));
    const B = make("B", [{ name: "X", loc: 0 }], { name: "Y", loc: 0, type: Tvec3f },
      () => writeOut("Y", readI("X", Tvec3f)));
    const C: EntryDef = {
      name: "C", stage: "fragment",
      inputs: [{ name: "Y", type: Tvec3f, semantic: "Y", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", newV4(
        { kind: "Field", target: readI("Y", Tvec3f), name: "x", type: Tf32 },
        constF(0), constF(0), constF(1),
      )),
      decorations: [],
    };

    let m: Module = { types: [], values: [
      { kind: "Uniform", uniforms: [{ name: "uA", type: Tf32 }] },
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B }, { kind: "Entry", entry: C },
    ]};
    m = composeStages(m);
    m = linkFragmentOutputs(m, { locations: new Map([["outColor", 0]]) });
    m = linkHelpers(m);
    m = reduceUniforms(m);

    // Every helper's write survives.
    for (const h of helpersOf(m)) {
      const json = JSON.stringify(h.body);
      // Each helper has a non-Nop body.
      expect(json).not.toBe("{\"kind\":\"Nop\"}");
    }
    // uA stays live — A still reads it.
    const u = m.values.find((v) => v.kind === "Uniform");
    if (u?.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["uA"]);
    }
  });

  it("drops a whole helper whose only writes are to fields nothing reads", () => {
    // A writes only `dead` (consumed by no surfaced output).
    // B writes `outColor` independently of A.
    const A: EntryDef = {
      name: "A", stage: "fragment", inputs: [],
      outputs: [{ name: "dead", type: Tvec3f, semantic: "Dead", decorations: [{ kind: "Location", value: 1 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("dead", newV3(readU("u_dead", Tf32), constF(0), constF(0))),
      decorations: [],
    };
    const B: EntryDef = {
      name: "B", stage: "fragment", inputs: [],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", newV4(readU("u_live", Tf32), constF(0), constF(0), constF(1))),
      decorations: [],
    };
    let m: Module = { types: [], values: [
      { kind: "Uniform", uniforms: [{ name: "u_dead", type: Tf32 }, { name: "u_live", type: Tf32 }] },
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B },
    ]};
    m = composeStages(m);
    m = linkFragmentOutputs(m, { locations: new Map([["outColor", 0]]) });
    m = linkHelpers(m);
    m = reduceUniforms(m);

    const u = m.values.find((v) => v.kind === "Uniform");
    if (u?.kind === "Uniform") {
      expect(u.uniforms.map((d) => d.name)).toEqual(["u_live"]);
    }

    // Helper A's body collapsed — only Nop or empty Sequential remain.
    const helperA = helpersOf(m).find((h) => h.signature.name.includes("_A_"))!;
    const aJson = JSON.stringify(helperA.body);
    expect(aJson).not.toContain("u_dead");
  });
});

// ─── tightest inputs ────────────────────────────────────────────────

describe("composeStages: wrapper inputs reflect actual reads, not declared inputs", () => {
  it("a declared input that the body never reads doesn't surface on the wrapper", () => {
    const A: EntryDef = {
      name: "A", stage: "fragment",
      // Declares `unused` as input but the body doesn't read it.
      inputs: [
        { name: "unused", type: Tvec3f, semantic: "Unused", decorations: [{ kind: "Location", value: 0 }] },
        { name: "used", type: Tvec3f, semantic: "Used", decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("outColor", newV4(
        { kind: "Field", target: readI("used", Tvec3f), name: "x", type: Tf32 },
        constF(0), constF(0), constF(1),
      )),
      decorations: [],
    };
    const B: EntryDef = {
      name: "B", stage: "fragment", inputs: [],
      outputs: [{ name: "passthrough", type: Tvec3f, semantic: "X", decorations: [{ kind: "Location", value: 1 }] }],
      arguments: [], returnType: Tvoid,
      body: writeOut("passthrough", newV3(constF(0), constF(0), constF(0))),
      decorations: [],
    };
    const composed = composeStages({ types: [], values: [
      { kind: "Entry", entry: A }, { kind: "Entry", entry: B },
    ]});
    const wrapper = entriesOf(composed)[0]!;
    const inputNames = wrapper.inputs.map((p) => p.name).sort();
    expect(inputNames).toEqual(["used"]);
  });
});

// ─── intrinsics + discard inside extracted helpers ─────────────────

describe("extracted helpers: intrinsics + discard work end-to-end", () => {
  // Two fragments composed: A normalises a normal, computes a Lambert
  // term using `dot`/`abs`, `discard`s near-grazing fragments, and
  // tints the surviving Colors. B applies a simple `pow`/`max` tone
  // map. Exercises every commonly-needed intrinsic AND a fragment-
  // stage `discard` inside an extracted helper.
  const sourceLighting = `
    function fragLambert(input: { Normals: V3f; Colors: V4f }): { Colors: V4f } {
      const n = normalize(input.Normals);
      const lit = abs(n.dot(new V3f(0.0, 0.0, 1.0)));
      if (lit < 0.05) { discard; }
      return { Colors: new V4f(input.Colors.x * lit, input.Colors.y * lit, input.Colors.z * lit, input.Colors.w) };
    }
    function fragTonemap(input: { Colors: V4f }): { outColor: V4f } {
      const c = input.Colors;
      const tone = pow(max(c.x, max(c.y, c.z)), 0.45);
      return { outColor: new V4f(c.x * tone, c.y * tone, c.z * tone, c.w) };
    }
  `;
  const entries = [
    {
      name: "fragLambert", stage: "fragment" as const,
      inputs: [
        { name: "Normals", type: Tvec3f, semantic: "Normals", decorations: [{ kind: "Location" as const, value: 0 }] },
        { name: "Colors", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 1 }] },
      ],
      outputs: [
        { name: "Colors", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 1 }] },
      ],
    },
    {
      name: "fragTonemap", stage: "fragment" as const,
      inputs: [{ name: "Colors", type: Tvec4f, semantic: "Colors", decorations: [{ kind: "Location" as const, value: 1 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location" as const, value: 0 }] }],
    },
  ];

  it("WGSL emits every intrinsic + `discard` inside the helper Function", () => {
    const r = compileShaderSource(sourceLighting, entries, { target: "wgsl" });
    const src = r.stages[0]!.source;
    // Intrinsics — emitted by name in WGSL.
    expect(src).toMatch(/normalize\(/);
    expect(src).toMatch(/dot\(/);
    expect(src).toMatch(/abs\(/);
    expect(src).toMatch(/pow\(/);
    expect(src).toMatch(/max\(/);
    // `discard;` survives extraction, sits inside the lambert helper.
    expect(src).toContain("discard;");
    // The helper Function's `discard` precedes `return out;` (the
    // lambert helper's tail), proving the discard is inside the
    // extracted body, not the entry's main.
    const helperBlockStart = src.indexOf("fn _fragLambert_fragTonemap_fragLambert_0");
    expect(helperBlockStart).toBeGreaterThanOrEqual(0);
    const helperBlockEnd = src.indexOf("\n}\n", helperBlockStart);
    const helperBlock = src.slice(helperBlockStart, helperBlockEnd);
    expect(helperBlock).toContain("discard;");
    expect(helperBlock).toContain("normalize(");
  });

  it("GLSL emits intrinsics + discard inside helpers, and avoids the `out` reserved word", () => {
    const r = compileShaderSource(sourceLighting, entries, { target: "glsl" });
    const src = r.stages[0]!.source;
    expect(src).toMatch(/normalize\(/);
    expect(src).toMatch(/dot\(/);
    expect(src).toMatch(/abs\(/);
    expect(src).toMatch(/pow\(/);
    expect(src).toMatch(/max\(/);
    expect(src).toContain("discard;");
    // GLSL reserves `out` as a parameter qualifier — extracted
    // helpers must use a non-reserved local name (`_out_`).
    expect(src).not.toMatch(/^\s*FragA_fragBState out =/m);
    expect(src).toMatch(/_out_\s*=\s*s_in/);
    // The merged-state struct holds every port across all helpers.
    expect(src).toContain("struct ");
    expect(src).toContain("Normals;");
    expect(src).toContain("outColor;");
  });

  it("`discard` survives the cross-helper liveness DCE (a downstream live read keeps the discarding helper alive)", () => {
    // Same source as above. tonemap reads Colors, which lambert
    // writes — so lambert's body is live, including its `discard`.
    const r = compileShaderSource(sourceLighting, entries, { target: "wgsl" });
    const src = r.stages[0]!.source;
    expect(src).toContain("discard;");
  });
});

// ─── full-pipeline integration ─────────────────────────────────────

describe("compileShaderSource end-to-end with imperative early-return", () => {
  it("WGSL: a single fragment with `if (cond) return X;` keeps the early-exit semantics", () => {
    const source = `
      function fsMain(input: { v_color: V4f }): { outColor: V4f } {
        if (input.v_color.w < 0.5) {
          return { outColor: new V4f(0.0, 0.0, 0.0, 1.0) };
        }
        return { outColor: input.v_color };
      }
    `;
    const r = compileShaderSource(source, [{
      name: "fsMain", stage: "fragment",
      outputs: [{
        name: "outColor", type: Tvec4f, semantic: "Color",
        decorations: [{ kind: "Location", value: 0 }],
      }],
    }], { target: "wgsl" });
    const fs = r.stages[0]!.source;
    // The early-exit is `return out;` inside the `if`. The natural
    // tail return is also `return out;`. Two `return out;` total
    // (one per fall-out path), no duplicate.
    const matches = fs.match(/return out;/g) ?? [];
    expect(matches).toHaveLength(2);
  });
});
