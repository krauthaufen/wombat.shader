// Dawn-validated tests for the `instanceUniforms` IR pass. We feed
// each test source through `compileShaderSource` to get an IR
// Module, run `instanceUniforms(module, attrNames)` on it, then
// emit WGSL via the runtime stage/effect path so the rewritten
// shader rides through the rest of the pipeline (linkCrossStage,
// pruneCrossStage, …) and finally lands in Dawn for validation.

import { describe, it, expect } from "vitest";
import { compileShaderSource, effect, stage } from "@aardworx/wombat.shader";
import {
  Tf32, Ti32, Tu32, Tvoid, Vec, Mat,
  type Module, type Type,
} from "@aardworx/wombat.shader/ir";
import { instanceUniforms, liftReturns } from "@aardworx/wombat.shader/passes";
import { validateWgsl, formatFailure } from "./_wgsl-validator.js";

const Tvec3f = Vec(Tf32, 3);
const Tvec4f = Vec(Tf32, 4);
const TM44f  = Mat(Tf32, 4, 4);
const TM33f  = Mat(Tf32, 3, 3);

/** Compile + rewrite + emit + validate. Returns the per-stage WGSL. */
async function rewriteAndValidate(
  source: string,
  entries: Parameters<typeof compileShaderSource>[1],
  attrNames: ReadonlySet<string>,
  uniformTypes: ReadonlyMap<string, Type>,
  opts?: Parameters<typeof compileShaderSource>[2],
): Promise<{ vs: string; fs?: string }> {
  // First a full compile to grab the un-rewritten Module — we want
  // the post-frontend, post-extras module that has `Uniform`
  // ValueDefs etc.
  const r = compileShaderSource(source, entries, { target: "wgsl", ...(opts as object ?? {}) });
  // The runtime doesn't expose the IR directly; reach into stage().
  // The cleanest route is `stage(module).compile(...)` — run our own
  // pipeline on a hand-built module that pre-applies the rewrite.
  void r; // emitted via the alt path below
  // Build the module via parseShader, apply the rewrite, run through
  // the standard runtime pipeline.
  const { parseShader } = await import("@aardworx/wombat.shader/frontend");
  const externalTypes = new Map<string, Type>();
  for (const v of opts?.extraValues ?? []) {
    if (v.kind === "Uniform") for (const u of v.uniforms) externalTypes.set(u.name, u.type);
  }
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = opts?.extraValues
    ? { ...parsed, values: [...opts.extraValues, ...parsed.values] }
    : parsed;
  // The pass runs at the post-`liftReturns` IR — entry returns have
  // already been split into explicit `WriteOutput` statements. The
  // production pipeline already calls `liftReturns` first; mirror it
  // here so the test exercises the same input shape.
  const lifted = liftReturns(merged);
  const rewritten = instanceUniforms(lifted, attrNames, uniformTypes);
  const fx = effect(stage(rewritten));
  const compiled = fx.compile({ target: "wgsl" });
  const out: { vs: string; fs?: string } = { vs: "" };
  for (const s of compiled.stages) {
    const v = await validateWgsl(s.source);
    if (!v.ok) {
      throw new Error(`${s.entryName} (${s.stage}) — ${formatFailure(v)}\n--- emitted ---\n${s.source}\n--- end ---`);
    }
    if (s.stage === "vertex") out.vs = s.source;
    if (s.stage === "fragment") out.fs = s.source;
  }
  return out;
}

const camUBO: Parameters<typeof compileShaderSource>[2]["extraValues"] = [
  { kind: "Uniform", uniforms: [
    { name: "ModelTrafo",        type: TM44f },
    { name: "ViewTrafo",         type: TM44f },
    { name: "ProjTrafo",         type: TM44f },
    { name: "ModelViewTrafo",    type: TM44f },
    { name: "ModelViewProjTrafo",type: TM44f },
    { name: "ModelTrafoInv",     type: TM44f },
    { name: "NormalMatrix",      type: TM44f },
    { name: "Color",             type: Tvec4f },
  ] },
];

const trafoTypes = new Map<string, Type>([
  ["ModelTrafo",         TM44f],
  ["ViewTrafo",          TM44f],
  ["ProjTrafo",          TM44f],
  ["ModelViewTrafo",     TM44f],
  ["ModelViewProjTrafo", TM44f],
  ["ModelTrafoInv",      TM44f],
  ["NormalMatrix",       TM44f],
  ["Color",              Tvec4f],
]);

// ─── Plain uniform → instance attribute ──────────────────────────────

describe("instanceUniforms: plain uniforms", () => {
  it("a single non-trafo uniform read in VS becomes a per-vertex input", async () => {
    const src = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: ProjTrafo.mul(input.Positions) };
      }
      function fsMain(): { outColor: V4f } { return { outColor: Color }; }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["Color"]), trafoTypes, { extraValues: camUBO });
    // The FS reads `Color` as a varying now, the VS exports it.
    expect(r.fs ?? "").toMatch(/_inst_Color/);
  });

  it("plain V3f uniform read only in VS — no FS varying needed", async () => {
    const src = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f; v_off: V3f } {
        const off = OffsetVec;
        return {
          gl_Position: new V4f(input.Positions.x + off.x, input.Positions.y + off.y, input.Positions.z + off.z, 1.0),
          v_off: off,
        };
      }
      function fsMain(input: { v_off: V3f }): { outColor: V4f } {
        return { outColor: new V4f(input.v_off.x, input.v_off.y, input.v_off.z, 1.0) };
      }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_off", type: Tvec3f, semantic: "v_off", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      { name: "fsMain", stage: "fragment",
        inputs: [{ name: "v_off", type: Tvec3f, semantic: "v_off", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["OffsetVec"]), new Map([["OffsetVec", Tvec3f]]), {
      extraValues: [{ kind: "Uniform", uniforms: [{ name: "OffsetVec", type: Tvec3f }] }],
    });
    // VS now reads `OffsetVec` as a per-instance attribute.
    expect(r.vs).toMatch(/OffsetVec/);
    // No FS varying — FS doesn't read the rewritten uniform.
    expect(r.fs ?? "").not.toMatch(/_inst_OffsetVec/);
  });
});

// ─── Trafo special cases ─────────────────────────────────────────────

describe("instanceUniforms: trafo special cases", () => {
  it("ModelTrafo in attrs → uniform.ModelTrafo * input.InstanceTrafo", async () => {
    const src = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        const wp = ModelTrafo.mul(input.Positions);
        const cp = ProjTrafo.mul(ViewTrafo.mul(wp));
        return { gl_Position: cp };
      }
      function fsMain(): { outColor: V4f } {
        return { outColor: new V4f(1.0, 0.0, 0.0, 1.0) };
      }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["ModelTrafo"]), trafoTypes, { extraValues: camUBO });
    // The VS now reads `_InstanceTrafo_col*` attributes, and the body's
    // uniform read of ModelTrafo was rewritten to multiply with the
    // reconstructed matrix.
    expect(r.vs).toMatch(/_InstanceTrafo_col0\b/);
  });

  it("ModelTrafoInv read in FS (after rewrite) → flat varying carrying `InstanceTrafoInv * uniform.ModelTrafoInv`", async () => {
    const src = `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
        return { gl_Position: input.Positions };
      }
      function fsMain(): { outColor: V4f } {
        const m = ModelTrafoInv;
        return { outColor: new V4f(m[0][3], m[1][3], m[2][3], 1.0) };
      }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      { name: "fsMain", stage: "fragment",
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["ModelTrafo"]), trafoTypes, { extraValues: camUBO });
    expect(r.vs).toMatch(/_inst_ModelTrafoInv/);
    expect(r.fs ?? "").toMatch(/_inst_ModelTrafoInv/);
  });

  it("NormalMatrix rebuilds from `uniform.NormalMatrix · transpose(m33(InstanceTrafoInv))` on the GPU", async () => {
    // Aardvark's per-instance normal-matrix formula. With wombat's
    // `simplifyTranspose` + `reverseMatrixOps` (Transpose preservation
    // + MulMatVec(Transpose) absorb + chain-lowering), this produces
    // clean WGSL using only the InstanceTrafoInv attribute — no
    // separate per-instance NormalMatrix buffer needed.
    const src = `
      function vsMain(input: { Positions: V4f; Normals: V3f }):
        { gl_Position: V4f; v_n: V3f } {
        const tn = NormalMatrix.mul(new V4f(input.Normals.x, input.Normals.y, input.Normals.z, 0.0));
        return {
          gl_Position: ProjTrafo.mul(input.Positions),
          v_n: new V3f(tn.x, tn.y, tn.z),
        };
      }
      function fsMain(input: { v_n: V3f }): { outColor: V4f } {
        return { outColor: new V4f(input.v_n.x, input.v_n.y, input.v_n.z, 1.0) };
      }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Normals",   type: Tvec3f, semantic: "Normals",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_n",         type: Tvec3f, semantic: "v_n",       decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      { name: "fsMain", stage: "fragment",
        inputs: [{ name: "v_n", type: Tvec3f, semantic: "v_n", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["ModelTrafo"]), trafoTypes, { extraValues: camUBO });
    // The InstanceTrafoInv columns appear (per-instance attribute is
    // declared), but no separate `_InstanceNormalMatrix` buffer is.
    expect(r.vs).toMatch(/_InstanceTrafoInv_col0\b/);
    expect(r.vs).not.toMatch(/_InstanceNormalMatrix/);
  });
});

// ─── Multi-attribute ─────────────────────────────────────────────────

describe("instanceUniforms: multi-attribute", () => {
  it("ModelTrafo + Color simultaneously", async () => {
    const src = `
      function vsMain(input: { Positions: V4f }):
        { gl_Position: V4f; v_color: V4f } {
        return {
          gl_Position: ProjTrafo.mul(ModelTrafo.mul(input.Positions)),
          v_color: Color,
        };
      }
      function fsMain(input: { v_color: V4f }): { outColor: V4f } {
        return { outColor: input.v_color };
      }
    `;
    const r = await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_color",     type: Tvec4f, semantic: "v_color",   decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      { name: "fsMain", stage: "fragment",
        inputs: [{ name: "v_color", type: Tvec4f, semantic: "v_color", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["ModelTrafo", "Color"]), trafoTypes, { extraValues: camUBO });
    expect(r.vs).toMatch(/_InstanceTrafo_col0\b/);
    expect(r.vs).toMatch(/\bColor\b/);
  });
});

// ─── Composition with v+f compose path ──────────────────────────────

describe("instanceUniforms: composes with linkCrossStage", () => {
  it("rewritten module survives the standard compile pipeline", async () => {
    // Two-effect compose: a vertex that writes Positions, plus a
    // fragment that reads it. After rewriting ModelTrafo, the VS
    // should still be valid + the FS still gets its varyings.
    const src = `
      function vsMain(input: { Positions: V4f; Normals: V3f }):
        { gl_Position: V4f; Normals: V3f } {
        const wp = ModelTrafo.mul(input.Positions);
        return { gl_Position: ProjTrafo.mul(ViewTrafo.mul(wp)), Normals: input.Normals };
      }
      function fsMain(input: { Normals: V3f }): { outColor: V4f } {
        return { outColor: new V4f(input.Normals.x, input.Normals.y, input.Normals.z, 1.0) };
      }
    `;
    await rewriteAndValidate(src, [
      { name: "vsMain", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Normals",   type: Tvec3f, semantic: "Normals",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "Normals",     type: Tvec3f, semantic: "Normals",   decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      { name: "fsMain", stage: "fragment",
        inputs: [{ name: "Normals", type: Tvec3f, semantic: "Normals", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], new Set(["ModelTrafo"]), trafoTypes, { extraValues: camUBO });
  });
});
