// End-to-end WGSL validation: every emitted shader runs through
// real Dawn (Chromium's WebGPU implementation) via the `webgpu`
// node binding, so type errors, struct redeclarations, builtin
// slot violations etc. fail at unit-test time instead of at
// pipeline-creation in the browser.
//
// Inspired by FShade's GLSL test corpus
// (`fshade/src/Tests/Tests.GLSL/`) — every category a shader can
// trip on has at least one example. Adding a test here is the
// fastest way to lock in a regression after a real-world bug.

import { describe, expect, it } from "vitest";
import {
  compileShaderSource,
  effect,
} from "@aardworx/wombat.shader";
import {
  Tf32, Ti32, Tu32, Tvoid, Vec, Mat,
  type EntryDef, type Module, type Stmt, type Type, type Var, type Expr,
} from "@aardworx/wombat.shader/ir";
import { expectValidWgsl, expectInvalidWgsl, validateWgsl, formatFailure } from "./_wgsl-validator.js";

const Tvec2f = Vec(Tf32, 2);
const Tvec3f = Vec(Tf32, 3);
const Tvec4f = Vec(Tf32, 4);
const TM44f = Mat(Tf32, 4, 4);

const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const constI = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const readU = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Uniform", name, type });
const readI = (name: string, type: Type): Expr => ({ kind: "ReadInput", scope: "Input", name, type });

async function emitAndValidate(source: string, entries: Parameters<typeof compileShaderSource>[1], opts?: Parameters<typeof compileShaderSource>[2]): Promise<void> {
  const r = compileShaderSource(source, entries, { target: "wgsl", ...(opts as object ?? {}) });
  for (const stage of r.stages) {
    const v = await validateWgsl(stage.source);
    if (!v.ok) {
      throw new Error(`${stage.entryName} (${stage.stage}) — ${formatFailure(v)}`);
    }
  }
}

// ─── 1. Bit ops ──────────────────────────────────────────────────────

describe("bit ops", () => {
  it("i32 << i32 (literal int) compiles — emit auto-casts shift amount", async () => {
    const src = `
      function fsMain(): { outColor: V4f } {
        const x: i32 = 5 as i32;
        const y = x << 12;
        return { outColor: new V4f((y as f32) / 100000.0, 0.0, 0.0, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });

  it("u32 bitwise AND/OR/XOR + right shift — all valid", async () => {
    const src = `
      function fsMain(): { outColor: V4f } {
        const a: u32 = 7 as u32;
        const b: u32 = 12 as u32;
        const r = ((a & b) | (a ^ b)) >> 2 as u32;
        return { outColor: new V4f((r as f32) / 100.0, 0.0, 0.0, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });
});

// ─── 2. Helper functions ─────────────────────────────────────────────

describe("helper functions (top-level + cross-effect)", () => {
  it("entry calls a top-level helper that calls another helper", async () => {
    const src = `
      function square(x: f32): f32 { return x * x; }
      function cubeOfSum(a: f32, b: f32): f32 { return square(a + b) * (a + b); }
      function fsMain(input: { uv: V2f }): { outColor: V4f } {
        const c = cubeOfSum(input.uv.x, input.uv.y);
        return { outColor: new V4f(c, c, c, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "uv", type: Tvec2f, semantic: "uv", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }], { helpers: ["square", "cubeOfSum"] });
  });

  it("vector helpers: normalize / dot / cross / length", async () => {
    const src = `
      function lambert(n: V3f, l: V3f): f32 {
        const nn = normalize(n);
        const ll = normalize(l);
        return max(nn.dot(ll), 0.0);
      }
      function fsMain(input: { Normals: V3f }): { outColor: V4f } {
        const lit = lambert(input.Normals, new V3f(0.0, 0.0, 1.0));
        return { outColor: new V4f(lit, lit, lit, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "Normals", type: Tvec3f, semantic: "Normals", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }], { helpers: ["lambert"] });
  });
});

// ─── 3. Arithmetic — vector/matrix/scalar combos ─────────────────────

describe("arithmetic — vector / matrix / scalar mixes", () => {
  it("matrix * vector, vector * scalar, vector + vector all coexist", async () => {
    const src = `
      function vsMain(input: { Positions: V3f }): { gl_Position: V4f } {
        const wp = M.mul(new V4f(input.Positions.x, input.Positions.y, input.Positions.z, 1.0));
        const scaled = new V4f(wp.x * 2.0, wp.y * 2.0, wp.z * 2.0, wp.w);
        const offset = new V4f(scaled.x + 1.0, scaled.y, scaled.z, scaled.w);
        return { gl_Position: offset };
      }
    `;
    await emitAndValidate(src, [{
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec3f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }], { extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
  });

  it("matrix * matrix * vector chain", async () => {
    const src = `
      function vsMain(input: { Positions: V3f }): { gl_Position: V4f } {
        const mvp = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo));
        return { gl_Position: mvp.mul(new V4f(input.Positions.x, input.Positions.y, input.Positions.z, 1.0)) };
      }
    `;
    await emitAndValidate(src, [{
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec3f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }], { extraValues: [{ kind: "Uniform", uniforms: [
      { name: "ModelTrafo", type: TM44f },
      { name: "ViewTrafo",  type: TM44f },
      { name: "ProjTrafo",  type: TM44f },
    ] }] });
  });
});

// ─── 4. Swizzles ─────────────────────────────────────────────────────

describe("swizzles", () => {
  it(".xyz / .xy / .zyx / .xxxx / mixed swizzle reads", async () => {
    const src = `
      function fsMain(input: { p: V4f }): { outColor: V4f } {
        const a = input.p.xyz;
        const b = input.p.zyx;
        const c = new V2f(input.p.x, input.p.w);
        const d = a.x * a.y + b.z * c.y;
        return { outColor: new V4f(d, a.x, b.x, c.y) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "p", type: Tvec4f, semantic: "p", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });
});

// ─── 5. Builtins ─────────────────────────────────────────────────────

describe("builtins", () => {
  it("fragment fragCoord (via FragmentBuiltinIn record) → @builtin(position) input", async () => {
    const src = `
      function fsMain(_in: {}, b: FragmentBuiltinIn): { outColor: V4f } {
        return { outColor: new V4f(b.fragCoord.x / 800.0, b.fragCoord.y / 600.0, 0.0, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });

  it("vertex vertexIndex / instanceIndex via VertexBuiltinIn", async () => {
    const src = `
      function vsMain(_in: {}, b: VertexBuiltinIn): { gl_Position: V4f } {
        const x = (b.vertexIndex as f32) * 0.1;
        const y = (b.instanceIndex as f32) * 0.05;
        return { gl_Position: new V4f(x, y, 0.0, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "vsMain", stage: "vertex",
      inputs: [],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
    }]);
  });
});

// ─── 6. Branching + early return + discard ───────────────────────────

describe("branching, early return, discard", () => {
  it("if/else with early-exit return", async () => {
    const src = `
      function fsMain(input: { v_color: V4f }): { outColor: V4f } {
        if (input.v_color.w < 0.5) {
          return { outColor: new V4f(0.0, 0.0, 0.0, 1.0) };
        }
        return { outColor: input.v_color };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_color", type: Tvec4f, semantic: "v_color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });

  it("nested if + discard", async () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): { outColor: V4f } {
        if (input.v_uv.x < 0.0) {
          if (input.v_uv.y < 0.0) {
            discard;
          }
          return { outColor: new V4f(0.5, 0.5, 0.5, 1.0) };
        }
        return { outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_uv", type: Tvec2f, semantic: "v_uv", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });
});

// ─── 7. Loops ────────────────────────────────────────────────────────

describe("loops", () => {
  it("for-loop with accumulator (Mandelbrot-ish)", async () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): { outColor: V4f } {
        var x: f32 = 0.0;
        var y: f32 = 0.0;
        var i: i32 = 0;
        for (let it = 0 as i32; it < 32; it = (it as i32) + 1) {
          const xn = x * x - y * y + input.v_uv.x;
          const yn = 2.0 * x * y + input.v_uv.y;
          x = xn;
          y = yn;
          if ((x * x + y * y) > 4.0) {
            break;
          }
          i = i + 1;
        }
        const t = (i as f32) / 32.0;
        return { outColor: new V4f(t, t * t, 1.0 - t, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_uv", type: Tvec2f, semantic: "v_uv", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });

  it("while-loop + continue", async () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): { outColor: V4f } {
        var sum: f32 = 0.0;
        var i: i32 = 0;
        while (i < 16) {
          i = i + 1;
          if ((i % 2) == 0) { continue; }
          sum = sum + (i as f32) * input.v_uv.x;
        }
        return { outColor: new V4f(sum / 100.0, 0.0, 0.0, 1.0) };
      }
    `;
    await emitAndValidate(src, [{
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_uv", type: Tvec2f, semantic: "v_uv", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    }]);
  });
});

// ─── 8. Composition (vertex + fragment, multi-helper fusion) ─────────

describe("multi-stage / multi-helper composition", () => {
  it("v+f compose with auto-pass-through Position", async () => {
    const src = `
      function vsMain(input: { Positions: V3f }): { gl_Position: V4f } {
        return { gl_Position: M.mul(new V4f(input.Positions.x, input.Positions.y, input.Positions.z, 1.0)) };
      }
      function fsMain(input: { Positions: V4f }): { outColor: V4f } {
        return { outColor: input.Positions };
      }
    `;
    await emitAndValidate(src, [
      {
        name: "vsMain", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec3f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] }],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], { extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
  });

  it("two-helper VS fusion (extractFusedEntry path) — both helpers' bodies validate", async () => {
    const src = `
      function vsA(input: { Positions: V4f }): { gl_Position: V4f; carrier: V4f } {
        const wp = M.mul(input.Positions);
        return { gl_Position: wp, carrier: wp };
      }
      function vsB(input: { carrier: V4f; tint: V4f }): { gl_Position: V4f; outVarying: V4f } {
        return { gl_Position: input.carrier, outVarying: input.tint };
      }
      function fsMain(input: { outVarying: V4f }): { outColor: V4f } {
        return { outColor: input.outVarying };
      }
    `;
    await emitAndValidate(src, [
      {
        name: "vsA", stage: "vertex",
        inputs: [{ name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "carrier",     type: Tvec4f, semantic: "carrier",   decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "vsB", stage: "vertex",
        inputs: [
          { name: "carrier", type: Tvec4f, semantic: "carrier", decorations: [{ kind: "Location", value: 0 }] },
          { name: "tint",    type: Tvec4f, semantic: "tint",    decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Positions",  decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "outVarying",  type: Tvec4f, semantic: "outVarying", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [{ name: "outVarying", type: Tvec4f, semantic: "outVarying", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], { extraValues: [{ kind: "Uniform", uniforms: [{ name: "M", type: TM44f }] }] });
  });
});

// ─── 9. Negative — known-bad shaders that SHOULD be rejected ────────

describe("negative cases — Dawn rejects what our emit shouldn't have produced", () => {
  it("two structs with the same name (would be `_UB_uniform` redeclaration)", async () => {
    await expectInvalidWgsl(
      `struct A { x: f32 };
       struct A { y: f32 };
       @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); }`,
      /redeclar|already declared/i,
    );
  });

  it("i32 << i32 directly (without the u32 cast we now emit)", async () => {
    await expectInvalidWgsl(
      `@vertex fn vs() -> @builtin(position) vec4f {
        let x: i32 = 5;
        return vec4f(f32(x << 12i));
      }`,
      /no matching overload|operator <<|u32/i,
    );
  });

  it("unresolved identifier", async () => {
    await expectInvalidWgsl(
      `@fragment fn fs() -> @location(0) vec4f { return undefined_thing; }`,
      /unresolved|identifier|undefined/i,
    );
  });
});

// ─── 10. End-to-end pick-chain compose (the actual failure case) ─────

describe("regression: pick-chain compose validates", () => {
  it("3-stage compose (VS-only viewSpaceNormal + VS+FS user effect + FS-only pickFinal)", async () => {
    // Three-effect compose with builtin reads, helper calls,
    // bit ops, casts — exactly the shape that broke the showcase.
    const src = `
      function n24Encode(v: V3f): f32 {
        const inv = 1.0 / (abs(v.x) + abs(v.y) + abs(v.z));
        const px0 = v.x * inv;
        const py0 = v.y * inv;
        const sx = px0 >= 0.0 ? 1.0 : -1.0;
        const sy = py0 >= 0.0 ? 1.0 : -1.0;
        const fx = v.z <= 0.0 ? (1.0 - abs(py0)) * sx : px0;
        const fy = v.z <= 0.0 ? (1.0 - abs(px0)) * sy : py0;
        const cx = clamp(fx, -1.0, 1.0);
        const cy = clamp(fy, -1.0, 1.0);
        const x0 = floor((cx * 0.5 + 0.5) * 4095.0) as i32;
        const y0 = floor((cy * 0.5 + 0.5) * 4095.0) as i32;
        return ((x0 << 12) | y0) as f32;
      }
      function vsMain(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; ViewSpaceNormal: V3f } {
        const t = ModelViewTrafoInv.transpose();
        const v4 = new V4f(input.Normals.x, input.Normals.y, input.Normals.z, 0.0);
        const vn = t.mul(v4);
        const n = new V3f(vn.x, vn.y, vn.z);
        return { gl_Position: input.Positions, ViewSpaceNormal: n.normalize() };
      }
      function fsMain(input: {
        outColor: V4f;
        ViewSpaceNormal: V3f;
      }, b: FragmentBuiltinIn): { outColor: V4f; pickId: V4f; Depth: f32 } {
        const n24 = n24Encode(input.ViewSpaceNormal.normalize());
        const id = new V4f(PickId as f32, n24, b.fragCoord.z, 0.0);
        return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
      }
    `;
    await emitAndValidate(src, [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Normals",   type: Tvec3f, semantic: "Normals",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position",     type: Tvec4f, semantic: "Positions",       decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [
          { name: "outColor",        type: Tvec4f, semantic: "Color",           decorations: [{ kind: "Location", value: 1 }] },
          { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
        ],
        outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Color",     decorations: [{ kind: "Location", value: 0 }] },
          { name: "pickId",   type: Tvec4f, semantic: "PickId",    decorations: [{ kind: "Location", value: 1 }] },
          { name: "Depth",    type: Tf32,   semantic: "Depth",     decorations: [{ kind: "Builtin", value: "frag_depth" }] },
        ],
      },
    ], {
      extraValues: [
        { kind: "Uniform", uniforms: [{ name: "ModelViewTrafoInv", type: TM44f }] },
        { kind: "Uniform", uniforms: [{ name: "PickId", type: Tu32 }] },
      ],
      helpers: ["n24Encode"],
    });
  });

  it("two VS effects each with locations starting at 0 — composeStages must renumber to avoid duplicate @location(0)", async () => {
    // Both VS write a non-position varying at @location(0). After
    // composeStages → extractFusedEntry the wrapper merges outputs
    // wholesale; without per-side location-renumbering the WGSL has
    // two outputs at @location(0) — exactly the showcase regression.
    const src = `
      function vsA(input: { Positions: V4f; Normals: V3f }):
        { gl_Position: V4f; ViewSpaceNormal: V3f } {
        return { gl_Position: input.Positions, ViewSpaceNormal: input.Normals };
      }
      function vsB(input: { Positions: V4f; Colors: V4f }):
        { gl_Position: V4f; WorldPositions: V4f; Colors: V4f } {
        return { gl_Position: input.Positions, WorldPositions: input.Positions, Colors: input.Colors };
      }
      function fsMain(input: { ViewSpaceNormal: V3f; WorldPositions: V4f; Colors: V4f }):
        { outColor: V4f } {
        return { outColor: new V4f(input.ViewSpaceNormal, 1.0) };
      }
    `;
    await emitAndValidate(src, [
      {
        name: "vsA", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Normals",   type: Tvec3f, semantic: "Normals",   decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position",     type: Tvec4f, semantic: "Positions",       decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "vsB", stage: "vertex",
        inputs: [
          { name: "Positions", type: Tvec4f, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Colors",    type: Tvec4f, semantic: "Colors",    decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position",    type: Tvec4f, semantic: "Positions",     decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "WorldPositions", type: Tvec4f, semantic: "WorldPositions", decorations: [{ kind: "Location", value: 0 }] },
          { name: "Colors",         type: Tvec4f, semantic: "Colors",         decorations: [{ kind: "Location", value: 1 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [
          { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
          { name: "WorldPositions",  type: Tvec4f, semantic: "WorldPositions",  decorations: [{ kind: "Location", value: 1 }] },
          { name: "Colors",          type: Tvec4f, semantic: "Colors",          decorations: [{ kind: "Location", value: 2 }] },
        ],
        outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
    ]);
  });
});
