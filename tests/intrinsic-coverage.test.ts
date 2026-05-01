// Phase 5 intrinsic-coverage additions: math (sinh/degrees/radians/trunc),
// packing (pack2x16float/unpack2x16float, pack4x8unorm/unpack4x8unorm),
// bit ops (countOneBits, extractBits, reverseBits, firstLeadingBit).

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import type { ValueDef } from "@aardworx/wombat.shader/ir";

function compileBoth(source: string): { glsl: string; wgsl: string } {
  const glsl = compileShaderSource(source, [{ name: "fsMain", stage: "fragment", outputs: [{
    name: "outColor",
    type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
    semantic: "Color",
    decorations: [{ kind: "Location", value: 0 }],
  }]}], { target: "glsl" });
  const wgsl = compileShaderSource(source, [{ name: "fsMain", stage: "fragment", outputs: [{
    name: "outColor",
    type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 },
    semantic: "Color",
    decorations: [{ kind: "Location", value: 0 }],
  }]}], { target: "wgsl" });
  return {
    glsl: glsl.stages[0]!.source,
    wgsl: wgsl.stages[0]!.source,
  };
}

describe("intrinsic coverage", () => {
  it("hyperbolic / degrees / radians / trunc emit identically (same WGSL/GLSL spelling)", () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): V4f {
        const a = sinh(input.v_uv.x);
        const b = cosh(input.v_uv.y);
        const c = degrees(input.v_uv.x);
        const d = radians(input.v_uv.y);
        const e = trunc(input.v_uv.x);
        return new V4f(a + c, b + d, e, 1.0);
      }
    `;
    const { glsl, wgsl } = compileBoth(src);
    for (const code of [glsl, wgsl]) {
      expect(code).toContain("sinh(");
      expect(code).toContain("cosh(");
      expect(code).toContain("degrees(");
      expect(code).toContain("radians(");
      expect(code).toContain("trunc(");
    }
  });

  it("packing intrinsics map to GLSL ES 3.00 spellings on the GLSL target", () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): V4f {
        const packed = pack2x16float(input.v_uv);
        const unpacked = unpack2x16float(packed);
        return new V4f(unpacked.x, unpacked.y, 0.5, 1.0);
      }
    `;
    const { glsl, wgsl } = compileBoth(src);
    expect(glsl).toContain("packHalf2x16(");
    expect(glsl).toContain("unpackHalf2x16(");
    expect(wgsl).toContain("pack2x16float(");
    expect(wgsl).toContain("unpack2x16float(");
  });

  it("4x8unorm packing maps to packUnorm4x8/unpackUnorm4x8 on GLSL", () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): V4f {
        const v = new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0);
        const packed = pack4x8unorm(v);
        return unpack4x8unorm(packed);
      }
    `;
    const { glsl, wgsl } = compileBoth(src);
    expect(glsl).toContain("packUnorm4x8(");
    expect(glsl).toContain("unpackUnorm4x8(");
    expect(wgsl).toContain("pack4x8unorm(");
    expect(wgsl).toContain("unpack4x8unorm(");
  });

  it("bit ops map to GLSL ES 3.00 names (bitCount, bitfieldExtract, ...)", () => {
    const src = `
      function fsMain(input: { v_uv: V2f }): V4f {
        const x: i32 = (input.v_uv.x as i32);
        const a = countOneBits(x);
        const b = firstLeadingBit(x);
        const c = firstTrailingBit(x);
        const d = reverseBits(x);
        const e = extractBits(x, 0, 8);
        return new V4f((a + b + c + d + e) as f32, 0, 0, 1);
      }
    `;
    const { glsl, wgsl } = compileBoth(src);
    expect(glsl).toContain("bitCount(");
    expect(glsl).toContain("findMSB(");
    expect(glsl).toContain("findLSB(");
    expect(glsl).toContain("bitfieldReverse(");
    expect(glsl).toContain("bitfieldExtract(");
    expect(wgsl).toContain("countOneBits(");
    expect(wgsl).toContain("firstLeadingBit(");
    expect(wgsl).toContain("firstTrailingBit(");
    expect(wgsl).toContain("reverseBits(");
    expect(wgsl).toContain("extractBits(");
  });
});
