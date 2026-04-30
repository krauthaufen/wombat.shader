// Tests for the ProgramInterface and the std140 / WGSL layout
// calculators. Verifies that callers can read attribute formats,
// uniform-block field offsets, and binding info from the compile
// result without doing manual string matching.

import { describe, expect, it } from "vitest";
import {
  Mat,
  Tf32,
  Ti32,
  Tvoid,
  Vec,
  type Type,
} from "@aardworx/wombat.shader-ir";
import { compileModule, compileShaderSource, computeLayout } from "@aardworx/wombat.shader-runtime";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Tmat4f: Type = Mat(Tf32, 4, 4);

// ─── std140 layout primitives ────────────────────────────────────────

describe("std140 layout primitives", () => {
  it("scalars: size 4 align 4", () => {
    expect(computeLayout(Tf32, "glsl-std140")).toMatchObject({ size: 4, align: 4 });
    expect(computeLayout(Ti32, "glsl-std140")).toMatchObject({ size: 4, align: 4 });
  });

  it("vec2: size 8 align 8", () => {
    expect(computeLayout(Tvec2f, "glsl-std140")).toMatchObject({ size: 8, align: 8 });
  });

  it("vec3: size 12 align 16 (the std140 trap)", () => {
    expect(computeLayout(Tvec3f, "glsl-std140")).toMatchObject({ size: 12, align: 16 });
  });

  it("vec4: size 16 align 16", () => {
    expect(computeLayout(Tvec4f, "glsl-std140")).toMatchObject({ size: 16, align: 16 });
  });

  it("mat4: size 64, align 16, stride 16", () => {
    const l = computeLayout(Tmat4f, "glsl-std140");
    expect(l.size).toBe(64);
    expect(l.align).toBe(16);
    expect(l.stride).toBe(16);
  });

  it("array<f32, 4> in std140 pads each element to 16 bytes", () => {
    const arr: Type = { kind: "Array", element: Tf32, length: 4 };
    const l = computeLayout(arr, "glsl-std140");
    expect(l.stride).toBe(16);
    expect(l.size).toBe(64);
  });
});

// ─── std140 struct layout ────────────────────────────────────────────

describe("std140 struct layout — field offsets", () => {
  it("Camera { view: mat4, projection: mat4, eye: vec3, _pad: f32, near: f32, far: f32 }", () => {
    const Camera: Type = {
      kind: "Struct", name: "Camera",
      fields: [
        { name: "view",       type: Tmat4f },
        { name: "projection", type: Tmat4f },
        { name: "eye",        type: Tvec3f },
        { name: "near",       type: Tf32 },
        { name: "far",        type: Tf32 },
      ],
    };
    const l = computeLayout(Camera, "glsl-std140");
    expect(l.fields).toBeTruthy();
    const fields = l.fields!;
    expect(fields[0]!.name).toBe("view");
    expect(fields[0]!.offset).toBe(0);
    expect(fields[1]!.name).toBe("projection");
    expect(fields[1]!.offset).toBe(64); // after mat4 (64 bytes)
    expect(fields[2]!.name).toBe("eye");
    expect(fields[2]!.offset).toBe(128); // after second mat4
    expect(fields[3]!.name).toBe("near");
    // eye is vec3 (size 12, align 16). The next f32 starts at 128 + 12 = 140.
    // f32 align is 4, so no pad — offset 140.
    expect(fields[3]!.offset).toBe(140);
    expect(fields[4]!.name).toBe("far");
    expect(fields[4]!.offset).toBe(144);
    // total size rounds up to maxAlign (16): 148 → 160.
    expect(l.size).toBe(160);
  });
});

// ─── ProgramInterface from a hand-built Module ───────────────────────

describe("ProgramInterface — basic shape", () => {
  const u_mvp = { name: "u_mvp", type: Tmat4f, buffer: "Globals" } as const;
  const module = {
    types: [],
    values: [
      { kind: "Uniform" as const, uniforms: [u_mvp] },
      {
        kind: "Entry" as const,
        entry: {
          name: "vsMain", stage: "vertex" as const,
          inputs: [
            { name: "a_position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location" as const, value: 0 }] },
            { name: "a_color",    type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location" as const, value: 1 }] },
          ],
          outputs: [
            { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin" as const, value: "position" as const }] },
          ],
          arguments: [], returnType: Tvoid,
          body: { kind: "Nop" as const },
          decorations: [],
        },
      },
    ],
  };

  it("vertex attributes show up with locations and formats", () => {
    const r = compileModule(module, { target: "glsl", skipOptimisations: true });
    const attribs = r.interface.attributes;
    expect(attribs.length).toBe(2);
    expect(attribs.find((a) => a.name === "a_position")).toMatchObject({ location: 0, format: "float32x3", components: 3, byteSize: 12 });
    expect(attribs.find((a) => a.name === "a_color")).toMatchObject({ location: 1, format: "float32x3" });
  });

  it("uniform block — std140 layout, total size, fields with offsets", () => {
    const r = compileModule(module, { target: "glsl", skipOptimisations: true });
    const block = r.interface.uniformBlocks.find((b) => b.name === "Globals");
    expect(block).toBeTruthy();
    expect(block!.size).toBe(64); // mat4 only
    expect(block!.fields[0]!).toMatchObject({ name: "u_mvp", offset: 0, size: 64, align: 16 });
  });

  it("WGSL target: same uniform block via WGSL layout rules", () => {
    const r = compileModule(module, { target: "wgsl", skipOptimisations: true });
    const block = r.interface.uniformBlocks.find((b) => b.name === "Globals");
    expect(block).toBeTruthy();
    expect(block!.size).toBe(64);
    expect(block!.fields[0]!.offset).toBe(0);
  });
});

// ─── End-to-end: from TS source ──────────────────────────────────────

describe("ProgramInterface — end-to-end from TS source", () => {
  it("hello-triangle interface has the exact shape callers need", () => {
    const src = `
      function vsMain(input: { a_position: V2f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
        return {
          gl_Position: vec4(input.a_position.x, input.a_position.y, 0.0, 1.0),
          v_color: input.a_color,
        };
      }
      function fsMain(input: { v_color: V3f }): { outColor: V4f } {
        return { outColor: vec4(input.v_color.x, input.v_color.y, input.v_color.z, 1.0) };
      }
    `;
    const r = compileShaderSource(src, [
      {
        name: "vsMain", stage: "vertex",
        inputs: [
          { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
          { name: "a_color",    type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 1 }] },
        ],
        outputs: [
          { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
          { name: "v_color",     type: Tvec3f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
        ],
      },
      {
        name: "fsMain", stage: "fragment",
        inputs: [{ name: "v_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
        outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      },
    ], { target: "glsl" });

    expect(r.interface.attributes.map((a) => `${a.name}@${a.location}:${a.format}`)).toEqual([
      "a_position@0:float32x2",
      "a_color@1:float32x3",
    ]);
    expect(r.interface.fragmentOutputs.map((o) => `${o.name}@${o.location}`)).toEqual([
      "outColor@0",
    ]);
    expect(r.interface.uniformBlocks).toEqual([]);
    expect(r.interface.samplers).toEqual([]);
    expect(r.interface.storageBuffers).toEqual([]);
    expect(r.interface.stages.length).toBe(2);
    expect(r.interface.stages.every((s) => s.source.length > 0)).toBe(true);
  });
});
