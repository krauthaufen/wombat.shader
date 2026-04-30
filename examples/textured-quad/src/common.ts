// Shared helpers used by both the WebGL2 and WebGPU paths.
//
// Both back-ends compile the SAME shader source through the runtime's
// `compileShaderSource`, just with different `target`s. The
// `ProgramInterface` returned by the compile gives us everything else
// we need — attribute formats, uniform-block layouts (with field
// offsets), sampler / texture bindings.

import type { Module, Type, ValueDef } from "@aardworx/wombat.shader-ir";

export const Tf32: Type = { kind: "Float", width: 32 };
export const Tvec2f: Type = { kind: "Vector", element: Tf32, dim: 2 };
export const Tvec3f: Type = { kind: "Vector", element: Tf32, dim: 3 };
export const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };
export const TsamplerLinear2D: Type = { kind: "Sampler", target: "2D", sampled: { kind: "Float" }, comparison: false };

export interface QuadGeometry {
  /** Interleaved [x, y, u, v]. */
  readonly vertices: Float32Array;
  /** Number of triangles to draw. */
  readonly triangleCount: number;
}

/** A 2-triangle full-screen quad with UVs in [0, 1]. */
export function unitQuad(): QuadGeometry {
  const vertices = new Float32Array([
    // x,    y,    u,  v
    -1.0, -1.0,  0,  0,
     1.0, -1.0,  1,  0,
    -1.0,  1.0,  0,  1,
     1.0,  1.0,  1,  1,
    -1.0,  1.0,  0,  1,
     1.0, -1.0,  1,  0,
  ]);
  return { vertices, triangleCount: 2 };
}

/** Procedural 64×64 RGBA checkerboard with a hue gradient on the squares. */
export function makeCheckerboard(size = 64): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const cell = ((x >> 3) ^ (y >> 3)) & 1;
      const u = x / size, v = y / size;
      if (cell) {
        out[i + 0] = Math.round(255 * u);
        out[i + 1] = Math.round(255 * v);
        out[i + 2] = 64;
        out[i + 3] = 255;
      } else {
        out[i + 0] = 32;
        out[i + 1] = 32;
        out[i + 2] = Math.round(255 * (0.4 + 0.6 * (1 - u)));
        out[i + 3] = 255;
      }
    }
  }
  return out;
}

/** Module-level value definitions shared by both targets. */
export function commonValueDefs(): readonly ValueDef[] {
  return [
    {
      kind: "Uniform",
      uniforms: [
        // Standalone uniform (no block). GLSL emits `uniform float u_time;`,
        // WGSL emits `@group(0) @binding(0) var<uniform> u_time: f32;`.
        // Both backends accept `u_time` as a free identifier in the
        // shader body, no struct prefix needed.
        { name: "u_time", type: Tf32, group: 0, slot: 0 },
      ],
    },
    {
      kind: "Sampler",
      binding: { group: 0, slot: 1 },
      name: "u_tex",
      type: TsamplerLinear2D,
    },
  ];
}

/** Append the common defs to a Module. */
export function withCommonDefs(module: Module): Module {
  return { ...module, values: [...commonValueDefs(), ...module.values] };
}

export const log = (id: string, ...args: unknown[]): void => {
  console.log(...args);
  const el = document.getElementById(id);
  if (el) el.textContent += args.map(String).join(" ") + "\n";
};
