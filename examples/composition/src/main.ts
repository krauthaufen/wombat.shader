// Composition demo. Two fragment shaders fuse via composeStages:
//
//   `base`        produces a base colour from UV.
//   `colourGrade` reads `base_color` and applies a saturation tint.
//
// Both are compiled together; the optimiser fuses them into a single
// fragment Entry with an internal carrier var connecting them. The
// `before` panel shows what happens with skipOptimisations: two
// fragment functions in the IR. The `after` panel shows the fused
// result.

import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { linkEffect } from "@aardworx/wombat.shader-runtime/webgl2";

const log = (id: string, ...args: unknown[]): void => {
  const el = document.getElementById(id);
  if (el) el.textContent += args.map(String).join(" ") + "\n";
};

const SOURCE = `
function vsMain(input: { a_position: V2f; a_uv: V2f })
  : { gl_Position: V4f; v_uv: V2f }
{
  return {
    gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
    v_uv: input.a_uv,
  };
}

// Stage A: base — a procedural radial gradient.
function base(input: { v_uv: V2f })
  : { base_color: V3f }
{
  const cx = input.v_uv.x - 0.5;
  const cy = input.v_uv.y - 0.5;
  const r = sqrt(cx * cx + cy * cy);
  return { base_color: new V3f(0.4 + 0.6 * (1.0 - r), 0.2, 0.8 - 0.6 * r) };
}

// Stage B: colour grade — boosts saturation per channel.
// Reads base_color from stage A, writes outColor.
function colourGrade(input: { base_color: V3f })
  : { outColor: V4f }
{
  const tint = new V3f(input.base_color.x * 1.4, input.base_color.y + 0.2, input.base_color.z * 1.2);
  return { outColor: new V4f(tint.x, tint.y, tint.z, 1.0) };
}
`;

const Tf32 = { kind: "Float", width: 32 } as const;
const Tvec2f = { kind: "Vector", element: Tf32, dim: 2 } as const;
const Tvec3f = { kind: "Vector", element: Tf32, dim: 3 } as const;
const Tvec4f = { kind: "Vector", element: Tf32, dim: 4 } as const;

function entries() {
  return [
    {
      name: "vsMain", stage: "vertex" as const,
      inputs: [
        { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location" as const, value: 0 }] },
        { name: "a_uv",       type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location" as const, value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin" as const, value: "position" as const }] },
        { name: "v_uv",        type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location" as const, value: 0 }] },
      ],
    },
    {
      name: "base", stage: "fragment" as const,
      inputs: [{ name: "v_uv", type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location" as const, value: 0 }] }],
      outputs: [{ name: "base_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location" as const, value: 0 }] }],
    },
    {
      name: "colourGrade", stage: "fragment" as const,
      inputs: [{ name: "base_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location" as const, value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location" as const, value: 0 }] }],
    },
  ];
}

function main(): void {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("WebGL2 not supported");

  // Without optimisations: two fragment Entries side-by-side in the
  // IR. (The runtime can still emit them as separate strings.)
  const before = compileShaderSource(SOURCE, entries(), { target: "glsl", skipOptimisations: true });
  log("before", `// ${before.stages.length} stages emitted (vertex + 2 fragments)`);
  for (const s of before.stages) {
    log("before", `// --- ${s.entryName} (${s.stage}) ---`);
    log("before", s.source);
  }

  // With optimisations on, composeStages fuses base + colourGrade
  // into a single fragment entry whose body internalises base_color
  // as a carrier `_pipe_base_color` var.
  const after = compileShaderSource(SOURCE, entries(), { target: "glsl" });
  log("after", `// ${after.stages.length} stages emitted (vertex + fused fragment)`);
  for (const s of after.stages) {
    log("after", `// --- ${s.entryName} (${s.stage}) ---`);
    log("after", s.source);
  }
  log("log", `before: ${before.stages.length} stages, ${before.stages.map((s) => s.stage).join(", ")}`);
  log("log", `after:  ${after.stages.length} stages, ${after.stages.map((s) => s.stage).join(", ")}`);
  log("log", `attributes:`, JSON.stringify(after.interface.attributes.map((a) => ({ name: a.name, location: a.location, format: a.format }))));
  log("log", `outputs:   `, JSON.stringify(after.interface.fragmentOutputs.map((o) => ({ name: o.name, location: o.location }))));

  // Render the fused effect.
  const program = linkEffect(gl, after);
  const data = new Float32Array([
    -1, -1, 0, 0,   1, -1, 1, 0,   -1, 1, 0, 1,
     1,  1, 1, 1,  -1,  1, 0, 1,    1,-1, 1, 0,
  ]);
  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  for (const a of [...after.interface.attributes].sort((x, y) => x.location - y.location)) {
    const loc = program.attribs.get(a.name);
    if (loc !== undefined) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, a.components, gl.FLOAT, false, 16, a.location * 8);
    }
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program.program);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

try {
  main();
} catch (e) {
  log("log", "ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
}
