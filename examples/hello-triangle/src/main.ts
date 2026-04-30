// Hello triangle — written as plain TypeScript shader source, compiled
// at runtime by `compileShaderSource`, linked into a WebGL2 program.
//
// In a real app, the `compileShaderSource` step would be moved to a
// build-time vite plugin so the browser only sees emitted GLSL/WGSL.
// We do it inline here to keep the example self-contained.

import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { linkEffect } from "@aardworx/wombat.shader-runtime/webgl2";

const log = (...args: unknown[]): void => {
  console.log(...args);
  document.getElementById("log")!.textContent += args.map(String).join(" ") + "\n";
};

const SHADER_SOURCE = `
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

function main(): void {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("WebGL2 not supported");

  const Tvec2f = { kind: "Vector" as const, element: { kind: "Float" as const, width: 32 as const }, dim: 2 as const };
  const Tvec3f = { kind: "Vector" as const, element: { kind: "Float" as const, width: 32 as const }, dim: 3 as const };
  const Tvec4f = { kind: "Vector" as const, element: { kind: "Float" as const, width: 32 as const }, dim: 4 as const };

  const compiled = compileShaderSource(SHADER_SOURCE, [
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
      inputs: [
        { name: "v_color", type: Tvec3f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ], { target: "glsl" });

  log("--- vertex shader ---\n" + compiled.stages.find((s) => s.stage === "vertex")!.source);
  log("--- fragment shader ---\n" + compiled.stages.find((s) => s.stage === "fragment")!.source);

  const program = linkEffect(gl, compiled);
  log("linked! attributes:", [...program.attribs]);

  // Build VBO — interleaved [x, y, r, g, b].
  const data = new Float32Array([
    // x,    y,    r, g, b
       0.0,  0.7,  1, 0, 0,
      -0.7, -0.5,  0, 1, 0,
       0.7, -0.5,  0, 0, 1,
  ]);
  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  const stride = 5 * 4;
  const posLoc = program.attribs.get("a_position")!;
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
  const colLoc = program.attribs.get("a_color")!;
  gl.enableVertexAttribArray(colLoc);
  gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, stride, 2 * 4);

  // Draw.
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program.program);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  log("drew triangle. if you see RGB on the canvas, the toolchain works.");
}

try {
  main();
} catch (e) {
  log("ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
}
