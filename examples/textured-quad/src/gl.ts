// WebGL2 path: compile to GLSL ES 3.00, link, drive uniform updates
// via the ProgramInterface (no manual `getUniformLocation` round-trips).

import { compileShaderSource } from "@aardworx/wombat.shader";
import { linkEffect } from "@aardworx/wombat.shader/webgl2";
import { SOURCE } from "./shaders.js";
import {
  Tvec2f, Tvec4f, commonValueDefs, log, makeCheckerboard, unitQuad,
} from "./common.js";

export function runWebGL2(canvas: HTMLCanvasElement): void {
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("WebGL2 not supported");

  const compiled = compileShaderSource(SOURCE, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_uv",       type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_uv",        type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_uv", type: Tvec2f, semantic: "Texcoord", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ], { target: "glsl", extraValues: commonValueDefs() });

  const program = linkEffect(gl, compiled);

  log("gl-log", "--- vertex ---\n" + compiled.stages.find((s) => s.stage === "vertex")!.source);
  log("gl-log", "--- fragment ---\n" + compiled.stages.find((s) => s.stage === "fragment")!.source);
  log("gl-log", "attributes:", [...compiled.interface.attributes.map((a) => `${a.name}@${a.location}:${a.format}`)].join(", "));
  log("gl-log", "uniforms:", [...compiled.interface.uniforms.map((u) => u.name)].join(", "));

  // VBO. Vertex layout is whatever the *application* picks; we use
  // the order from the interface to derive per-attribute offsets,
  // which matches the unitQuad() interleaved layout (position, uv).
  const geo = unitQuad();
  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, geo.vertices, gl.STATIC_DRAW);
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  const attribs = [...compiled.interface.attributes].sort((a, b) => a.location - b.location);
  const stride = attribs.reduce((s, a) => s + a.byteSize, 0);
  let offset = 0;
  for (const a of attribs) {
    const loc = program.attribs.get(a.name);
    if (loc !== undefined) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, a.components, gl.FLOAT, false, stride, offset);
    }
    offset += a.byteSize;
  }

  // Texture
  const tex = gl.createTexture()!;
  const texData = makeCheckerboard(64);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 64, 64, 0, gl.RGBA, gl.UNSIGNED_BYTE, texData);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

  const uTexLoc = program.uniforms.get("u_tex");
  const uTimeLoc = program.uniforms.get("u_time");

  function frame(t: number): void {
    if (!gl) throw new Error();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program.program);
    gl.bindVertexArray(vao);
    if (uTimeLoc) gl.uniform1f(uTimeLoc, t * 0.001);
    if (uTexLoc) gl.uniform1i(uTexLoc, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
