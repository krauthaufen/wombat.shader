// WebGL2 Program — wraps a vertex+fragment compile into a linked
// program plus a binding map for setting uniforms / samplers /
// attribute locations at draw time.

import type { CompiledEffect, CompiledStage } from "../compile.js";

export interface WebGL2Program {
  readonly gl: WebGL2RenderingContext;
  readonly program: WebGLProgram;
  readonly attribs: ReadonlyMap<string, number>;
  readonly uniforms: ReadonlyMap<string, WebGLUniformLocation>;
  readonly uniformBlocks: ReadonlyMap<string, number>;
  /** Free underlying GL resources. */
  dispose(): void;
}

export class ShaderCompileError extends Error {
  constructor(stage: string, log: string, source: string) {
    super(`${stage} compile failed:\n${log}\n\n--- source ---\n${source}`);
    this.name = "ShaderCompileError";
  }
}

export class ProgramLinkError extends Error {
  constructor(log: string) {
    super(`program link failed:\n${log}`);
    this.name = "ProgramLinkError";
  }
}

export function linkEffect(
  gl: WebGL2RenderingContext,
  effect: CompiledEffect,
): WebGL2Program {
  if (effect.target !== "glsl") {
    throw new Error(`linkEffect (WebGL2) requires a GLSL effect; got ${effect.target}`);
  }
  const vertex = stage(effect, "vertex");
  const fragment = stage(effect, "fragment");
  if (!vertex) throw new Error("linkEffect: missing vertex stage");
  if (!fragment) throw new Error("linkEffect: missing fragment stage");

  const vs = compile(gl, gl.VERTEX_SHADER, vertex.source);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragment.source);

  const program = gl.createProgram();
  if (!program) throw new Error("createProgram returned null");
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? "(no log)";
    gl.deleteProgram(program);
    gl.deleteShader(vs); gl.deleteShader(fs);
    throw new ProgramLinkError(log);
  }
  // Shaders can be deleted after the program links; the GL keeps them
  // alive as long as the program needs them.
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  const attribs = new Map<string, number>();
  for (const b of vertex.bindings.inputs) {
    const loc = gl.getAttribLocation(program, b.name);
    if (loc >= 0) attribs.set(b.name, loc);
  }

  const uniforms = new Map<string, WebGLUniformLocation>();
  // Loop both stages' uniforms; they often coincide.
  for (const stage of [vertex, fragment]) {
    for (const u of stage.bindings.uniforms) {
      if (uniforms.has(u.name)) continue;
      const loc = gl.getUniformLocation(program, u.name);
      if (loc !== null) uniforms.set(u.name, loc);
    }
    for (const s of stage.bindings.samplers) {
      if (uniforms.has(s.name)) continue;
      const loc = gl.getUniformLocation(program, s.name);
      if (loc !== null) uniforms.set(s.name, loc);
    }
  }

  // UBO blocks (binding GLSL ES 3.00 std140 blocks).
  const uniformBlocks = new Map<string, number>();
  const blockCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORM_BLOCKS) as number;
  for (let i = 0; i < blockCount; i++) {
    const name = gl.getActiveUniformBlockName(program, i);
    if (!name) continue;
    uniformBlocks.set(name, i);
    gl.uniformBlockBinding(program, i, i);
  }

  return {
    gl,
    program,
    attribs,
    uniforms,
    uniformBlocks,
    dispose() {
      gl.deleteProgram(program);
    },
  };
}

function stage(effect: CompiledEffect, kind: "vertex" | "fragment"): CompiledStage | undefined {
  return effect.stages.find((s) => s.stage === kind);
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader returned null");
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? "(no log)";
    gl.deleteShader(sh);
    const stageName = type === gl.VERTEX_SHADER ? "vertex" : "fragment";
    throw new ShaderCompileError(stageName, log, source);
  }
  return sh;
}
