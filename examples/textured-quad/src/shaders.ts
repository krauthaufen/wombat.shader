// Shader source — single string compiled to both GLSL and WGSL by the
// runtime. Demonstrates: a Uniform block (Globals) with mat4 + vec3 +
// f32 fields, a sampler2D bound at group 0 / slot 1 (WebGPU) /
// uniform sampler2D (WebGL2 — auto-binding), and an animated UV
// distortion driven by a time uniform.

export const SOURCE = `
function vsMain(input: { a_position: V2f; a_uv: V2f })
  : { gl_Position: V4f; v_uv: V2f }
{
  return {
    gl_Position: vec4(input.a_position.x, input.a_position.y, 0.0, 1.0),
    v_uv: input.a_uv,
  };
}

function fsMain(input: { v_uv: V2f })
  : { outColor: V4f }
{
  // Sample the procedural texture; tint with a time-driven sweep.
  const sampled = texture(u_tex, input.v_uv);
  const wave = sin(u_time);
  const r = mix(sampled.x, 1.0 - sampled.x, wave * 0.5 + 0.5);
  return { outColor: vec4(r, sampled.y, sampled.z, 1.0) };
}
`;
