// Instanced cubes via WebGPU. The Camera UBO carries view + projection
// + time. Per-instance offsets come from a second vertex buffer with
// step-mode "instance". A grid of 8×8×8 cubes is drawn with one draw
// call.
//
// Demonstrates:
//   - std140-equivalent WGSL uniform layout, with field offsets read
//     from the runtime's ProgramInterface (no manual byte counting)
//   - per-instance vertex attributes (position offset)
//   - depth testing
//   - real perspective + orbit camera

import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";

const log = (...args: unknown[]): void => {
  const el = document.getElementById("log")!;
  el.textContent += args.map(String).join(" ") + "\n";
  console.log(...args);
};

const SHADER = `
function vsMain(input: { a_position: V3f; a_normal: V3f; a_offset: V3f })
  : { gl_Position: V4f; v_normal: V3f; v_world: V3f }
{
  const world = vec3(input.a_position.x + input.a_offset.x,
                     input.a_position.y + input.a_offset.y,
                     input.a_position.z + input.a_offset.z);
  return {
    gl_Position: u_camera_projection.mul(u_camera_view.mul(vec4(world.x, world.y, world.z, 1.0))),
    v_normal: input.a_normal,
    v_world: world,
  };
}

function fsMain(input: { v_normal: V3f; v_world: V3f })
  : { outColor: V4f }
{
  // Lambertian-ish shade against a fixed light direction.
  const lightDir = vec3(0.577, 0.577, 0.577);
  const ndotl = max(input.v_normal.x * lightDir.x + input.v_normal.y * lightDir.y + input.v_normal.z * lightDir.z, 0.0);
  const ambient = 0.3;
  // Tint by world-space y so cubes higher up are bluer.
  const heightHue = 0.5 + 0.05 * input.v_world.y;
  const lit = vec3(0.9 * (ambient + 0.7 * ndotl) * (1.0 - heightHue * 0.4),
                   0.6 * (ambient + 0.7 * ndotl) * 0.8,
                   1.0 * (ambient + 0.7 * ndotl) * heightHue);
  return { outColor: vec4(lit.x, lit.y, lit.z, 1.0) };
}
`;

const Tf32 = { kind: "Float", width: 32 } as const;
const Tvec3f = { kind: "Vector", element: Tf32, dim: 3 } as const;
const Tvec4f = { kind: "Vector", element: Tf32, dim: 4 } as const;
const Tmat4f = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as const;

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    log("WebGPU not available; this demo requires WebGPU.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter returned null");
  const device = await adapter.requestDevice();
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: presentationFormat, alphaMode: "premultiplied" });

  const compiled = compileShaderSource(SHADER, [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_offset",   type: Tvec3f, semantic: "Offset",   decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_normal",    type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_world",     type: Tvec3f, semantic: "WorldPos", decorations: [{ kind: "Location", value: 1 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [
        { name: "v_normal", type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_world",  type: Tvec3f, semantic: "WorldPos", decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [{ name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ], {
    target: "wgsl",
    extraValues: [
      // A single uniform block "Camera" with two mat4s. Field offsets
      // come from the WGSL layout calculator; we read them out below
      // and write the bytes into the same slots.
      {
        kind: "Uniform",
        uniforms: [
          { name: "u_camera_view",       type: Tmat4f, buffer: "Camera", group: 0, slot: 0 },
          { name: "u_camera_projection", type: Tmat4f, buffer: "Camera", group: 0, slot: 0 },
        ],
      },
    ],
  });

  log(`stages: ${compiled.stages.map((s) => `${s.entryName}(${s.stage})`).join(", ")}`);
  for (const ub of compiled.interface.uniformBlocks) {
    log(`UBO ${ub.name} group=${ub.group} slot=${ub.slot} size=${ub.size}`);
    for (const f of ub.fields) {
      log(`  .${f.name} offset=${f.offset} size=${f.size} align=${f.align}`);
    }
  }
  log(`attributes: ${compiled.interface.attributes.map((a) => `${a.name}@${a.location}:${a.format}`).join(", ")}`);

  const modules = createShaderModules(device, compiled);

  // Cube geometry — 36 vertices, with per-vertex normal.
  const cube = makeCubeMesh();
  const vbo = device.createBuffer({ size: cube.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vbo, 0, cube);

  // Per-instance offsets — 8x8x8 grid of cubes.
  const N = 8;
  const offsets = new Float32Array(N * N * N * 3);
  let oi = 0;
  for (let z = 0; z < N; z++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        offsets[oi++] = (x - (N - 1) / 2) * 1.6;
        offsets[oi++] = (y - (N - 1) / 2) * 1.6;
        offsets[oi++] = (z - (N - 1) / 2) * 1.6;
      }
  const ibo = device.createBuffer({ size: offsets.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ibo, 0, offsets);

  // Camera UBO — read size from interface; the runtime computed it.
  const cameraBlock = compiled.interface.uniformBlocks.find((b) => b.name === "Camera")!;
  const cameraBuffer = device.createBuffer({ size: cameraBlock.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // Depth.
  const depth = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: modules.vertex!,
      entryPoint: "vsMain",
      buffers: [
        // Per-vertex (position, normal) — interleaved.
        {
          arrayStride: 6 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0,     format: "float32x3" },  // a_position
            { shaderLocation: 1, offset: 3 * 4, format: "float32x3" },  // a_normal
          ],
        },
        // Per-instance (offset).
        {
          arrayStride: 3 * 4,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 2, offset: 0, format: "float32x3" },      // a_offset
          ],
        },
      ],
    },
    fragment: {
      module: modules.fragment!,
      entryPoint: "fsMain",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { format: "depth24plus", depthCompare: "less", depthWriteEnabled: true },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: cameraBlock.slot, resource: { buffer: cameraBuffer } }],
  });

  const t0 = performance.now();
  const cameraData = new Float32Array(cameraBlock.size / 4);

  function updateCamera(t: number): void {
    // WGSL/GLSL read matrices column-major from uniform buffers; we
    // build column-major directly. Convention: WebGPU NDC z is [0, 1].
    const aspect = canvas.width / canvas.height;
    const fov = (60 * Math.PI) / 180;
    const near = 0.1, far = 100;
    const f = 1 / Math.tan(fov / 2);
    const proj = new Float32Array([
      // column 0
      f / aspect, 0, 0, 0,
      // column 1
      0, f, 0, 0,
      // column 2
      0, 0, far / (near - far), -1,
      // column 3
      0, 0, (far * near) / (near - far), 0,
    ]);

    const eyeX = 14 * Math.cos(t);
    const eyeZ = 14 * Math.sin(t);
    const eyeY = 8 + 2 * Math.sin(t * 0.5);
    const view = lookAtColMajor([eyeX, eyeY, eyeZ], [0, 0, 0], [0, 1, 0]);

    const fields = cameraBlock.fields;
    const viewField = fields.find((x) => x.name === "u_camera_view")!;
    const projField = fields.find((x) => x.name === "u_camera_projection")!;
    cameraData.set(view, viewField.offset / 4);
    cameraData.set(proj, projField.offset / 4);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData);
  }

  function frame(): void {
    const t = (performance.now() - t0) * 0.0004;
    updateCamera(t);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vbo);
    pass.setVertexBuffer(1, ibo);
    pass.setBindGroup(0, bindGroup);
    pass.draw(36, N * N * N);
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ─── Math helpers (CPU-side; not part of wombat) ─────────────────────

function lookAtColMajor(eye: number[], target: number[], up: number[]): Float32Array {
  // Right-handed lookAt. Column-major output for direct upload to a
  // GPU uniform buffer.
  const f = sub(target, eye); norm(f);
  const s = cross(f, up);     norm(s);
  const u = cross(s, f);
  // Row-major view of the matrix (rows of the rotation+translation):
  //   row0 = [ s.x,  s.y,  s.z, -s·eye]
  //   row1 = [ u.x,  u.y,  u.z, -u·eye]
  //   row2 = [-f.x, -f.y, -f.z,  f·eye]
  //   row3 = [   0,    0,    0,      1]
  // Column-major in memory transposes these.
  const m = new Float32Array(16);
  // column 0
  m[0]  =  s[0]!; m[1]  =  u[0]!; m[2]  = -f[0]!; m[3]  = 0;
  // column 1
  m[4]  =  s[1]!; m[5]  =  u[1]!; m[6]  = -f[1]!; m[7]  = 0;
  // column 2
  m[8]  =  s[2]!; m[9]  =  u[2]!; m[10] = -f[2]!; m[11] = 0;
  // column 3 (translation)
  m[12] = -dot(s, eye);
  m[13] = -dot(u, eye);
  m[14] =  dot(f, eye);
  m[15] = 1;
  return m;
}
function sub(a: number[], b: number[]): number[] { return [a[0]!-b[0]!, a[1]!-b[1]!, a[2]!-b[2]!]; }
function cross(a: number[], b: number[]): number[] {
  return [a[1]!*b[2]! - a[2]!*b[1]!, a[2]!*b[0]! - a[0]!*b[2]!, a[0]!*b[1]! - a[1]!*b[0]!];
}
function dot(a: number[], b: number[]): number { return a[0]!*b[0]! + a[1]!*b[1]! + a[2]!*b[2]!; }
function norm(v: number[]): void {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  if (l > 0) { v[0]! /= l; v[1]! /= l; v[2]! /= l; }
}

function makeCubeMesh(): Float32Array {
  // Each face: 2 triangles × 3 vertices × (position + normal) = 6 × 6 floats per face.
  const faces: Array<[number[], number[][]]> = [
    [[ 0,  0,  1], [[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1,-1, 1],[ 1, 1, 1],[-1, 1, 1]]],
    [[ 0,  0, -1], [[ 1,-1,-1],[-1,-1,-1],[-1, 1,-1],[ 1,-1,-1],[-1, 1,-1],[ 1, 1,-1]]],
    [[ 1,  0,  0], [[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]]],
    [[-1,  0,  0], [[-1,-1,-1],[-1,-1, 1],[-1, 1, 1],[-1,-1,-1],[-1, 1, 1],[-1, 1,-1]]],
    [[ 0,  1,  0], [[-1, 1, 1],[ 1, 1, 1],[ 1, 1,-1],[-1, 1, 1],[ 1, 1,-1],[-1, 1,-1]]],
    [[ 0, -1,  0], [[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1,-1],[ 1,-1, 1],[-1,-1, 1]]],
  ];
  const out = new Float32Array(36 * 6);
  let i = 0;
  for (const [n, tris] of faces) {
    for (const v of tris) {
      out[i++] = v[0]! * 0.5; out[i++] = v[1]! * 0.5; out[i++] = v[2]! * 0.5;
      out[i++] = n[0]!; out[i++] = n[1]!; out[i++] = n[2]!;
    }
  }
  return out;
}

main().catch((e) => {
  log("ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
});
