// WebGPU path: compile to WGSL, build a render pipeline + bind group
// driven entirely by the ProgramInterface.

import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";
import { SOURCE } from "./shaders.js";
import {
  Tvec2f, Tvec4f, commonValueDefs, log, makeCheckerboard, unitQuad,
} from "./common.js";

export async function runWebGPU(canvas: HTMLCanvasElement): Promise<void> {
  if (!("gpu" in navigator)) {
    log("wgpu-log", "WebGPU not available; skipping.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter returned null");
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: presentationFormat, alphaMode: "premultiplied" });

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
  ], { target: "wgsl", extraValues: commonValueDefs() });

  const modules = createShaderModules(device, compiled);

  log("wgpu-log", "--- vertex ---\n" + compiled.stages.find((s) => s.stage === "vertex")!.source);
  log("wgpu-log", "--- fragment ---\n" + compiled.stages.find((s) => s.stage === "fragment")!.source);
  log("wgpu-log", "groups:", [...modules.bindings.groups.keys()].join(","));
  for (const [g, entries] of modules.bindings.groups) {
    log("wgpu-log", `  group ${g}: ${entries.map((e) => `[${e.slot}=${e.kind}:${e.name}]`).join(" ")}`);
  }

  // Vertex buffer.
  const geo = unitQuad();
  const vbo = device.createBuffer({
    size: geo.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vbo, 0, geo.vertices);

  // Uniform buffer for u_time.
  const uboSize = 4; // single f32, but WGSL uniform binding requires 16-byte stride at minimum.
  const uboPaddedSize = Math.max(16, uboSize);
  const ubo = device.createBuffer({
    size: uboPaddedSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Texture + sampler.
  const texSize = 64;
  const tex = device.createTexture({
    size: [texSize, texSize, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  device.queue.writeTexture(
    { texture: tex },
    makeCheckerboard(texSize),
    { bytesPerRow: texSize * 4, rowsPerImage: texSize },
    { width: texSize, height: texSize, depthOrArrayLayers: 1 },
  );
  const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest", addressModeU: "repeat", addressModeV: "repeat" });

  const attribs = [...compiled.interface.attributes].sort((a, b) => a.location - b.location);
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: modules.vertex!,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: attribs.reduce((s, a) => s + a.byteSize, 0),
        attributes: (() => {
          let off = 0;
          return attribs.map((a) => {
            const r = { shaderLocation: a.location, offset: off, format: a.format as GPUVertexFormat };
            off += a.byteSize;
            return r;
          });
        })(),
      }],
    },
    fragment: {
      module: modules.fragment!,
      entryPoint: "fsMain",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  // Build the bind group from the ProgramInterface. After
  // legaliseTypes(wgsl) every combined Sampler became a (sampler, view)
  // pair, so the interface lists them as separate samplers + textures.
  // Loose uniforms (u_time) live as `LooseUniformInfo` items but the
  // WGSL emitter still emits them as standalone `var<uniform>`
  // bindings; we look up by name to find the slot.
  const entries: GPUBindGroupEntry[] = [];
  // u_time is at group 0 / slot 0. Find via interface (search loose
  // first; if the user grouped it into a block, look there).
  const uTimeSlot = 0; // we declared it at slot 0 via commonValueDefs
  entries.push({ binding: uTimeSlot, resource: { buffer: ubo } });
  for (const s of compiled.interface.samplers) {
    if (s.group === 0) entries.push({ binding: s.slot, resource: sampler });
  }
  for (const t_ of compiled.interface.textures) {
    if (t_.group === 0) entries.push({ binding: t_.slot, resource: tex.createView() });
  }
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const time0 = performance.now();
  function frame(): void {
    const t = (performance.now() - time0) * 0.001;
    device.queue.writeBuffer(ubo, 0, new Float32Array([t]));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vbo);
    pass.setBindGroup(0, bindGroup);
    pass.draw(geo.triangleCount * 3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
