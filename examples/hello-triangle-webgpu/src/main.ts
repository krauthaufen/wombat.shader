// Hello triangle on WebGPU. Same shader source as the WebGL2 example,
// but compiled to WGSL and submitted through a WebGPU render pipeline.

import { compileShaderSource } from "@aardworx/wombat.shader-runtime";
import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";

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

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU not available in this browser");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter returned null");
  const device = await adapter.requestDevice();

  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: presentationFormat, alphaMode: "premultiplied" });

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
  ], { target: "wgsl" });

  log("--- vertex (WGSL) ---\n" + compiled.stages.find((s) => s.stage === "vertex")!.source);
  log("--- fragment (WGSL) ---\n" + compiled.stages.find((s) => s.stage === "fragment")!.source);

  const modules = createShaderModules(device, compiled);

  // Vertex buffer: interleaved [x, y, r, g, b].
  const data = new Float32Array([
    // x,    y,    r, g, b
       0.0,  0.7,  1, 0, 0,
      -0.7, -0.5,  0, 1, 0,
       0.7, -0.5,  0, 0, 1,
  ]);
  const vbo = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vbo, 0, data);

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: modules.vertex!,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: 5 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0,     format: "float32x2" }, // a_position
          { shaderLocation: 1, offset: 2 * 4, format: "float32x3" }, // a_color
        ],
      }],
    },
    fragment: {
      module: modules.fragment!,
      entryPoint: "fsMain",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, vbo);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);

  log("submitted. if you see RGB on the canvas, WGSL works end-to-end.");
}

main().catch((e) => {
  log("ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
});
