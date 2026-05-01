// Compute kernel example. Source-form shader fed through the frontend:
// the body uses `ComputeBuiltins` for `@builtin(global_invocation_id)`
// access, the `@workgroupSize` JSDoc tag carries the workgroup size,
// and reads/writes against `out_buffer` are inferred — the storage
// buffer becomes `read_write` automatically because the source assigns
// to it.

import { compileShaderSource } from "@aardworx/wombat.shader";
import { createShaderModules } from "@aardworx/wombat.shader/webgpu";
import type { ValueDef } from "@aardworx/wombat.shader/ir";

const log = (...args: unknown[]): void => {
  const el = document.getElementById("log")!;
  el.textContent += args.map(String).join(" ") + "\n";
  console.log(...args);
};

const SIZE = 64;
const MAX_ITER = 64;

const SHADER_SRC = `
import type { ComputeBuiltins } from "@aardworx/wombat.shader/types";

declare const out_buffer: number[];

const SIZE = ${SIZE};
const MAX_ITER = ${MAX_ITER};

/** @workgroupSize 8 8 1 */
export function csMain(b: ComputeBuiltins): void {
  const ix = b.globalInvocationId.x as i32;
  const iy = b.globalInvocationId.y as i32;
  if (ix >= SIZE || iy >= SIZE) { return; }
  const cx = (ix as f32) / (SIZE as f32) * 3.0 - 2.0;
  const cy = (iy as f32) / (SIZE as f32) * 2.4 - 1.2;
  let zx: f32 = 0.0;
  let zy: f32 = 0.0;
  let iter: i32 = 0;
  for (let i: i32 = 0; i < MAX_ITER; ++i) {
    const zx2 = zx * zx;
    const zy2 = zy * zy;
    if (zx2 + zy2 > 4.0) { iter = i; break; }
    const nx = zx2 - zy2 + cx;
    const ny = 2.0 * zx * zy + cy;
    zx = nx;
    zy = ny;
    iter = i;
  }
  out_buffer[iy * SIZE + ix] = iter as u32;
}
`;

function buildExtraValues(): readonly ValueDef[] {
  // Storage-buffer ValueDef. The frontend resolves `out_buffer` to this
  // declaration via the externalTypes table; the inferStorageAccess
  // pass flips access to read_write because the body writes to it.
  return [
    {
      kind: "StorageBuffer",
      binding: { group: 0, slot: 0 },
      name: "out_buffer",
      layout: {
        kind: "Array",
        element: { kind: "Int", signed: false, width: 32 },
        length: SIZE * SIZE,
      },
      access: "read", // overridden by inferStorageAccess
    },
  ];
}

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    log("WebGPU not available; this demo requires WebGPU.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter returned null");
  const device = await adapter.requestDevice();

  const compiled = compileShaderSource(
    SHADER_SRC,
    [{ name: "csMain", stage: "compute" }],
    { target: "wgsl", extraValues: buildExtraValues() },
  );
  const stage = compiled.stages.find((s) => s.stage === "compute")!;
  log(stage.source);
  log(`storage buffers: ${JSON.stringify(compiled.interface.storageBuffers.map((b) => ({
    name: b.name, group: b.group, slot: b.slot, size: b.size, access: b.access,
  })))}`);

  const modules = createShaderModules(device, compiled);

  const bufSize = SIZE * SIZE * 4;
  const storage = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: modules.compute!, entryPoint: "csMain" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: storage } }],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(SIZE / 8, SIZE / 8, 1);
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, bufSize);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // Render to canvas: hue-mapped by iteration count.
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const it = data[y * SIZE + x] ?? 0;
      const t = it / MAX_ITER;
      const r = Math.round(255 * Math.sin(t * Math.PI) ** 2);
      const g = Math.round(255 * Math.sin(t * Math.PI + 1.0) ** 2);
      const b = Math.round(255 * Math.sin(t * Math.PI + 2.0) ** 2);
      const ip = (y * SIZE + x) * 4;
      img.data[ip + 0] = r;
      img.data[ip + 1] = g;
      img.data[ip + 2] = b;
      img.data[ip + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  log(`compute done. saturated pixels: ${data.filter((x) => x >= MAX_ITER - 1).length}; zero-iter: ${data.filter((x) => x === 0).length}`);
}

main().catch((e) => {
  log("ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
});
