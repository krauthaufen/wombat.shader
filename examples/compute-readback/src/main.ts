// Compute kernel example. Hand-builds an IR Module (instead of going
// through the TS-source frontend, which doesn't yet have first-class
// support for compute @builtin arguments). Dispatches the kernel,
// reads back the storage buffer, draws into a 2D canvas.

import { compileModule } from "@aardworx/wombat.shader-runtime";
import { createShaderModules } from "@aardworx/wombat.shader-runtime/webgpu";
import type { Expr, LExpr, Module, Stmt, Type, Var } from "@aardworx/wombat.shader-ir";

const log = (...args: unknown[]): void => {
  const el = document.getElementById("log")!;
  el.textContent += args.map(String).join(" ") + "\n";
  console.log(...args);
};

const SIZE = 64;
const MAX_ITER = 64;

const Tf32: Type = { kind: "Float", width: 32 };
const Tu32: Type = { kind: "Int", signed: false, width: 32 };
const Ti32: Type = { kind: "Int", signed: true, width: 32 };
const Tbool: Type = { kind: "Bool" };
const Tvec3u: Type = { kind: "Vector", element: Tu32, dim: 3 };
const Tvoid: Type = { kind: "Void" };

const constI = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });
const v = (va: Var): Expr => ({ kind: "Var", var: va, type: va.type });
const lvar = (va: Var): LExpr => ({ kind: "LVar", var: va, type: va.type });
const swizU = (e: Expr, c: "x" | "y" | "z"): Expr => ({ kind: "VecSwizzle", value: e, comps: [c], type: Tu32 });

function buildKernelIR(): Module {
  const gidParam = {
    name: "gid", type: Tvec3u, semantic: "GlobalInvocationId",
    decorations: [{ kind: "Builtin" as const, value: "global_invocation_id" as const }],
  };

  const ix: Var = { name: "ix", type: Ti32, mutable: false };
  const iy: Var = { name: "iy", type: Ti32, mutable: false };

  // Compute @builtin args are referenced as bare identifiers, not
  // through the per-stage input struct.
  const gidRead: Expr = { kind: "ReadInput", scope: "Builtin", name: "gid", type: Tvec3u };
  const ixExpr: Expr = { kind: "Convert", value: swizU(gidRead, "x"), type: Ti32 };
  const iyExpr: Expr = { kind: "Convert", value: swizU(gidRead, "y"), type: Ti32 };

  const cx: Var = { name: "cx", type: Tf32, mutable: false };
  const cy: Var = { name: "cy", type: Tf32, mutable: false };
  const zx: Var = { name: "zx", type: Tf32, mutable: true };
  const zy: Var = { name: "zy", type: Tf32, mutable: true };
  const iter: Var = { name: "iter", type: Ti32, mutable: true };
  const i: Var = { name: "i", type: Ti32, mutable: true };
  const zx2v: Var = { name: "zx2", type: Tf32, mutable: false };
  const zy2v: Var = { name: "zy2", type: Tf32, mutable: false };
  const nx: Var = { name: "nx", type: Tf32, mutable: false };
  const ny: Var = { name: "ny", type: Tf32, mutable: false };

  const cxInit: Expr = {
    kind: "Sub",
    lhs: { kind: "Mul",
      lhs: { kind: "Div", lhs: { kind: "Convert", value: v(ix), type: Tf32 }, rhs: constF(SIZE), type: Tf32 },
      rhs: constF(3), type: Tf32 },
    rhs: constF(2), type: Tf32,
  };
  const cyInit: Expr = {
    kind: "Sub",
    lhs: { kind: "Mul",
      lhs: { kind: "Div", lhs: { kind: "Convert", value: v(iy), type: Tf32 }, rhs: constF(SIZE), type: Tf32 },
      rhs: constF(2.4), type: Tf32 },
    rhs: constF(1.2), type: Tf32,
  };

  const innerLoop: Stmt = {
    kind: "Sequential",
    body: [
      { kind: "Declare", var: zx2v, init: { kind: "Expr", value: { kind: "Mul", lhs: v(zx), rhs: v(zx), type: Tf32 } } },
      { kind: "Declare", var: zy2v, init: { kind: "Expr", value: { kind: "Mul", lhs: v(zy), rhs: v(zy), type: Tf32 } } },
      {
        kind: "If",
        cond: { kind: "Gt", lhs: { kind: "Add", lhs: v(zx2v), rhs: v(zy2v), type: Tf32 }, rhs: constF(4), type: Tbool },
        then: { kind: "Sequential", body: [
          { kind: "Write", target: lvar(iter), value: v(i) },
          { kind: "Break" },
        ]},
      },
      { kind: "Declare", var: nx, init: { kind: "Expr", value: { kind: "Add", lhs: { kind: "Sub", lhs: v(zx2v), rhs: v(zy2v), type: Tf32 }, rhs: v(cx), type: Tf32 } } },
      { kind: "Declare", var: ny, init: { kind: "Expr", value: { kind: "Add", lhs: { kind: "Mul", lhs: { kind: "Mul", lhs: constF(2), rhs: v(zx), type: Tf32 }, rhs: v(zy), type: Tf32 }, rhs: v(cy), type: Tf32 } } },
      { kind: "Write", target: lvar(zx), value: v(nx) },
      { kind: "Write", target: lvar(zy), value: v(ny) },
      { kind: "Write", target: lvar(iter), value: v(i) },
    ],
  };

  const forLoop: Stmt = {
    kind: "For",
    init: { kind: "Declare", var: i, init: { kind: "Expr", value: constI(0) } },
    cond: { kind: "Lt", lhs: v(i), rhs: constI(MAX_ITER), type: Tbool },
    step: { kind: "Increment", target: lvar(i), prefix: false },
    body: innerLoop,
  };

  const arrType: Type = { kind: "Array", element: Tu32, length: SIZE * SIZE };
  const storageVar: Var = { name: "out_buffer", type: arrType, mutable: true };

  const writeStorage: Stmt = {
    kind: "Write",
    target: {
      kind: "LItem",
      target: { kind: "LVar", var: storageVar, type: arrType },
      index: { kind: "Add", lhs: { kind: "Mul", lhs: v(iy), rhs: constI(SIZE), type: Ti32 }, rhs: v(ix), type: Ti32 },
      type: Tu32,
    },
    value: { kind: "Convert", value: v(iter), type: Tu32 },
  };

  const body: Stmt = {
    kind: "Sequential",
    body: [
      { kind: "Declare", var: ix, init: { kind: "Expr", value: ixExpr } },
      { kind: "Declare", var: iy, init: { kind: "Expr", value: iyExpr } },
      {
        kind: "If",
        cond: {
          kind: "Or",
          lhs: { kind: "Ge", lhs: v(ix), rhs: constI(SIZE), type: Tbool },
          rhs: { kind: "Ge", lhs: v(iy), rhs: constI(SIZE), type: Tbool },
          type: Tbool,
        },
        then: { kind: "Return" },
      },
      { kind: "Declare", var: cx, init: { kind: "Expr", value: cxInit } },
      { kind: "Declare", var: cy, init: { kind: "Expr", value: cyInit } },
      { kind: "Declare", var: zx, init: { kind: "Expr", value: constF(0) } },
      { kind: "Declare", var: zy, init: { kind: "Expr", value: constF(0) } },
      { kind: "Declare", var: iter, init: { kind: "Expr", value: constI(0) } },
      forLoop,
      writeStorage,
    ],
  };

  return {
    types: [],
    values: [
      {
        kind: "StorageBuffer",
        binding: { group: 0, slot: 0 },
        name: "out_buffer",
        layout: arrType,
        access: "read_write",
      },
      {
        kind: "Entry",
        entry: {
          name: "csMain", stage: "compute",
          inputs: [], outputs: [],
          arguments: [gidParam],
          returnType: Tvoid,
          body,
          decorations: [{ kind: "WorkgroupSize", x: 8, y: 8, z: 1 }],
        },
      },
    ],
  };
}

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    log("WebGPU not available; this demo requires WebGPU.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter returned null");
  const device = await adapter.requestDevice();

  const module_ = buildKernelIR();
  const compiled = compileModule(module_, { target: "wgsl" });
  log(compiled.stages.find((s) => s.stage === "compute")!.source);
  log(`storage buffers:`, JSON.stringify(compiled.interface.storageBuffers.map((b) => ({ name: b.name, group: b.group, slot: b.slot, size: b.size }))));

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
