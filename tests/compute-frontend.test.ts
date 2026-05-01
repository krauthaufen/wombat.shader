// Compute-stage frontend coverage:
//   - `ComputeBuiltins` parameter → ReadInput("Builtin") + entry.arguments
//   - `@workgroupSize` JSDoc → WorkgroupSize decoration
//   - Storage-buffer access mode is inferred from usage
//   - Atomic intrinsics promote the buffer's element type to AtomicU32

import { describe, expect, it } from "vitest";
import { compileShaderSource } from "@aardworx/wombat.shader";
import type { ValueDef } from "@aardworx/wombat.shader/ir";

function storageBuf(name: string, length: number): ValueDef {
  return {
    kind: "StorageBuffer",
    binding: { group: 0, slot: 0 },
    name,
    layout: {
      kind: "Array",
      element: { kind: "Int", signed: false, width: 32 },
      length,
    },
    access: "read", // overridden by inferStorageAccess
  };
}

describe("compute frontend", () => {
  it("ComputeBuiltins param + workgroupSize JSDoc + read_write inference", () => {
    const source = `
      declare const out_buffer: number[];

      /** @workgroupSize 8 8 1 */
      export function csMain(b: ComputeBuiltins): void {
        const i = b.globalInvocationId.x as i32;
        out_buffer[i] = (i as u32);
      }
    `;
    const compiled = compileShaderSource(
      source,
      [{ name: "csMain", stage: "compute" }],
      { target: "wgsl", extraValues: [storageBuf("out_buffer", 64)] },
    );
    const wgsl = compiled.stages.find((s) => s.stage === "compute")!.source;
    expect(wgsl).toContain("@compute @workgroup_size(8, 8, 1)");
    expect(wgsl).toContain("@builtin(global_invocation_id)");
    expect(wgsl).toContain("read_write");
    expect(wgsl).not.toContain("var<storage, read>");
  });

  it("read-only buffer stays read", () => {
    const source = `
      declare const in_buffer: number[];
      declare const sink: number[];

      /** @workgroupSize 1 */
      export function csMain(b: ComputeBuiltins): void {
        const i = b.globalInvocationId.x as i32;
        sink[i] = in_buffer[i];
      }
    `;
    const compiled = compileShaderSource(
      source,
      [{ name: "csMain", stage: "compute" }],
      {
        target: "wgsl",
        extraValues: [
          { ...storageBuf("in_buffer", 64), access: "read" } as ValueDef,
          { ...storageBuf("sink", 64), binding: { group: 0, slot: 1 } } as ValueDef,
        ],
      },
    );
    const wgsl = compiled.stages.find((s) => s.stage === "compute")!.source;
    // in_buffer untouched (read), sink written (read_write)
    expect(wgsl).toMatch(/var<storage,\s*read>\s+in_buffer/);
    expect(wgsl).toMatch(/var<storage,\s*read_write>\s+sink/);
  });

  it("atomicAdd promotes element type to atomic<u32> and emits &", () => {
    const source = `
      declare const counter: number[];

      /** @workgroupSize 64 */
      export function csMain(b: ComputeBuiltins): void {
        const i = b.globalInvocationId.x as i32;
        atomicAdd(counter[0], 1 as u32);
        // intentionally ignore i to exercise unused builtin path
      }
    `;
    const compiled = compileShaderSource(
      source,
      [{ name: "csMain", stage: "compute" }],
      { target: "wgsl", extraValues: [storageBuf("counter", 1)] },
    );
    const wgsl = compiled.stages.find((s) => s.stage === "compute")!.source;
    expect(wgsl).toContain("atomic<u32>");
    expect(wgsl).toContain("atomicAdd(&counter");
    expect(wgsl).toContain("read_write");
    // Storage interface should also reflect the new layout.
    const sb = compiled.interface.storageBuffers.find((b) => b.name === "counter");
    expect(sb?.access).toBe("read_write");
  });

  it("workgroupBarrier lowers to a Barrier statement", () => {
    const source = `
      declare const buf: number[];

      /** @workgroupSize 64 */
      export function csMain(b: ComputeBuiltins): void {
        const i = b.localInvocationIndex as i32;
        buf[i] = (i as u32);
        workgroupBarrier();
        buf[i] = buf[i] + (1 as u32);
      }
    `;
    const compiled = compileShaderSource(
      source,
      [{ name: "csMain", stage: "compute" }],
      { target: "wgsl", extraValues: [storageBuf("buf", 64)] },
    );
    const wgsl = compiled.stages.find((s) => s.stage === "compute")!.source;
    expect(wgsl).toContain("workgroupBarrier();");
  });

  it("GLSL target rejects atomic intrinsic with a clear error", () => {
    const source = `
      declare const counter: number[];

      /** @workgroupSize 1 */
      export function csMain(b: ComputeBuiltins): void {
        atomicAdd(counter[0], 1 as u32);
      }
    `;
    expect(() =>
      compileShaderSource(
        source,
        [{ name: "csMain", stage: "compute" }],
        { target: "glsl", extraValues: [storageBuf("counter", 1)] },
      )
    ).toThrow(/atomic.*not supported|GLSL ES 3\.00/);
  });
});
