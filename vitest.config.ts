import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // The native WebGPU module (webgpu/linux-x64.dawn.node) throws from a
    // pending callback during teardown and `napi_fatal_error` aborts the
    // whole process — with the default `threads` pool that is the vitest
    // process itself, so the run dies without a summary about every second
    // time. A fork per test file contains the abort to a file that has
    // already finished; the run is stable.
    pool: "forks",
  },
});
