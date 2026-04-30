// Both backends running in parallel: the same TS shader source is
// compiled to GLSL ES 3.00 (WebGL2) and to WGSL (WebGPU), then
// drawn into two side-by-side canvases.
//
// The point of this example: show that the runtime's
// `compileShaderSource` plus the `ProgramInterface` it returns is
// enough to drive both back-ends with no string-keyed lookups in the
// app code.

import { runWebGL2 } from "./gl.js";
import { runWebGPU } from "./wgpu.js";
import { log } from "./common.js";

try {
  runWebGL2(document.getElementById("gl") as HTMLCanvasElement);
} catch (e) {
  log("gl-log", "ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
}

runWebGPU(document.getElementById("wgpu") as HTMLCanvasElement).catch((e) => {
  log("wgpu-log", "ERROR:", e instanceof Error ? e.message : String(e));
  console.error(e);
});
