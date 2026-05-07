// `effectFromWgsl` — escape hatch that wraps a pre-built WGSL source
// string into the `Effect`-shaped runtime object. Used by the
// wombat.fable shader plugin's v0 path, before the F#-AST → IR
// translator is wired up: the plugin emits a hardcoded WGSL string
// and we hand back something with a stable `.id` and a working
// `.compile()` that returns the WGSL verbatim.
//
// Limitations (intentional, v0 only):
//   - No IR template, no optimiser passes, no closure holes.
//   - `compile({ target: "glsl" })` throws — only WGSL is carried.
//   - The `interface` is empty; callers that wire bindings must
//     either know the layout out-of-band or not be on this path.
//   - `effect(rawEffect, otherEffect)` won't fuse stages because the
//     IR `template` is a placeholder Module with no Entry values.
//
// The "right" path is the IR-based `stage()` factory; this exists
// purely so the plugin plumbing can be proven end-to-end before the
// translator lands.
//
// The single WGSL source is expected to contain both `@vertex` and
// `@fragment` entries named `vs` and `fs`, matching the passthrough
// shape the plugin currently emits. Future v0.x can take an explicit
// `{ vertex, fragment }` pair.

import { hashValue, type Module } from "../ir/index.js";
import type { CompileOptions, CompiledEffect, CompiledStage, Target } from "./compile.js";
import type { Effect, Stage } from "./stage.js";
import type { ProgramInterface } from "./interface.js";

const EMPTY_MODULE: Module = { types: [], values: [] };

const EMPTY_INTERFACE = (target: Target): ProgramInterface => ({
  target,
  stages: [],
  attributes: [],
  fragmentOutputs: [],
  uniforms: [],
  uniformBlocks: [],
  samplers: [],
  textures: [],
  storageBuffers: [],
});

export interface RawWgslOptions {
  /** Vertex entry point name (default `"vs"`). */
  readonly vertexEntry?: string;
  /** Fragment entry point name (default `"fs"`). */
  readonly fragmentEntry?: string;
}

export function effectFromWgsl(wgsl: string, options: RawWgslOptions = {}): Effect {
  const id = hashValue({ raw: "wgsl", wgsl, opts: options });
  const placeholderStage: Stage = {
    template: EMPTY_MODULE,
    holes: {},
    avalHoles: {},
    id,
  };
  const stages: readonly Stage[] = [placeholderStage];
  return {
    stages,
    id,
    dumpIR() {
      return `// raw WGSL effect (id=${id})\n${wgsl}`;
    },
    compile(options: CompileOptions): CompiledEffect {
      if (options.target !== "wgsl") {
        throw new Error(
          `effectFromWgsl: only target="wgsl" is supported (got "${options.target}").`,
        );
      }
      const compiledStage: CompiledStage = {
        // Cast — there's no real IR stage to point at. The WGSL source
        // is what the runtime ultimately needs.
        stage: "vertex" as unknown as CompiledStage["stage"],
        entryName: options.fragmentOutputLayout ? "" : (options as RawWgslOptions).vertexEntry ?? "vs",
        source: wgsl,
        bindings: {
          inputs: [], outputs: [], uniforms: [], storage: [], textures: [], samplers: [],
        } as unknown as CompiledStage["bindings"],
        meta: {} as CompiledStage["meta"],
        sourceMap: null,
      };
      return {
        target: "wgsl",
        stages: [compiledStage],
        interface: EMPTY_INTERFACE("wgsl"),
        avalBindings: {},
      };
    },
  };
}
