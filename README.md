# @aardworx/wombat.shader

A TypeScript port of [FShade](https://fshade.org) — write shaders as
TypeScript functions, get WGSL or GLSL out, with the same IR-level
optimisation passes (cross-stage I/O elimination, function inlining,
constant folding, CSE, DCE) and the same composition story (effects
match outputs to inputs by semantic and fuse at IR level).

Part of the Wombat TypeScript port of the Aardvark stack:

1. [`@aardworx/wombat.adaptive`](https://github.com/krauthaufen/wombat.adaptive) — incremental adaptive computations (`aval`/`aset`/`alist`/`amap`).
2. [`@aardworx/wombat.base`](https://github.com/krauthaufen/wombat.base) — math/geometry primitives (`V*f` / `M*f` / `Trafo3d` / …).
3. **`@aardworx/wombat.shader`** — this repo: TS-as-shader DSL with composition + optimiser passes + WGSL / GLSL emit.
4. [`@aardworx/wombat.rendering`](https://github.com/krauthaufen/wombat.rendering) — RenderObject + RenderTask + window on top of WebGPU and wombat.shader.

## Install

```bash
npm install @aardworx/wombat.shader
# build-time plugin (Vite):
npm install -D @aardworx/wombat.shader-vite
```

ESM only. Node ≥ 18, modern bundlers.

## Quick start

Write a shader as TypeScript and let the build plugin compile it:

```ts
import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";

const helloTriangle = effect(
  vertex<{ a_position: V2f; a_color: V3f }>(input => ({
    gl_Position: new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
    v_color: input.a_color,
  })),
  fragment<{ v_color: V3f }>(input =>
    new V4f(input.v_color.x, input.v_color.y, input.v_color.z, 1.0),
  ),
);

const compiled = helloTriangle.compile({ target: "wgsl" });
console.log(compiled.stages.find(s => s.stage === "vertex")!.source);
```

Run with the Vite plugin (`@aardworx/wombat.shader-vite`) installed —
it scans for `vertex(...)` / `fragment(...)` / `compute(...)` calls
and replaces each with a frozen IR template at build time. Without
the plugin, the markers throw at runtime.

## Compute

```ts
import { compute } from "@aardworx/wombat.shader";
import type { ComputeBuiltins } from "@aardworx/wombat.shader/types";

declare const out: Storage<u32[]>;

const cs = compute((b: ComputeBuiltins) => {
  out[b.globalInvocationId.x as i32] = b.globalInvocationId.x;
});
```

`compute(...)` returns a `ComputeShader` (peer of `Effect` — single
stage, no graphics pipeline state).

## Module map

| Subpath | Contents |
| --- | --- |
| `@aardworx/wombat.shader` | runtime API: `vertex`/`fragment`/`compute` markers, `effect`, `compileShaderSource`, `Effect`/`ComputeShader`/`CompiledEffect`/`ProgramInterface` |
| `@aardworx/wombat.shader/ir` | IR types (`Module`, `EntryDef`, `Type`, `Expr`, …), visitors, `buildSourceMap` |
| `@aardworx/wombat.shader/frontend` | TS source → IR walker (`parseShader`) |
| `@aardworx/wombat.shader/passes` | optimiser passes (DCE / fold / inline / `liftReturns` / cross-stage prune) |
| `@aardworx/wombat.shader/glsl` | IR → GLSL ES 3.00 emitter (WebGL2) |
| `@aardworx/wombat.shader/wgsl` | IR → WGSL emitter (WebGPU) |
| `@aardworx/wombat.shader/types` | shipped TS declarations for shader source: `V*`/`M*`/sampler types/intrinsics/`ComputeBuiltins` |
| `@aardworx/wombat.shader/webgpu` | `createShaderModules` helper |
| `@aardworx/wombat.shader/webgl2` | `linkEffect` (program linking) helper |

## Pipeline

```
TS source ──► parseShader ──► IR Module ──► passes ──► emit{wgsl,glsl} ──► CompiledEffect
              (frontend)                  (passes)        (wgsl/glsl)
```

`compileShaderSource(src, entries, opts)` runs the whole chain;
`Effect.compile({ target })` does the same plus closure-hole
specialisation (build-time captures inlined as constants per call).

## Composition

`effect(v, f, …)` flattens stage lists; the optimiser's
`composeStages` pass fuses `vertex + vertex` and `fragment + fragment`
into single stages and connects `vertex → fragment` at compile time.
`Effect.id` is a stable build-time hash so consumers (the rendering
layer's pipeline cache) can key off `(effectId, signature)`.

## What's optimised

- `liftReturns` — bare-value returns lift to `WriteOutput` for the
  declared output (handles `gl_Position` / `outColor` builtins).
- `inferStorageAccess` — flips storage-buffer access from `read` to
  `read_write` when the body assigns into it.
- `composeStages` — fuses adjacent same-stage effects (vertex+vertex /
  fragment+fragment).
- `pruneCrossStage` — drops vertex outputs the fragment doesn't read.
- `inline`, `reduceUniforms`, `cse`, `foldConstants`, `dce` — standard
  passes; ordered by `compileModule`.
- `legaliseTypes` — rewrites IR types that don't translate to the
  target (e.g. atomics → `atomic<u32>`).
- `reverseMatrixOps` — flips matrix product order to match WGSL/GLSL
  column-major convention against incoming row-major ASTs.

## Source maps

Every emitted stage carries a v3 source map. The WGSL emitter ships
per-segment maps (sub-line granularity for assignments and
expression statements); GLSL ships line-granular. Decode to TS
`(file, line, col)` via wombat.rendering's `decodePosition`.

## Build & test

```bash
npm install
npm run typecheck
npm test
npm run build
```

182 tests covering IR ↔ WGSL/GLSL round-trips, the inline-marker
plugin pipeline, and the optimiser passes.

## License

MIT.
