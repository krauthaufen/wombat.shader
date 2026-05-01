# wombat.shader

TypeScript port of FShade. Write shaders as TS arrow functions, get
WGSL/GLSL out via the same IR-level optimiser passes as the F#
original. Two published packages: `@aardworx/wombat.shader` (the
runtime + IR + frontend + emitters + types, all in one tree) and
`@aardworx/wombat.shader-vite` (the build-time inline-marker plugin).

## Repository layout

```
packages/
├── shader/   @aardworx/wombat.shader
│             src/
│             ├── ir/         IR types, visitors, source-map writer
│             ├── frontend/   TS source → IR (uses TS Compiler API)
│             ├── passes/     optimiser passes
│             ├── glsl/       IR → GLSL ES 3.00 emit
│             ├── wgsl/       IR → WGSL emit
│             ├── runtime/    Effect / ComputeShader / compileShaderSource
│             │   └── webgpu/, webgl2/    backend-specific helpers
│             ├── types/      shipped TS declarations users import in
│             │               shader source (V*/M*/sampler/intrinsic)
│             └── index.ts    re-exports the runtime API
└── vite/     @aardworx/wombat.shader-vite
              src/inline.ts   transformInlineShaders() — marker rewrite
              src/index.ts    Vite plugin
              src/typeMapper.ts, typeResolver.ts
```

Originally split into 8 packages (ir / frontend / passes / glsl /
wgsl / runtime / types / vite) — collapsed to 2 in 0.2.0. Internal
imports inside `packages/shader/src/<sub>/` use relative paths
(`../ir/index.js` etc.); the package's subpath exports surface each
subdir.

## Tooling

- `npm test` — vitest, 182 tests covering IR↔WGSL/GLSL round-trips,
  inline-marker plugin pipeline, optimiser passes.
- `npm run typecheck` — `tsc -b --noEmit`.
- `npm run build` — emits `dist/` per package via `tsc -b`.

Real-GPU validation lives downstream in
[`wombat.rendering`](https://github.com/krauthaufen/wombat.rendering)
(Playwright + system Chromium + Vulkan).

## Architecture

```
TS source ──► parseShader ──► IR Module ──► passes ──► emit{wgsl,glsl} ──► CompiledEffect
              (frontend)                  (passes)       (wgsl/glsl)
```

- **frontend** (`parseShader`): walks the TS AST for the named
  entries, lowers expressions/statements to IR. Free identifiers
  resolved via `externalTypes` (storage buffers / uniforms /
  samplers); `b: ComputeBuiltins` parameter recognised for compute
  builtins. Helper-function support via `helpers: string[]`.
- **passes**: `liftReturns` (bare-value → WriteOutput), `inline`,
  `reduceUniforms`, `cse`, `foldConstants`, `dce`, `composeStages`
  (v+v / f+f fuse), `pruneCrossStage`, `inferStorageAccess`,
  `legaliseTypes`, `reverseMatrixOps`, `resolveHoles` (closure-hole
  specialisation).
- **emitters**: WGSL emits per-segment source maps (per-Expr
  granularity for assignments + expression statements). GLSL emits
  line-granular maps. Both produce a `bindings` summary
  (uniformBlocks/storageBuffers/textures/samplers with group/slot).
- **runtime**: `Effect` / `ComputeShader` / `compileShaderSource` /
  `compile({ target })`. `Effect.id` is a build-time stable hash
  (downstream pipeline cache key).

## Markers

`vertex`/`fragment`/`compute` in `packages/shader/src/runtime/stage.ts`
are zero-runtime stubs that throw if called at runtime. The Vite
plugin scans for them and replaces each call with a frozen IR
template:

- `vertex(...)` and `fragment(...)` → `__wombat_stage(template, holes, id, avalHoles)` → returns an `Effect`
- `compute(...)` → `__wombat_compute(...)` → returns a `ComputeShader`
  (single stage, peer of Effect, no graphics pipeline state).

`__wombat_stage` and `__wombat_compute` are the runtime functions
`stage` and `computeShader` in `runtime/stage.ts`.

## Closure holes

Free identifiers in a marker arrow body that aren't intrinsics or
ambient declarations are treated as **closure captures**, encoded as
`ReadInput("Closure", name, type)` in the IR and specialised at
`compile()` time as constants. Different captured values → different
specialised shader. The build plugin emits a getter map per call;
the runtime samples it on each compile.

Aval-typed captures get separate plumbing (`avalHoles`) — the
backend subscribes via the rendering layer's `ProgramInterface` and
writes the GPU buffer slot when the value changes.

## WGSL builtin convention

Compute entry parameters use the **semantic name** (snake_case)
both in the signature and the body — e.g.
`@builtin(global_invocation_id) global_invocation_id: vec3<u32>`,
body reads `global_invocation_id.x`. Without this alignment the
generated WGSL fails to validate (the body uses one name and the
signature declares another). The rendering layer's compute test
catches regressions of this on real GPU.

## Source maps

Per-line, per-segment v3 maps written via
`packages/shader/src/ir/sourcemap.ts`'s `buildSourceMap`. Inputs:

- `lineSegments: ReadonlyArray<readonly { col, span }[]>` (preferred,
  multi-segment).
- `lineSpans: ReadonlyArray<Span | undefined>` (legacy, line-only —
  internally normalised to single-segment lines).

The WGSL Writer (`packages/shader/src/wgsl/emit.ts`) supports
piecewise emit via `write()` / `writeSpan()` / `endLine()` for
sub-line segments, with `setSpan()` / `line()` as the whole-line
shortcut.

## Don'ts

- Don't add new packages. The collapse to 2 was deliberate; subpath
  exports cover the granularity needs.
- Don't break the WGSL builtin-name alignment (signature param name
  must equal the @builtin semantic for compute arguments).
- Don't edit `dist/` directly — it's emitted by `tsc -b`.
- Don't `npm publish` from a dirty tree, and bump both
  `@aardworx/wombat.shader` and `@aardworx/wombat.shader-vite` in
  lockstep (vite's dep is pinned, not a range).
- Don't rely on `Effect.id` being stable across **closure-hole
  values** — closure captures DO move the id. They don't move
  across builds with the same captures, which is what the
  consumer's pipeline cache wants.
