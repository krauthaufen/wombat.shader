# wombat.shader

A TypeScript port of [FShade](https://fshade.org) — write shaders as
TypeScript functions, get GLSL/WGSL out, with the same IR-level
optimisation passes (cross-stage I/O elimination, function inlining,
constant folding, CSE, DCE) and the same composition story (effects
match outputs to inputs by semantic and fuse at IR level).

This is the third leg of a three-part journey to bring
`Aardvark.Dom`'s feature set to TypeScript:

1. [`@aardworx/adaptive`](https://www.npmjs.com/package/@aardworx/adaptive)
   — incremental adaptive computations (`aval`/`aset`/`alist`/`amap`),
   port of `FSharp.Data.Adaptive`. ✓
2. [`@aardworx/adaptive-ui`](https://www.npmjs.com/package/@aardworx/adaptive-ui)
   — direct-DOM JSX runtime where adaptive values, lists, and maps sit
   in the same JSX positions as plain values. ✓
3. **`@aardworx/wombat.shader`** — what this repo will become. Lets adaptive
   collections drive WebGL2 / WebGPU pipelines through real shader
   composition with a working optimiser.

## Scope

| In scope | Out of scope |
| --- | --- |
| vertex, fragment, compute stages | tessellation, geometry, raytracing |
| WebGL2 (GLSL ES 3.00) and WebGPU (WGSL) backends | desktop GLSL 4.x, SPIR-V (no Web target) |
| Effect composition by semantic — fuse v+v / f+f, pipeline v→f | dynamic dispatch, function pointers |
| Compute kernels with workgroup memory, atomics, barriers | multi-queue scheduling |
| Cross-stage input pruning + uniform reduction (the load-bearing pass) | runtime shader hot-reload (later) |
| Inlining, constant folding, CSE, DCE | platform-specific optimisations |
| TypeScript subset translated to IR via TS compiler API | `eval`, dynamic require, classes outside the IR types |
| Vite plugin + SWC plugin | hand-rolled bundler integration |

Stage matrix is reduced from FShade's full set because the Web has no
tessellation, geometry, or raytracing exposed. The composition rules
collapse to three:

- `vertex + vertex → vertex` (sequential fuse)
- `fragment + fragment → fragment` (sequential fuse)
- `vertex + fragment → vertex+fragment` (pipeline)
- `compute` stands alone (composed at the dispatch level, not IR level)

## Architecture

```
┌─────────────────────────┐
│  user .tsx files        │   <Pipeline> uses fragment(...) / vertex(...)
└────────────┬────────────┘
             │
             │  build-time (Vite/SWC plugin)
             ▼
┌─────────────────────────┐
│  TS AST                 │   tsc parses the file as normal
└────────────┬────────────┘
             │
             │  @aardworx/wombat.shader-frontend uses TypeScript Compiler API:
             │   walks the arrow-function body, type-checks operands
             │   against shipped .d.ts (V2/V3/V4/M2/M3/M4/sampler*),
             │   emits IR
             ▼
┌─────────────────────────┐       ┌──────────────────────────────┐
│  wombat.shader IR              │ ◄──►  │  @aardworx/wombat.shader-passes               │
│  (Type, Expr, LExpr,    │       │   inline / fold / CSE / DCE  │
│   Stmt, Module — see    │       │   compose / cross-stage prune│
│   docs/IR.md)           │       │   uniform-reduce             │
└────────────┬────────────┘       └──────────────────────────────┘
             │
             │  emit
             ▼
   ┌─────────────────┐    ┌─────────────────┐
   │  @aardworx/wombat.shader-glsl    │    │  @aardworx/wombat.shader-wgsl    │
   │  (GLSL ES 3.00) │    │  (WGSL)         │
   └────────┬────────┘    └────────┬────────┘
            │                      │
            ▼                      ▼
      WebGL2 program        WebGPU pipeline
            │                      │
            └──── @aardworx/wombat.shader-runtime ──┘
                  • Effect / ComputeShader / sampler scope
                  • aval-driven uniforms, alist-driven VBOs
                  • UIScheduler integration with adaptive-ui
```

The IR is the contract. Frontend produces it, passes operate on it,
emitters consume it. Everything else is plumbing.

## Package layout

Phase 0 in place; later phases scaffolded but not yet present.

```
wombat.shader/                   (npm workspaces; tsc project references)
├─ packages/
│  ├─ ir/         @aardworx/wombat.shader-ir         IR types + visitors + JSON
│  ├─ passes/     @aardworx/wombat.shader-passes     fold/dce/cse/inline/compose/prune/lift/legalise
│  ├─ glsl/       @aardworx/wombat.shader-glsl       IR → GLSL ES 3.00
│  ├─ wgsl/       @aardworx/wombat.shader-wgsl       IR → WGSL
│  ├─ types/      @aardworx/wombat.shader-types      shipped .d.ts: V*/M*/sampler*/intrinsics
│  ├─ frontend/   @aardworx/wombat.shader-frontend   TS source → IR (TS Compiler API)
│  ├─ runtime/    @aardworx/wombat.shader-runtime    compileShaderSource + WebGL2 / WebGPU
│  └─ vite/       @aardworx/wombat.shader-vite       build-time shader compilation
└─ examples/
   ├─ hello-triangle             WebGL2 — simplest case
   ├─ hello-triangle-webgpu      WebGPU mirror
   ├─ textured-quad              WebGL2 + WebGPU side-by-side, sampler + UBO
   ├─ composition                composeStages fusing two fragments
   ├─ instanced-cubes            WebGPU — Camera UBO + instance buffer + depth
   └─ compute-readback           Compute kernel + storage buffer + mapAsync readback
```

`@aardworx/wombat.shader-ir` and `@aardworx/wombat.shader-passes` are pure — no DOM, no toolchain.
Everything is unit-testable from Node. Frontend is the only piece
that imports `typescript`. Emitters take IR in and produce strings;
they have no platform dependencies either.

## Concrete low-level plan

The IR is documented in [`docs/IR.md`](docs/IR.md). Modelled after
FShade's `CType` / `CExpr` / `CLExpr` / `CStatement` / `CValueDef` /
`CModule` with the following deliberate departures from the F#
original:

- **No raytracing nodes** — the Web has no surface for them.
- **No `byref` / `out` parameter modifiers and no `CAddressOf` /
  `CLPtr`** — neither WebGL2 GLSL nor WGSL exposes pointers in a
  user-visible way. Functions are inline-only or pass-by-value.
- **No `CColor` distinct from `CVector`** — a colour is just a
  `Vector(Float, n)` with a semantic decoration on the parameter.
- **Aardvark naming conventions for shipped types.** The internal
  IR stays structural (`Vector(Float, 2)`), but `@aardworx/wombat.shader-types` ships
  `V2i`/`V3i`/`V4i`, `V2u`/`V3u`/`V4u`, `V2f`/`V3f`/`V4f`,
  `V2b`/`V3b`/`V4b`, and the rectangular matrices `M22f`/`M33f`/
  `M44f`/`M23f`/`M24f`/`M32f`/`M34f`/`M42f`/`M43f`. These are what
  users see in their TS code; the frontend translates them to the
  structural IR types.
- **No `float16` / `float64` / `decimal`** — Web shaders have no
  surface for them. `Float(32)` is the only `Float` width; `Half` is
  a precision qualifier on the parameter, not a separate type.
- **WGSL address-spaces are explicit** — `Type` distinguishes
  `uniform`, `storage`, `workgroup`, `private`, `function` for buffer
  and variable nodes (GLSL ignores the distinction, WGSL requires it).
- **Side-effect tag on every intrinsic and call node** — DCE depends
  on this. `texture()` is pure; `imageStore` is not.
- **Stage tag on `EntryDef`** — `vertex | fragment | compute`. No
  legacy stages.
- **JSON-serialisable IR** — discriminated unions encoded as
  `{ kind: "..." , ... }` so build-time-emitted IR can be persisted
  and runtime can compose without re-parsing TS.

## Roadmap

- **Phase 0 — IR and passes (~6 weeks)**
  - [ ] `@aardworx/wombat.shader-ir`: every node type, type checker, pretty-printer
        for IR-level debug output, JSON (de)serialisation
  - [ ] `@aardworx/wombat.shader-passes`: DCE, constant folding, inlining, CSE, the
        cross-stage pruning pass — all driven by hand-written IR
        programs in tests
- **Phase 1 — emitters and runtime (~3 weeks)**
  - [ ] `@aardworx/wombat.shader-glsl`: round-trip the test IR programs to GLSL ES
        3.00, link in WebGL2
  - [ ] `@aardworx/wombat.shader-wgsl`: same for WGSL
  - [ ] `@aardworx/wombat.shader-runtime`: `Effect`, `ComputeShader`, samplers,
        adaptive uniform binding, alist-driven VBO
- **Phase 2 — frontend (~4 weeks)**
  - [ ] `@aardworx/wombat.shader-types`: V2/V3/V4/M2-4/sampler\* declarations
  - [ ] `@aardworx/wombat.shader-frontend`: TS arrow-function body → IR via the
        TypeScript compiler API. Operators (`+`/`-`/`*`/`/`/`%`/
        comparisons) and swizzle property access translated; `if`,
        `for`, `while`, ternaries; calls to other shader functions
        inlined or linked; intrinsic table for built-ins
  - [ ] Source-map fidelity: every IR node carries the originating
        TS span; emitter forwards it to GLSL/WGSL line maps
- **Phase 3 — toolchain (~2 weeks)**
  - [ ] `@aardworx/wombat.shader-vite`: detect `vertex(...)` / `fragment(...)` /
        `compute(...)` calls, run frontend, replace call with
        compiled handle + serialised IR
  - [ ] `@aardworx/wombat.shader-swc`: same for non-Vite stacks
- **Phase 4 — examples and demos**
  - [ ] hello-triangle (statically composed effect)
  - [ ] instanced-cubes (alist-driven instance buffer)
  - [ ] compute-particles (compute → vertex/fragment, all adaptive)

Total realistic budget: 3–4 months full-time for a usable v0.1.

## License

MIT, mirroring the rest of the stack.
