# wombat.shader

A TypeScript port of [FShade](https://fshade.org) — write shaders as
TypeScript functions, get GLSL/WGSL out, with the same IR-level
optimisation passes (cross-stage I/O elimination, function inlining,
constant folding, CSE, DCE) and the same composition story (effects
match outputs to inputs by semantic and fuse at IR level).

Shipped at `0.1.0`. Part of a port of the Aardvark stack to
TypeScript:

1. [`@aardworx/wombat.adaptive`](https://github.com/krauthaufen/wombat.adaptive)
   — incremental adaptive computations (`aval`/`aset`/`alist`/`amap`),
   port of `FSharp.Data.Adaptive`. ✓
2. [`@aardworx/wombat.base`](https://github.com/krauthaufen/wombat.base)
   — math/geometry primitives (`V*f` / `M*f` / `Trafo3d` / …). ✓
3. **`@aardworx/wombat.shader-*`** — this repo: TS-as-shader DSL with
   composition + optimiser passes + WGSL / GLSL emit. ✓
4. (in progress) `wombat.rendering` — RenderObject + RenderTask +
   Window on top of WebGPU.

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
                  • Effect.compile({ target }) — cached by hole-values
                  • CompiledEffect carries a full ProgramInterface
                  • avalBindings expose aval-driven uniform sources
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

## IR design

Documented in [`docs/IR.md`](docs/IR.md). Modelled after FShade's
`CType` / `CExpr` / `CLExpr` / `CStatement` / `CValueDef` /
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

## Status

`0.1.0` shipped on npm — every package below is published public:

- [x] `@aardworx/wombat.shader-ir` — node types, type checker, IR
      pretty-printer, JSON (de)serialisation, source-map building.
- [x] `@aardworx/wombat.shader-passes` — DCE, constant folding,
      inlining, CSE, cross-stage prune, uniform reduce, lift-returns,
      type legalisation, matrix-reversal, infer-storage-access,
      resolve-holes.
- [x] `@aardworx/wombat.shader-glsl` — IR → GLSL ES 3.00.
- [x] `@aardworx/wombat.shader-wgsl` — IR → WGSL.
- [x] `@aardworx/wombat.shader-frontend` — TypeScript Compiler API
      walker; arrow body → IR. Operators, ternaries, control flow,
      helper-function calls, sampler / storage-buffer captures.
- [x] `@aardworx/wombat.shader-runtime` — `compileShaderSource`,
      `effect()` / `stage()` / `vertex()` / `fragment()` / `compute()`,
      `CompiledEffect` with full `ProgramInterface`, hole-value
      cache.
- [x] `@aardworx/wombat.shader-types` — shipped `.d.ts` for
      `V*`/`M*`/`sampler*`/intrinsics.
- [x] `@aardworx/wombat.shader-vite` — build-time inline-marker
      transform with TS type-checker contextual inference; `*.shader.ts`
      module form for fully precompiled shaders.
- [x] Examples: `hello-triangle`, `hello-triangle-webgpu`,
      `textured-quad`, `composition`, `instanced-cubes`,
      `compute-readback`. `wombat.rendering`'s `hello-triangle`
      example is the first downstream consumer.

Tests: 182 across 33 files; no untracked stub files.

## Math types & operators

Shader source can — and should — use the same `V*f` / `M*f` /
`V*i` / `V*ui` classes as the surrounding CPU code. The shipped
type re-exports of `@aardworx/wombat.base` give:

- Real runtime classes (`new V3f(1, 2, 3)` works on the CPU; the
  same call is recognised by the shader frontend and lowered to a
  `NewVector` IR node).
- Methods that match GLSL/WGSL semantics: `add` / `sub` / `mul` /
  `div` / `dot` / `cross` / `length` / `normalize` / `transpose` /
  `inverse` / `lerp` / `clamp` / `min` / `max` / `lengthSquared` /
  `distance` / etc.
- A `__aardworxMathBrand` field that
  [`boperators`](https://www.npmjs.com/package/boperators)
  recognises so users can write `+` / `-` / `*` / `/` instead of
  method calls.

To turn on operator syntax in app + shader code:

```jsonc
// tsconfig.json
{
  "compilerOptions": {
    "plugins": [
      { "transform": "@boperators/plugin-tsc", "transformProgram": true },
      { "name": "@boperators/plugin-ts-language-server" }
    ]
  }
}
```

```ts
// vite.config.ts
import { defineConfig } from "vite";
import { boperatorsVite } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

export default defineConfig({
  plugins: [
    boperatorsVite(),  // first — rewrites `a + b` to `a.add(b)`
    wombatShader(),    // second — sees `.add()` and translates to IR
  ],
});
```

Add the dev dependencies:

```
npm i -D boperators @boperators/plugin-tsc \
        @boperators/plugin-ts-language-server \
        @boperators/plugin-vite
```

After this, shader bodies look like:

```ts
import { V3f, M44f } from "@aardworx/wombat.shader-types";

declare const u: { mvp: M44f };

const fx = effect(
  vertex((v: { a_position: V3f }) => ({
    gl_Position: u.mvp * new V4f(v.a_position.x, v.a_position.y, v.a_position.z, 1),
  })),
  fragment((input: { v_color: V3f }) => ({
    outColor: new V4f(
      (input.v_color * 0.5).x, input.v_color.y, input.v_color.z, 1,
    ),
  })),
);
```

The IDE provides autocomplete on `V3f` methods, the LSP type-checks
operator overloads, and the shader plugin emits the right WGSL/GLSL
either way (with or without `boperators`).

## License

MIT, mirroring the rest of the stack.
