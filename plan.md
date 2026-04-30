# wombat.shader — FShade-parity plan

This is the work that gets us from "compiles a shader" to "ergonomic
enough to write real renderers in." Scoped to the inline-shader path
(`vertex/fragment/compute(arrow|named)` markers) since that's the
user-facing API. Adaptive (`aval`-driven) uniform binding is
explicitly **out of scope for now** — uniforms stay statically typed
and bound by the surrounding draw layer.

Phases are ordered by dependency, not by size. Within a phase the
tasks are roughly in implementation order.

---

## Phase 1 — Shader helper functions ✅ DONE

**Goal.** Let user code write helper functions and call them from
shader bodies. Today the frontend emits `Call(stub)` for any
unknown function, which means anything past trivial demos has no
factoring path.

```ts
function blendColors(a: V3f, b: V3f, t: number): V3f {
  return mix(a, b, t);
}

const fs = fragment(input => ({
  outColor: vec4(blendColors(u.tintA, u.tintB, time), 1.0),
}));
```

**What needs doing.**

1. **Plugin: helper resolution.** When walking a marker arrow body
   and encountering a free identifier whose symbol resolves (via the
   type checker) to a top-level `function` declaration or
   `const f = (…) => …` arrow, treat it as a shader helper.
   - Pull its source via `ts.SymbolDisplayPart` / declaration AST.
   - Walk its body recursively the same way as the marker arrow,
     classifying *its* free identifiers (uniforms, captures of *its*
     scope — same machinery).
   - Produce a `Function` ValueDef and add it to the module's `values`.
   - Replace each call site in the parent body with a `Call(fn)` that
     references the new function by signature/id.

2. **Cross-helper deduplication.** If two stages call the same
   helper, emit it once. Key: helper name + closure-resolved type
   parameters. The plugin maintains a per-build helper cache.

3. **Recursion check.** Shader functions can't recurse (no shader
   stage permits it). Detect cycles during resolution and error
   clearly.

4. **`@inline` / `@noinline` JSDoc tags.** Tag a helper with one of
   these to override `inlinePass`'s default policy. Maps to the
   existing `FnAttr` set.

5. **Tests.**
   - Helper called from one stage → emitted once, called.
   - Same helper called from vertex + fragment → still emitted once,
     after `composeStages` runs.
   - Recursive helper → clear error pointing at the call site.
   - Helper that itself uses a uniform / closure capture → binding
     flows through to the parent template's `Uniform` decl.

**Acceptance.** A non-trivial shader that factors out 3–4 helper
functions compiles cleanly and `inlinePass` flattens trivial ones.

---

## Phase 2 — Build-time stable IDs ✅ + Shader cache ✅ DONE

**Build-time IDs (done).** Each Effect/Stage carries a build-time
stable id (FNV-1a 64-bit hex of the IR template). Identical IR →
identical id; `effect(v, f).id = combineHashes(v.id, f.id)`. The
plugin emits the id as `__wombat_stage(template, holes, "<id>")`'s
third argument. Backends key emitted GLSL/WGSL by id.

**Cache (still TODO).** `Effect.compile()` re-emits GLSL/WGSL on
every call. With closure-specialization in play the same Effect
compiled with different closure values produces different shaders —
seen-before combinations should hit a cache.

**What needs doing.**

1. **Per-Effect cache.** `Map<cacheKey, CompiledEffect>` populated
   lazily. `compile()`:
   - Resolve holes from getters → Module.
   - Compute `cacheKey = effect.id + ":" + hashValue(holeValues) + ":" + target`.
   - Cache hit → return the existing CompiledEffect.
   - Cache miss → run the pipeline + emit, store, return.

2. **Cache size bound.** Default to unbounded (closures producing
   genuinely distinct shaders is the user's choice). Option
   `maxShaderCacheSize` for paranoid setups; LRU eviction past it.

3. **Tests.**
   - Same closure values across two `compile()` calls → same
     `CompiledEffect` reference (cache hit).
   - Different closure values → distinct CompiledEffects.
   - Different target → distinct cache entries (same id but
     different `target` should not collide).

**Acceptance.** A render loop that calls `compile()` every frame
with stable closure values takes O(1) shader work after the first
frame.

---

## Phase 3 — `aval`-typed captures become uniforms; values specialize ✅ DONE

**Goal.** The boundary between specialization and runtime binding is
the *type* of the captured value, not an explicit marker:

- Capture is an **`aval<T>`** (adaptive value) → uniform binding.
  The shader IR carries a `ReadInput("Uniform", name, T)`; the Effect
  retains the aval handle so the future rendering backend can
  subscribe to it and rewrite the GPU buffer when it changes.
- Capture is a plain **`T`** (or any non-`aval` value) → specialize.
  The capture lowers to a `Const` / `NewVector` / `MatrixFromCols`
  inlined into the IR (today's behavior).

This makes the user's intent clear at the call site: if it changes,
wrap in `aval()`; if it doesn't, capture directly. No special
runtime marker needed.

```ts
const tint: V3f = new V3f(1, 0, 0);          // doesn't change → specialize
const mvp: aval<M44f> = aval(...);           // animates → uniform binding

const fx = effect(
  vertex(v => ({ gl_Position: mvp.mul(vec4(v.a_position, 1.0)) })),
  fragment(input => ({ outColor: vec4(tint.x, tint.y, tint.z, 1.0) })),
);
```

**What needs doing.**

1. **Type recognition.** Extend `typeMapper.tsTypeToIR` to recognise
   `aval<T>` (from `@aardworx/adaptive`) by symbol name and return
   the *inner* IR type. The plugin classifies aval-typed captures as
   uniforms; everything else stays on the existing path.

2. **Plugin: split capture sets.** Today the plugin emits one map
   `holes: { name: () => value }` for every closure capture.
   Replace with two:
   - `valueHoles: { name: () => value }` — drives `resolveHoles`
     (inline as constant). This stays the existing path.
   - `avalHoles: { name: () => aval<T> }` — opaquely retained on
     the Effect for the future backend; the IR does NOT inline
     these, instead emits a `Uniform` decl + `ReadInput("Uniform")`.

3. **Effect carries avals.** Extend `Effect` (and `Stage`) to hold
   an `avalHoles` field alongside `holes`. The runtime never reads
   the avals' values — it just stores them and exposes them on the
   compiled effect:
   ```ts
   interface CompiledEffect {
     // existing: target, stages, interface
     readonly avalBindings: Record<string, () => unknown>;
   }
   ```
   The future rendering backend consults `avalBindings` keyed by
   uniform name to wire each aval to its GPU buffer slot.

4. **No constant inlining for aval captures.** `resolveHoles` only
   substitutes for names present in its value map. Aval-bound names
   stay as `ReadInput("Uniform", …)` and survive into the emitted
   shader source.

5. **Tests.**
   - `aval<V3f>` capture → Uniform decl, no closure hole, no inlined
     value in WGSL.
   - Plain `V3f` capture next to it → constant inlined.
   - Compiled output exposes `avalBindings` keyed by capture name.

**Acceptance.** Users naturally express "specialize me" vs "bind
me" by choice of type; switching one to the other is a one-line
change at the declaration site, no shader-side syntax.

---

## Phase 4 — Sampler & texture captures ✅ DONE

Includes the post-Phase audit: `Sampler2DShadow` emits `texture_depth_2d`
in WGSL; `Sampler2DMS`/`ISampler2DMS`/`USampler2DMS` emit
`texture_multisampled_*` (no companion sampler binding); GLSL throws
on multisample. Branded scalar aliases (`i32`/`u32`/`f32`) shipped in
`@aardworx/wombat.shader-types` so users can declare typed scalar
uniforms — plain `number` still defaults to `f32`.

**Goal.** Today only constants and (after Phase 3) uniforms can be
captured. Sampler captures should produce a **sampler binding**, not
a const-inlined value (you can't constant-fold a sampler).

```ts
const tex: Sampler2D = ...;  // somewhere — runtime sampler handle
const fs = fragment(input => ({
  outColor: texture(tex, input.v_uv),
}));
```

**What needs doing.**

1. **Plugin classification.** When a closure capture's TS type
   resolves to `Sampler2D` / `Sampler3D` / `SamplerCube` / etc.,
   route it to the sampler path instead of the closure-constant
   path. Emit a `Sampler` ValueDef + a `ReadInput("Uniform", name,
   samplerType)` (samplers ride the uniform binding scope today —
   the WGSL legalise pass handles the split into texture+sampler).

2. **No closure getter.** Sampler captures don't need a getter —
   the texture handle is a draw-time argument supplied alongside
   uniforms. Plugin emits no `holes` entry for them.

3. **Tests.**
   - Sampler closure capture → `Sampler` ValueDef in template.
   - `texture(sampler, uv)` translates correctly with the captured
     sampler resolved.
   - Same source pattern under WGSL emitter produces split
     `texture_2d`+`sampler` pair via `legaliseTypes`.

**Acceptance.** Realistic textured-quad / lit-cube examples written
inline with the marker pattern compile and run.

---

## Phase 5 — Intrinsic coverage ✅ DONE (initial batch)

**Goal.** Cover the calls a real renderer actually issues. Ship as
two passes: an audit + an additions PR.

**What needs doing.**

1. **Audit.** List every intrinsic FShade exposes vs ours. Annotate
   what's missing per stage (vertex / fragment / compute) and what's
   target-conditional (atomics already are).

2. **Additions, in expected use order.**
   - Packing: `packUnorm4x8`, `packSnorm4x8`, `unpackUnorm4x8`,
     `unpackSnorm4x8`, `pack2x16`, `unpack2x16`.
   - Bit ops: `countOneBits`, `countLeadingZeros`, `countTrailingZeros`,
     `firstLeadingBit`, `firstTrailingBit`, `extractBits`,
     `insertBits`, `reverseBits`.
   - Math: `fma`, `frexp`, `ldexp`, `modf`, `degrees`, `radians`,
     `sinh`, `cosh`, `tanh`.
   - Texture: `textureGather`, `textureGatherCompare`,
     `textureSampleCompare`, `textureLoad`, `textureNumLayers`,
     `textureNumLevels`, `textureNumSamples`.
   - Compute / atomics: any missing `atomic*` not already added.
     Subgroup ops (extension-gated).

3. **Each addition gets:**
   - `declare function` in `@aardworx/wombat.shader-types`.
   - Entry in `frontend/intrinsics.ts` with right `pure`, `atomic`,
     `samplerBinding` flags and a `returnTypeOf`.
   - One test exercising it through to GLSL + WGSL emit.

**Acceptance.** A user porting a non-trivial GLSL/WGSL shader
doesn't hit "frontend: unknown function" for stock library calls.

---

## Phase 6 — Source maps ✅ DONE (plugin side; emitter span maps deferred)

**Goal.** Build errors and runtime errors point back at the user's
source, not at the plugin-emitted JSON literal.

**What needs doing.**

1. **Plugin returns a map.** Vite's `transform` hook supports a
   `map` field; today we return `null`. Build a map that:
   - Maps the inserted `__wombat_stage(...)` call to the original
     marker call site.
   - Optionally: per-IR-node spans pointing back at the arrow body
     (`Span` already exists on Expr).

2. **Emitter forwards spans to GLSL/WGSL line maps.** Wire the
   IR's `Span` into the emitter so shader compiler errors include
   the originating TS line/column.

3. **Tests.** Inject a deliberate error in a marker body and verify
   the plugin error message pinpoints the right TS source location.

**Acceptance.** "Why doesn't this compile" answered by the source
location, not by reading IR JSON.

---

## Phase 7 — Debug pretty-printer ✅ DONE

**Goal.** Quick way to dump a Module's IR human-readably during pass
development and user debugging.

**What needs doing.**

1. **`prettyPrint(module): string`** in `@aardworx/wombat.shader-ir`.
   FShade-style: each ValueDef rendered with a header, each Stmt /
   Expr indented, type annotations on every node. Goal is "I can
   read this and understand what the IR says without consulting the
   types file."

2. **Hook on Effect.** `effect.dumpIR()` for runtime introspection.

3. **Plugin debug mode.** Plugin option `debug: true` writes the
   pre-emit IR to a side file alongside each transformed source.

**Acceptance.** Internal debugging stops requiring `JSON.stringify`.

---

## Phase 8 — Storage buffers + storage images / textures ✅ DONE

**Goal.** Compute shaders write to texture bindings via WGSL
`texture_storage_2d<format, access>` (`textureLoad` / `textureStore`)
and GLSL `image2D` / `imageLoad` / `imageStore`. We don't have IR
support today.

**What needs doing.**

1. **IR**: new type variant `StorageTexture { target, format, access }`
   alongside existing `Sampler` / `Texture`. `format` is one of
   `rgba8unorm`/`r32float`/`rgba16float`/etc.; `access` ∈ `read` /
   `write` / `read_write`.
2. **Shipped types**: `StorageTexture2D<F, A>` (format + access as
   string-literal type parameters). Variants for 3D and 2DArray.
3. **Intrinsics**: `textureLoad(tex, coord)` returning the format's
   element type, `textureStore(tex, coord, value)` returning void.
   Mark write variants `pure: false` so DCE keeps them.
4. **WGSL emit**: `texture_storage_2d<rgba8unorm, write>` etc.
5. **GLSL**: WebGL2 ES 3.00 has no `image*` types — error.
6. **Plugin**: classify storage-image captures via the sampler/aval
   path (binding-getter on Effect; emit `Sampler` ValueDef carrying
   the StorageTexture type).
7. **Storage access inference reuse**: extend `inferStorageAccess`
   to flip a storage-texture binding's `access` based on
   `textureLoad`/`textureStore` usage in the body.

## Phase 9 — Operators + reuse wombat.base everywhere ✅ DONE

- Method translator now covers the full wombat.base V*/M* API
  (`lengthSquared`, `distance`, `distanceSquared`, `lerp`,
  `abs`/`floor`/`ceil`/`round`/`fract`/`sign`, `min`/`max`/`clamp`,
  on top of the original `add`/`sub`/`mul`/`div`/`dot`/`cross`/
  `length`/`normalize`/`transpose`/`inverse`/`determinant`).
- `@aardworx/wombat.shader-types` now re-exports `V*b` / `V*i` /
  `V*ui` / `V*f` / `M{R}{C}f` from `@aardworx/wombat.base` (^0.1.0).
  Single source of truth for math types — same class is used on the
  CPU and in shader markers.
- Frontend type table accepts both `V*u` (legacy) and `V*ui`
  (wombat.base canonical) names for the unsigned vectors.
- README documents boperators wiring (`tsconfig` plugins + Vite
  plugin order) so users get `+`/`-`/`*`/`/` operator overloading
  with full LSP support.
- Sampler shims kept with `__aardworxShaderBrand` (samplers aren't
  math types and don't need operator support).

**Goal.** Eliminate the math-types duplication. Today we ship empty
`declare class V3f` shims in `@aardworx/wombat.shader-types`;
`@aardworx/wombat.base` already ships real runtime classes with full
method coverage and a `__aardworxMathBrand` that
[`boperators`](https://npmjs.com/package/boperators) recognizes for
operator overloading (`+` / `-` / `*` / `/`). Reusing them gives
shader code: real CPU instances of the same types, full operator
syntax, and one source of truth for `V3f`/`M44f`/etc.

**What needs doing.**

1. **`@aardworx/wombat.shader-types` re-exports wombat.base.** Drop
   the `declare class` shims; replace with `export { V3f, M44f, … }
   from "@aardworx/wombat.base"`. Keep shader-only types (samplers,
   `Storage<T>`, `StorageTexture*`, `ComputeBuiltins`, `i32`/`u32`/`f32`
   scalars) here.

2. **Frontend `tryResolveTypeName` already keys on names** (`"V3f"` →
   `Vector(Float, 3)`) — works the same whether the class is a
   `declare class` shim or a wombat.base runtime class. No change.

3. **Extend `translateMethodCall`** with the rest of wombat.base's
   API: `lengthSquared`, `distance`, `distanceSquared`, `lerp`,
   `abs`/`floor`/`ceil`/`round`/`fract`/`sign`, `min`/`max`/`clamp`,
   `minComp`/`maxComp`/`sumComp`. Map each to either an existing IR
   node or a `CallIntrinsic`. `lengthSquared(v)` lowers to
   `Dot(v, v)`; `lerp` → `mix`; the rest map 1:1 to intrinsics.

4. **boperators wiring.** Document the user-side setup:
   - Add `boperators`, `@boperators/plugin-tsc`,
     `@boperators/plugin-ts-language-server`, `@boperators/plugin-vite`
     as dev deps.
   - Vite config: `boperatorsVite()` *before* `wombatShader()` in the
     plugin array (operator rewrite must precede frontend AST walk).
   - tsconfig: register the tsc + LSP plugins so type checking sees
     operator overloads.

5. **Cleanup**: replace `vec4(...)` / `vec3(...)` / `mat4(...)` calls
   in tests, examples, and docs with `new V4f(...)` / etc. Both forms
   continue to work in the frontend, but the canonical style is
   `new V*f(…)` now that the constructor exists at runtime.

6. **Tests.**
   - Operator rewrite (run a tiny boperators-rewritten source through
     the plugin, check WGSL emit matches the method-form output).
   - Method-call coverage for the new translator entries.
   - Round-trip: import `V3f` from wombat.base, use in a shader, check
     the IR carries the right brand-resolved type.

**Acceptance.** A shader body uses `new V3f(…)` / `+` / `*` / `.lerp()`
naturally; the IDE provides autocomplete and type-checks operators;
the same V3f instance can be used in CPU code surrounding the shader.

---

## Phase 10 — Cross-file helpers ✅ DONE

**Goal.** Today the plugin's helper-discovery walker resolves only
top-level functions in the *same source file* as the marker call.
Real apps split helpers across modules.

**What needs doing.**

1. **Symbol resolution across files.** Use the TypeResolver's
   LanguageService to follow an `Identifier`'s symbol through `import`
   declarations to its declaring file. Already-discovered helpers
   in other files get their AST walked and translated.

2. **Synthesised source merging.** When a marker references helpers
   from N files, the plugin's synthesised source string concatenates
   each helper's source (with rewritten names if collisions). A
   `helperName_h<file-hash>` mangling scheme avoids collisions.

3. **Caching.** Resolved helpers (per file) cached on the
   TypeResolver across transforms — Vite calls `transform` once per
   file per build, so a helper used by N shaders gets parsed once.

4. **Tests.**
   - Helper in a sibling file used from a marker call.
   - Helper from a workspace package (cross-package).
   - Helper that itself imports from other files (transitive).
   - Name collision between two helpers from different files →
     mangle and verify the IR has both `Function` ValueDefs.

**Acceptance.** A user can put `function blendColors(…)` in a
`shaders/utils.ts` file, import it into many marker bodies, and
have it discovered + translated automatically.

---

## Phase 11 — Emitter source maps ✅ DONE

**Goal.** GPU compile errors and runtime warnings point back at the
user's TS source, not into the emitted GLSL/WGSL. The IR has `Span`
already; the emitters drop it.

**What needs doing.**

1. **Output buffer with line tracking.** Replace the emitters'
   plain string buffer with a structure that records, per emitted
   line, the originating `Span` (file/start/end). When a Stmt emits
   multiple lines, all lines map to the same span by default; nested
   Exprs may attach more granular spans.

2. **v3 source-map emit alongside the shader source.** Each
   `CompiledStage` gains a `sourceMap?: { mappings: string; sources:
   string[]; … }` field. Plugin re-uses the spans to chain the map
   back through the inline-shader transform's own map.

3. **Plugin diagnostics with TS spans.** Frontend translation
   diagnostics already carry file/start/end; the plugin's error
   message uses them. Sweep to make sure all error paths surface
   them consistently.

4. **Tests.**
   - Inject an undefined identifier in a marker body; verify the
     plugin error points at the right TS line/column.
   - Compile a shader that produces a syntactic GLSL error (forced
     via raw IR injection); verify the source-map locates the
     offending line in user TS.

**Acceptance.** "Why does the GPU compiler reject this" is answered
by a TS line number, not by reading WGSL.

---

## Phase 12 — Row/column-major reversal pass ✅ DONE

**Goal.** wombat.base's matrices are row-major (Aardvark convention).
GLSL/WGSL store matrices column-major by default. FShade's solution:
write matrix code naturally on the CPU side (`mvp * vec4(pos, 1)`,
`view = lookAt(...)` chained calls) and have the shader emitter
reverse all matrix operations at IR level so the column-major-on-GPU
result agrees with the row-major-on-CPU intent. No `transpose()`
calls peppered in user code; no manual upload-time transposes.

**What needs doing.**

1. **A normalisation pass `reverseMatrixOps(module)`** that runs late
   (after most optimisation, before legalisation):
   - `MulMatVec(M, v)` → `MulVecMat(v, M)`
   - `MulVecMat(v, M)` → `MulMatVec(M, v)`
   - `MulMatMat(A, B)` → `MulMatMat(B, A)`
   - `Transpose(M)` becomes a no-op (logically, the matrix is
     already transposed on the GPU side)
   - `MatrixFromCols(c0, c1, …)` ↔ `MatrixFromRows(c0, c1, …)` in
     the emitted shader (the row-major data uploaded becomes column-
     major from the shader's view, so a "from cols" call on CPU is
     a "from rows" decl in shader source)

2. **No double-reversal.** The pass runs exactly once per shader.
   Tag the module after applying so a second invocation is a no-op.

3. **Plugin/runtime integration.** Run the pass automatically as
   part of `compileModule`'s pipeline, after `inferStorageAccess`
   and before `legaliseTypes`. Optional `skipMatrixReversal: true`
   in `CompileOptions` for users who already pre-transpose their
   uniforms (the Phase 0 examples do this).

4. **Tests.**
   - A vertex shader that writes `mvp.mul(vec4(pos, 1))` produces
     WGSL that reads as `pos * mvp` (or equivalent) and gives the
     same projected position when the same row-major mvp matrix
     is uploaded.
   - Round-trip: same operation, same upload, same gl_Position
     under both targets.

**Acceptance.** A user can take a row-major `M44f.translation(v)`
from wombat.base, multiply with another row-major matrix on the
CPU, hand the result to a shader as a uniform, and use it via
`u.mvp.mul(vec4(pos, 1))` — it just works. The shader emits the
right reversed operation; the upload is a straight memcpy.

---

## Out of scope

For clarity — these are deliberately deferred:

- **Adaptive (`aval`-driven) uniforms.** Per the user, not now.
  Holes wired through the existing constant-specialization path
  remain the only capture mechanism.
- **Tessellation / geometry / raytracing.** No web surface.
- **SPIR-V.** No web surface.
- **Surface composition (FShade Surface).** Belongs to the rendering
  framework on top, not the shader compiler.
- **Bindless / texture arrays.** Punt until a real consumer needs it.
