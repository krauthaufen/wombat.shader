# wombat.shader

TypeScript port of [FShade](https://fshade.org). The pipeline:
**TS shader source → IR → optimisation passes → GLSL ES 3.00 / WGSL emit**.
See [`README.md`](README.md) for scope and roadmap, [`docs/IR.md`](docs/IR.md)
for the canonical IR specification.

## Workspace layout

npm workspaces, TypeScript project references. Each package builds to its
own `dist/` and depends on others via the workspace symlink.

```
wombat.shader/
├─ packages/
│  ├─ ir/      @aardworx/wombat.shader-ir       — IR types + visitors + JSON
│  ├─ glsl/    @aardworx/wombat.shader-glsl     — IR → GLSL ES 3.00
│  └─ wgsl/    @aardworx/wombat.shader-wgsl     — IR → WGSL
└─ tests/      vitest tests at the root
```

Future packages (planned, not yet present): `passes/`, `frontend/`,
`types/`, `runtime/`, `vite/`, `swc/`. See `README.md` roadmap.

## Tooling

- `npm run build` — `tsc -b` across the workspace, respects project
  references. Each package emits `.d.ts`, `.js`, source maps.
- `npm test` — vitest. Tests live at `tests/`. Each emitter test
  hand-builds an IR Module and compares emitter output via
  `toMatchInlineSnapshot` or substring assertions.
- `npm run typecheck` — `tsc -b --noEmit`.

## IR is the contract

Every node is a discriminated union on `kind`. JSON-serialisable. No
runtime classes. The frontend (TS source → IR), the passes
(IR → IR), and the emitters (IR → string) all communicate only through
this shape. The IR types live in `packages/ir/src/types.ts` and re-export
through `index.ts`. **Don't add ad-hoc fields to nodes** — extend the
discriminated union with a new `kind` and update visitors and emitters.

When adding a new IR node:
1. Add the variant in `packages/ir/src/types.ts`.
2. Add it to `visitExprChildren` / `visitLExprChildren` / `visitStmt`
   in `packages/ir/src/visit.ts` so passes traverse it.
3. Handle it in **both** emitters (`packages/glsl/src/emit.ts`,
   `packages/wgsl/src/emit.ts`). TypeScript exhaustiveness checks should
   force this if you switch on `e.kind` without a default.
4. Add a test exercising the new node.

## Composition vs. lowering

Some IR nodes are intentionally high-level and require an upstream
*legalisation* pass to translate them into target-specific shapes:

- `MatrixRow` — neither GLSL nor WGSL has a direct row accessor. Lower
  to `MatrixElement` reads, or to `transpose(...)[r]`.
- `MatrixFromRows` — both backends construct matrices column-major.
  Frontend should lower to `MatrixFromCols` of a transposed input, or
  emit `transpose(MatrixFromCols(...))`.
- `Inverse` for matrices is built into GLSL but not WGSL — WGSL needs
  a user-supplied helper. Legalisation should inject one when needed.
- `DebugPrintf` — neither backend supports it. Currently emits a noop
  comment; passes should drop these in production builds.
- `Texture` (without an accompanying `Sampler`) is invalid in GLSL —
  fold combined samplers when targeting GLSL.

The emitters each have a defensive comment-and-noop path for the
"frontend should have lowered this" cases so unlowered IR doesn't
silently produce wrong code; it produces obviously wrong code.

## Composition convention

Matrix · matrix uses standard math convention (do RHS first then LHS),
matching every other math type in the wombat stack except `Trafo3d` /
`Trafo2d` (which intentionally compose left-to-right). The IR doesn't
encode a "trafo" — that lives in the frontend translation rules.

## Don'ts

- Don't put runtime classes in the IR. Plain `{kind, ...}` only — JSON
  must round-trip.
- Don't import `typescript` (the compiler) from `ir/`, `glsl/`, or
  `wgsl/`. Only the future `frontend/` package may.
- Don't write platform-specific code (DOM, WebGL, WebGPU) in `ir/`,
  `glsl/`, or `wgsl/`. Emitters return strings — they don't link.
- Don't use `as any` to dodge the exhaustiveness checker on `Expr.kind`.
  If you're patching one emitter, patch the other.
