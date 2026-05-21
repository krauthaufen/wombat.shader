# wombat.shader — TODO

Status: ✅ stable (0.5.9). TS → WGSL/GLSL compiler with IR optimisation
(inlining, cross-stage I/O elimination, const-fold, CSE, DCE, helper-fusion
linker, matrix row/col reversal). All 12 planned phases shipped; 182+ tests.

Reference docs kept: `docs/INTRINSICS.md`, `docs/IR.md`.

## Open

- **Synthetic decoder-stage prepend** — the IR substitution passes
  (`substituteUniforms` / `substituteAttributes` / `substituteInputs`) already
  exist and ship via the `./passes` subpath. What's still missing for
  wombat.rendering's heap rewriter is a public way to prepend/compose a synthetic
  decoder vertex stage (central TODO #1–2). Also re-export the substitution
  passes from the root entry, not just `./passes`.
- **`NewStruct` IR node** — construct struct values in the IR. Unblocks
  wombat.fable storage-buffer record writeback + `arrayLength` (central TODO #4).
- **Intrinsic ergonomics** (low priority) — (a) a ships-with-package
  ambient-global types pattern so users don't `import { abs, max, … }` per file;
  (b) `.d.ts` gaps confirmed: the vec/mat constructors (`vec2..4`, `ivec*`,
  `uvec*`, `mat2..4` / `matNxM`), `textureGather`, and `any` / `all` are in
  `SHIPPED_INTRINSIC_NAMES` but have no type declaration. (`dot` / `length` /
  `cross` / `distance` / `normalize` are intentionally methods on wombat.base
  vectors — not gaps.)

## Out of scope (not planned)

- Tessellation / geometry / raytracing stages (no web surface).
- SPIR-V (no web surface).
- Surface composition (belongs to the rendering framework).
- Bindless / texture arrays (deferred until a concrete need).
- aval-driven adaptive uniforms (deliberately out of scope).
