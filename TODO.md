# wombat.shader — TODO

Status: ✅ stable (0.5.9). TS → WGSL/GLSL compiler with IR optimisation
(inlining, cross-stage I/O elimination, const-fold, CSE, DCE, helper-fusion
linker, matrix row/col reversal). All 12 planned phases shipped; 182+ tests.

Reference docs kept: `docs/INTRINSICS.md`, `docs/IR.md`.

## Open

- **Public IR substitution APIs** — expose `substituteUniforms` /
  `substituteAttributes` and a way to prepend a synthetic decoder vertex stage.
  Needed by wombat.rendering's IR-based heap rewriter (central TODO #1–2).
- **`NewStruct` IR node** — construct struct values in the IR. Unblocks
  wombat.fable storage-buffer record writeback + `arrayLength` (central TODO #4).
- **Intrinsic ergonomics** (low priority) — a ships-with-package ambient-global
  pattern so users don't `import { abs, max, … }` per file; audit
  `SHIPPED_INTRINSIC_NAMES` for any name lacking a matching `.d.ts` declaration.

## Out of scope (not planned)

- Tessellation / geometry / raytracing stages (no web surface).
- SPIR-V (no web surface).
- Surface composition (belongs to the rendering framework).
- Bindless / texture arrays (deferred until a concrete need).
- aval-driven adaptive uniforms (deliberately out of scope).
