// pruneToStage — drop function bodies / Entries unreachable from a target stage.
//
// Per-stage WGSL/GLSL emit produces one module-string per Entry, but the
// emitter walks `module.values` and writes EVERY `Function` ValueDef into
// the output regardless of which stage's Entry is being emitted. When a
// Module contains both vertex + fragment Entries plus same-stage helper
// functions (e.g. `extractFusedEntry`'s output: 3 VS-only helpers + a VS
// wrapper Entry + a FS Entry), the FS string ends up containing the VS
// helpers verbatim. Those helpers reference VS-only globals (e.g. a
// `var<private>` like `heap_drawIdx` or VS-side `ReadInput("Input", …)`),
// which don't exist in the FS module — Tint flags them as
// `unresolved value` and the WGSL compile fails.
//
// Fix: tree-shake by entry-reachability per stage. Walk the call graph
// from each target-stage Entry, mark every `Function` ValueDef
// transitively reachable, drop all other Entries (other-stage) and any
// non-live Function. Bindings (Uniform / Sampler / StorageBuffer),
// Constants, and Type defs stay — they're declarations, not function
// references, and surrounding passes already drop unused ones.

import type { Module, Stage, Stmt, ValueDef } from "../ir/index.js";
import { visitStmt } from "../ir/visit.js";

/**
 * Return a new Module containing only the items reachable from any
 * Entry whose `stage` matches `stage`:
 *  - all Entries of the target stage (kept verbatim)
 *  - every `Function` ValueDef transitively reachable from those
 *    entries' bodies via `Call(FunctionRef)` expressions
 *  - all non-Function / non-Entry ValueDefs (Uniform, Sampler,
 *    StorageBuffer, Constant) are passed through untouched
 *  - all TypeDefs are passed through
 *
 * Entries of OTHER stages are dropped, as are Function ValueDefs the
 * target-stage entries don't reach.
 */
export function pruneToStage(module: Module, stage: Stage): Module {
  // Index Function ValueDefs by their signature name.
  const fnByName = new Map<string, ValueDef & { kind: "Function" }>();
  for (const v of module.values) {
    if (v.kind === "Function") fnByName.set(v.signature.name, v);
  }

  // Seed the live set with names of target-stage Entries (they're
  // always kept) and walk outward.
  const live = new Set<string>();
  const worklist: Stmt[] = [];

  for (const v of module.values) {
    if (v.kind === "Entry" && v.entry.stage === stage) {
      worklist.push(v.entry.body);
    }
  }

  while (worklist.length > 0) {
    const body = worklist.pop()!;
    visitStmt(body, {
      expr: {
        pre: (e) => {
          if (e.kind === "Call") {
            const name = e.fn.signature.name;
            if (!live.has(name)) {
              live.add(name);
              const fn = fnByName.get(name);
              if (fn) worklist.push(fn.body);
            }
          }
        },
      },
    });
  }

  const values: ValueDef[] = [];
  for (const v of module.values) {
    if (v.kind === "Entry") {
      if (v.entry.stage === stage) values.push(v);
      // else drop (other-stage entry)
    } else if (v.kind === "Function") {
      if (live.has(v.signature.name)) values.push(v);
      // else drop (unreachable helper)
    } else {
      values.push(v);
    }
  }

  return { ...module, values };
}
