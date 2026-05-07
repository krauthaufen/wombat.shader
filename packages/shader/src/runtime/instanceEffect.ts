// `instanceEffect` — wrap an Effect so its `compile()` runs the
// `instanceUniforms` IR pass over the merged module. Used by the
// scene-graph layer when an `Sg.Instanced` scope is in effect: the
// scope-supplied uniform names get rewritten to read from per-instance
// vertex attributes instead of bind-group uniforms.
//
// The wrapper only adjusts the compile pipeline — the underlying
// stages are reused as-is. The new effect's `id` mixes the original
// id with the sorted attribute set so two instancing scopes with
// different attr sets produce distinct cache keys downstream.

import type { Effect } from "./stage.js";
import { combineHashes } from "../ir/hash.js";

/**
 * Return a new Effect whose `compile()` applies the `instanceUniforms`
 * pass with `attrNames` set. Reading a wrapped effect's id, dumping
 * its IR, or composing it via `effect(...)` all behave the same as
 * the inner effect.
 */
export function instanceEffect(
  inner: Effect,
  attrNames: ReadonlySet<string>,
): Effect {
  if (attrNames.size === 0) return inner;
  const sorted = [...attrNames].sort();
  const id = combineHashes(inner.id, "INST", ...sorted);
  return {
    stages: inner.stages,
    id,
    dumpIR: () => inner.dumpIR(),
    compile(options) {
      return inner.compile({
        ...options,
        instanceAttributes: attrNames,
      });
    },
  };
}
