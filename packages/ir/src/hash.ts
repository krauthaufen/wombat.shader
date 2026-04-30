// Stable, content-addressed hashing for IR Modules and arbitrary
// JSON-shaped values. Used to assign deterministic identifiers to
// shader effects at build time so backends can cache GLSL/WGSL by
// effect id without re-emitting source.
//
// The hash is FNV-1a 64-bit (two interleaved 32-bit FNVs, since JS
// has no native u64). Output is a 16-char lowercase hex string.
// 64-bit collision odds at the scale of shader-count we'll ever see
// are negligible; the hash isn't cryptographic and isn't meant to
// resist adversarial inputs.
//
// Object keys are sorted before serialisation so insertion-order
// changes don't shift the hash. Functions (e.g. `IntrinsicRef.returnTypeOf`)
// are dropped — `IntrinsicRef`'s identity-bearing fields (`name`,
// `pure`, `atomic`, `samplerBinding`, `emit.glsl`, `emit.wgsl`) are
// captured by the rest of the canonical form.

import type { Module } from "./types.js";

/** Hash of an IR Module. Stable across runs and across machines. */
export function hashModule(module: Module): string {
  return fnv1a64(stableStringify(module));
}

/** Hash of any JSON-shaped value (functions skipped). */
export function hashValue(value: unknown): string {
  return fnv1a64(stableStringify(value));
}

/** Combine N child hashes into one parent hash. Order matters. */
export function combineHashes(...ids: readonly string[]): string {
  return fnv1a64("compose:" + ids.join("|"));
}

// ─────────────────────────────────────────────────────────────────────

/**
 * Keys whose values are excluded from the canonical hash. Source
 * spans are positional info, not semantic — same shader compiled
 * from different file locations should hash identically.
 */
const HASH_IGNORED_KEYS = new Set(["span"]);

export function stableStringify(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "function") return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) {
    return "[" + value.map(stableStringify).join(",") + "]";
  }
  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    const parts: string[] = [];
    for (const k of keys) {
      if (HASH_IGNORED_KEYS.has(k)) continue;
      const v = (value as Record<string, unknown>)[k];
      if (typeof v === "function") continue;
      if (v === undefined) continue;
      parts.push(JSON.stringify(k) + ":" + stableStringify(v));
    }
    return "{" + parts.join(",") + "}";
  }
  return "null";
}

// FNV-1a 64-bit, computed as two interleaved 32-bit FNVs over the
// same input with different starting offsets. Implemented with
// Math.imul to keep multiplication in the 32-bit unsigned range.
function fnv1a64(str: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x84222325;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    h1 ^= c;
    h1 = Math.imul(h1, 0x01000193);
    h2 ^= c;
    h2 = Math.imul(h2, 0x01000193);
    // Mix the two streams so identical bytes don't produce equal halves.
    h2 = Math.imul(h2 ^ (h1 >>> 13), 0x9e3779b1);
  }
  const a = (h1 >>> 0).toString(16).padStart(8, "0");
  const b = (h2 >>> 0).toString(16).padStart(8, "0");
  return a + b;
}
