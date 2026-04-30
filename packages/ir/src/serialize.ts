// JSON (de)serialisation for IR Modules.
//
// Most IR nodes are plain `{kind, ...}` objects, so they round-trip via
// JSON.stringify / JSON.parse with no transformation. Exceptions:
//
//  - `IntrinsicRef.returnTypeOf` is a function — replaced with a stable
//    string id. The deserialiser looks up the function in a registry.
//  - `FunctionRef` is by id; the actual `FunctionDef` lives in
//    `Module.values`. We just persist the id and signature.
//
// The frontend is responsible for using the same intrinsic registry
// at serialise and deserialise time.

import type { IntrinsicRef, Module } from "./types.js";

export interface IntrinsicRegistry {
  get(name: string): IntrinsicRef;
}

// JSON-friendly mirror of IntrinsicRef (without the function field).
interface SerializedIntrinsic {
  readonly $intrinsic: true;
  readonly name: string;
}

function isSerializedIntrinsic(v: unknown): v is SerializedIntrinsic {
  return typeof v === "object" && v !== null && (v as Record<string, unknown>)["$intrinsic"] === true;
}

/** Serialise a Module to a JSON-safe object. */
export function serialise(module: Module): unknown {
  return JSON.parse(JSON.stringify(module, (_key, value) => {
    if (typeof value === "function") {
      throw new Error("IR contains a function; only IntrinsicRef may carry one (serialise it via the registry id)");
    }
    if (typeof value === "object" && value !== null) {
      const v = value as Record<string, unknown>;
      if (v["kind"] === "CallIntrinsic") {
        const op = v["op"] as IntrinsicRef;
        return {
          ...v,
          op: { $intrinsic: true, name: op.name } satisfies SerializedIntrinsic,
        };
      }
    }
    return value;
  }));
}

/** Deserialise a JSON object back into a Module, looking up intrinsics in the registry. */
export function deserialise(json: unknown, registry: IntrinsicRegistry): Module {
  const reviver = (_key: string, value: unknown): unknown => {
    if (isSerializedIntrinsic(value)) {
      return registry.get(value.name);
    }
    return value;
  };
  // Walk and revive (JSON.parse has reviver, but our input is already an object).
  return JSON.parse(JSON.stringify(json), reviver) as Module;
}
