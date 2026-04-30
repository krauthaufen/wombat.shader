// JSON round-trip test for IR Modules.

import { describe, expect, it } from "vitest";
import {
  Tbool,
  Tf32,
  Tvoid,
  Vec,
  deserialise,
  serialise,
  type IntrinsicRef,
  type IntrinsicRegistry,
  type Module,
} from "@aardworx/wombat.shader-ir";

const Tvec3f = Vec(Tf32, 3);
const Tvec4f = Vec(Tf32, 4);

const Sin: IntrinsicRef = {
  name: "sin", pure: true,
  emit: { glsl: "sin", wgsl: "sin" },
  returnTypeOf: ([t]) => t!,
};
const registry: IntrinsicRegistry = {
  get(name) {
    if (name === "sin") return Sin;
    throw new Error(`unknown intrinsic ${name}`);
  },
};

function sampleModule(): Module {
  return {
    types: [{
      kind: "Struct", name: "Material",
      fields: [
        { type: Tvec3f, name: "albedo" },
        { type: Tf32, name: "roughness" },
      ],
    }],
    values: [
      { kind: "Uniform", uniforms: [{ name: "u_time", type: Tf32 }] },
      {
        kind: "Entry",
        entry: {
          name: "main", stage: "fragment",
          inputs: [],
          outputs: [
            { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
          ],
          arguments: [],
          returnType: Tvoid,
          body: {
            kind: "Sequential",
            body: [
              {
                kind: "If",
                cond: {
                  kind: "Gt",
                  lhs: {
                    kind: "CallIntrinsic", op: Sin, args: [
                      { kind: "ReadInput", scope: "Uniform", name: "u_time", type: Tf32 },
                    ], type: Tf32,
                  },
                  rhs: { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 },
                  type: Tbool,
                },
                then: {
                  kind: "WriteOutput", name: "outColor",
                  value: {
                    kind: "Expr",
                    value: {
                      kind: "NewVector",
                      components: [
                        { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 },
                        { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 },
                        { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 },
                        { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 },
                      ],
                      type: Tvec4f,
                    },
                  },
                },
              },
            ],
          },
          decorations: [],
        },
      },
    ],
  };
}

describe("IR JSON round-trip", () => {
  it("serialises to JSON.stringify-safe form", () => {
    const m = sampleModule();
    const j = serialise(m);
    const text = JSON.stringify(j);
    expect(text).toContain("u_time");
    expect(text).toContain("\"$intrinsic\":true");
    expect(text).toContain("\"sin\"");
    // Functions must not appear in the serialised form.
    expect(text).not.toContain("returnTypeOf");
  });

  it("deserialises back through the intrinsic registry", () => {
    const m = sampleModule();
    const j = serialise(m);
    const round = deserialise(j, registry);
    // Round-tripped CallIntrinsic should reference the same registry entry.
    function findCall(value: unknown): IntrinsicRef | undefined {
      if (typeof value !== "object" || value === null) return undefined;
      const v = value as Record<string, unknown>;
      if (v["kind"] === "CallIntrinsic") return v["op"] as IntrinsicRef;
      for (const x of Object.values(v)) {
        const r = findCall(x);
        if (r) return r;
      }
      if (Array.isArray(value)) {
        for (const x of value) {
          const r = findCall(x);
          if (r) return r;
        }
      }
      return undefined;
    }
    const op = findCall(round);
    expect(op).toBe(Sin);
    expect(op?.returnTypeOf([Tf32])).toBe(Tf32);
  });

  it("rejects modules containing user-supplied functions outside intrinsics", () => {
    const m = {
      types: [], values: [{
        kind: "Constant" as const,
        varType: Tf32,
        name: "BAD",
        init: { kind: "Expr" as const, value: ({ getThing: () => 0 } as unknown as import("@aardworx/wombat.shader-ir").Expr) },
      }],
    };
    expect(() => serialise(m as unknown as Module)).toThrow();
  });
});
