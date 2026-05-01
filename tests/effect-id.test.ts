// Build-time stable effect identifiers — backends use these as cache
// keys for emitted shader source. Tests:
//
//   1. Identical IR → identical id (deterministic).
//   2. Different IR → different id.
//   3. effect(v, f) composes ids deterministically; same inputs in
//      same order always produce the same composed id.
//   4. Plugin-emitted output carries the same id as the runtime
//      hashModule of the embedded template.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";
import { effect, stage } from "@aardworx/wombat.shader";
import {
  combineHashes, hashModule,
  type EntryDef, type Module, type Stmt, type Type,
} from "@aardworx/wombat.shader/ir";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };

function trivialFragment(value: number): Module {
  const body: Stmt = {
    kind: "WriteOutput",
    name: "outColor",
    value: { kind: "Expr", value: {
      kind: "NewVector",
      components: [
        { kind: "Const", value: { kind: "Float", value }, type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 0.5 }, type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 0.5 }, type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 1.0 }, type: Tf32 },
      ],
      type: Tvec4f,
    }},
  };
  const entry: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [], outputs: [{
      name: "outColor", type: Tvec4f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    arguments: [], returnType: { kind: "Void" },
    body, decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
}

describe("Effect id (build-time stable hashes)", () => {
  it("identical IR templates produce identical ids", () => {
    const a = stage(trivialFragment(0.1));
    const b = stage(trivialFragment(0.1));
    expect(a.id).toBe(b.id);
  });

  it("different IR templates produce different ids", () => {
    const a = stage(trivialFragment(0.1));
    const b = stage(trivialFragment(0.9));
    expect(a.id).not.toBe(b.id);
  });

  it("effect(v, f).id is deterministic on input ids and order", () => {
    const a = stage(trivialFragment(0.1));
    const b = stage(trivialFragment(0.9));
    const ab = effect(a, b);
    const ab2 = effect(a, b);
    const ba = effect(b, a);
    expect(ab.id).toBe(ab2.id);
    expect(ab.id).toBe(combineHashes(a.id, b.id));
    expect(ab.id).not.toBe(ba.id);
  });

  it("plugin-emitted id matches hashModule of the embedded template", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, 0.5, 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    // Find the trailing id literal.
    const idMatch = /__wombat_stage\([\s\S]+, \{\}, "([0-9a-f]{16})", \{\}\)/.exec(r.code);
    expect(idMatch).not.toBeNull();
    const emittedId = idMatch![1];

    // Pull out the template JSON to verify the id matches what
    // hashModule would compute against it at runtime.
    const idx = r.code.indexOf("__wombat_stage(");
    let i = r.code.indexOf("{", idx);
    const start = i;
    let depth = 0;
    for (; i < r.code.length; i++) {
      if (r.code[i] === "{") depth++;
      else if (r.code[i] === "}") { depth--; if (depth === 0) { i++; break; } }
    }
    const tmpl = JSON.parse(r.code.slice(start, i)) as Module;
    expect(hashModule(tmpl)).toBe(emittedId);
  });
});
