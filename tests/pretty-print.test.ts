// Pretty-printer + Effect.dumpIR() — human-readable IR for debugging.

import { describe, expect, it } from "vitest";
import { prettyPrint, type EntryDef, type Module, type Stmt, type Type } from "@aardworx/wombat.shader/ir";
import { effect, stage } from "@aardworx/wombat.shader";

const Tf32: Type = { kind: "Float", width: 32 };
const Tvec2f: Type = { kind: "Vector", element: Tf32, dim: 2 };
const Tvec4f: Type = { kind: "Vector", element: Tf32, dim: 4 };

function fragmentTemplate(): Module {
  const body: Stmt = {
    kind: "WriteOutput",
    name: "outColor",
    value: { kind: "Expr", value: {
      kind: "NewVector",
      components: [
        { kind: "ReadInput", scope: "Input", name: "v_uv", type: Tvec2f },
        { kind: "Const", value: { kind: "Float", value: 0.5 }, type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 0.5 }, type: Tf32 },
        { kind: "Const", value: { kind: "Float", value: 1 }, type: Tf32 },
      ],
      type: Tvec4f,
    }},
  };
  const entry: EntryDef = {
    name: "fsMain", stage: "fragment",
    inputs: [{
      name: "v_uv", type: Tvec2f, semantic: "Uv",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    outputs: [{
      name: "outColor", type: Tvec4f, semantic: "Color",
      decorations: [{ kind: "Location", value: 0 }],
    }],
    arguments: [], returnType: { kind: "Void" },
    body, decorations: [],
  };
  return { types: [], values: [{ kind: "Entry", entry }] };
}

describe("pretty-printer", () => {
  it("dumps a Module with an entry as readable text", () => {
    const text = prettyPrint(fragmentTemplate());
    expect(text).toContain("module {");
    expect(text).toContain("entry fragment fsMain");
    expect(text).toContain("inputs:    (v_uv: vec2<f32>)");
    expect(text).toContain("outputs:   (outColor: vec4<f32>)");
    expect(text).toContain("out.outColor =");
    expect(text).toContain("vec4<f32>(Input.v_uv, 0.5, 0.5, 1.0)");
  });

  it("Effect.dumpIR() renders each stage's template", () => {
    const fx = effect(stage(fragmentTemplate()));
    const text = fx.dumpIR();
    expect(text).toContain("// stage 0");
    expect(text).toContain("entry fragment fsMain");
  });
});
