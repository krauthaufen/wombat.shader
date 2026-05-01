// Tests for legaliseTypes.

import { describe, expect, it } from "vitest";
import {
  Mat,
  Tf32,
  Ti32,
  Tvoid,
  Vec,
  type Expr,
  type Module,
  type Stmt,
} from "@aardworx/wombat.shader/ir";
import { legaliseTypes } from "@aardworx/wombat.shader/passes";
import { emitGlsl } from "@aardworx/wombat.shader/glsl";
import { emitWgsl } from "@aardworx/wombat.shader/wgsl";

const Tvec3f = Vec(Tf32, 3);
const Tvec4f = Vec(Tf32, 4);
const Tmat4f = Mat(Tf32, 4, 4);

const constI = (n: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: n }, type: Ti32 });
const constF = (n: number): Expr => ({ kind: "Const", value: { kind: "Float", value: n }, type: Tf32 });

function fragModule(body: Stmt): Module {
  return {
    types: [],
    values: [{
      kind: "Entry",
      entry: {
        name: "main", stage: "fragment", inputs: [], outputs: [
          { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
        ],
        arguments: [], returnType: Tvoid, body, decorations: [],
      },
    }],
  };
}

describe("legaliseTypes — MatrixRow", () => {
  it("MatrixRow lowers to NewVector of MatrixElement reads", () => {
    const m: Expr = { kind: "ReadInput", scope: "Uniform", name: "u_m", type: Tmat4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "outColor",
      value: {
        kind: "Expr",
        value: { kind: "MatrixRow", matrix: m, row: constI(2), type: Tvec4f },
      },
    };
    const out = legaliseTypes(fragModule(body), "glsl");
    const entry = out.values.find((x) => x.kind === "Entry")!;
    if (entry.kind !== "Entry") throw new Error();
    const wo = entry.entry.body as Extract<Stmt, { kind: "WriteOutput" }>;
    expect(wo.value.kind).toBe("Expr");
    if (wo.value.kind === "Expr") {
      expect(wo.value.value.kind).toBe("NewVector");
      if (wo.value.value.kind === "NewVector") {
        expect(wo.value.value.components.length).toBe(4);
        for (const c of wo.value.value.components) {
          expect(c.kind).toBe("MatrixElement");
        }
      }
    }
  });
});

describe("legaliseTypes — MatrixFromRows", () => {
  it("MatrixFromRows lowers to Transpose(MatrixFromCols)", () => {
    const r0: Expr = { kind: "ReadInput", scope: "Uniform", name: "r0", type: Tvec4f };
    const r1: Expr = { kind: "ReadInput", scope: "Uniform", name: "r1", type: Tvec4f };
    const r2: Expr = { kind: "ReadInput", scope: "Uniform", name: "r2", type: Tvec4f };
    const r3: Expr = { kind: "ReadInput", scope: "Uniform", name: "r3", type: Tvec4f };
    const body: Stmt = {
      kind: "Expression",
      value: { kind: "MatrixFromRows", rows: [r0, r1, r2, r3], type: Tmat4f },
    };
    const out = legaliseTypes(fragModule(body), "wgsl");
    const entry = out.values.find((x) => x.kind === "Entry")!;
    if (entry.kind !== "Entry") throw new Error();
    const ex = (entry.entry.body as Extract<Stmt, { kind: "Expression" }>).value;
    expect(ex.kind).toBe("Transpose");
    if (ex.kind === "Transpose") {
      expect(ex.value.kind).toBe("MatrixFromCols");
    }
  });
});

describe("legaliseTypes — WGSL Inverse helper", () => {
  it("Inverse rewrites to a call to a generated helper function", () => {
    const m: Expr = { kind: "ReadInput", scope: "Uniform", name: "u_m", type: Tmat4f };
    const body: Stmt = {
      kind: "Expression",
      value: { kind: "Inverse", value: m, type: Tmat4f },
    };
    const out = legaliseTypes(fragModule(body), "wgsl");
    // A helper Function should be present at module level.
    const fn = out.values.find((v) => v.kind === "Function" && v.signature.name === "_wombat_inverse4");
    expect(fn).toBeTruthy();
    // The Expression now uses CallIntrinsic.
    const entry = out.values.find((x) => x.kind === "Entry")!;
    if (entry.kind !== "Entry") throw new Error();
    const ex = (entry.entry.body as Extract<Stmt, { kind: "Expression" }>).value;
    expect(ex.kind).toBe("CallIntrinsic");
    if (ex.kind === "CallIntrinsic") {
      expect(ex.op.name).toBe("_wombat_inverse4");
    }
  });

  it("GLSL target leaves Inverse alone (built-in)", () => {
    const m: Expr = { kind: "ReadInput", scope: "Uniform", name: "u_m", type: Tmat4f };
    const body: Stmt = {
      kind: "Expression",
      value: { kind: "Inverse", value: m, type: Tmat4f },
    };
    const out = legaliseTypes(fragModule(body), "glsl");
    expect(out.values.some((v) => v.kind === "Function")).toBe(false);
    const entry = out.values.find((x) => x.kind === "Entry")!;
    if (entry.kind !== "Entry") throw new Error();
    const ex = (entry.entry.body as Extract<Stmt, { kind: "Expression" }>).value;
    expect(ex.kind).toBe("Inverse");
  });
});

describe("legaliseTypes — full GLSL+WGSL emit after legalisation", () => {
  it("MatrixRow lowering produces emittable code", () => {
    const m: Expr = { kind: "ReadInput", scope: "Uniform", name: "u_m", type: Tmat4f };
    const body: Stmt = {
      kind: "WriteOutput", name: "outColor",
      value: {
        kind: "Expr",
        value: { kind: "MatrixRow", matrix: m, row: constI(0), type: Tvec4f },
      },
    };
    const mod: Module = {
      types: [],
      values: [
        { kind: "Uniform", uniforms: [{ name: "u_m", type: Tmat4f }] },
        ...fragModule(body).values,
      ],
    };
    const glsl = emitGlsl(legaliseTypes(mod, "glsl")).source;
    expect(glsl).toContain("vec4(u_m[0][0], u_m[1][0], u_m[2][0], u_m[3][0])");
    const wgsl = emitWgsl(legaliseTypes(mod, "wgsl")).source;
    expect(wgsl).toContain("vec4<f32>(u_m[0i][0i], u_m[1i][0i], u_m[2i][0i], u_m[3i][0i])");
  });
});
