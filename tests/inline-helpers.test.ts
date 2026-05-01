// Plugin discovers and translates shader helper functions called
// from a marker arrow body.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";
import { effect, stage } from "@aardworx/wombat.shader";
import type { Module } from "@aardworx/wombat.shader/ir";

function extractTemplate(code: string): Module {
  const idx = code.indexOf("__wombat_stage(");
  let i = code.indexOf("{", idx);
  const start = i;
  let depth = 0;
  for (; i < code.length; i++) {
    if (code[i] === "{") depth++;
    else if (code[i] === "}") { depth--; if (depth === 0) { i++; break; } }
  }
  return JSON.parse(code.slice(start, i)) as Module;
}

describe("inline shader: helper functions", () => {
  it("translates a same-file helper called from a fragment body", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      function half(x: number): number { return x * 0.5; }
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(half(input.v_uv.x), half(input.v_uv.y), 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    // A Function ValueDef for `half` is emitted in the template.
    const fn = tmpl.values.find((v) => v.kind === "Function");
    expect(fn).toBeTruthy();
    if (fn && fn.kind === "Function") {
      expect(fn.signature.name).toBe("half");
      expect(fn.signature.parameters[0]?.type).toEqual({ kind: "Float", width: 32 });
    }
    // The body's Call references `half` (not Call(stub) — proper FunctionRef).
    const json = JSON.stringify(tmpl);
    expect(json).toContain('"id":"half"');
    expect(json).not.toMatch(/leaving as Call\(stub\)/);
  });

  it("transitively discovers helpers (helper calls another helper)", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      function half(x: number): number { return x * 0.5; }
      function quart(x: number): number { return half(half(x)); }
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(quart(input.v_uv.x), 0.5, 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    const fnNames = tmpl.values
      .filter((v) => v.kind === "Function")
      .map((v) => v.kind === "Function" ? v.signature.name : "");
    expect(fnNames).toContain("half");
    expect(fnNames).toContain("quart");
  });

  it("helper compiles end-to-end into WGSL with the helper function emitted", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      function half(x: number): number { return x * 0.5; }
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(half(input.v_uv.x), half(input.v_uv.y), 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    const fx = effect(stage(tmpl));
    const wgsl = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(wgsl).toContain("fn half(");
    expect(wgsl).toContain("half(");
  });

  it("`const fn = (…) => …` arrow helper works the same way", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const half = (x: number): number => x * 0.5;
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(half(input.v_uv.x), 0.5, 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    const fnNames = tmpl.values
      .filter((v) => v.kind === "Function")
      .map((v) => v.kind === "Function" ? v.signature.name : "");
    expect(fnNames).toContain("half");
  });
});
