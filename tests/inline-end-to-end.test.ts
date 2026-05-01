// End-to-end: extract the IR template the plugin emitted, hand it to
// the runtime stage/effect API, supply the closure value, and check
// the compiled WGSL.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";
import { effect, stage } from "@aardworx/wombat.shader";
import type { Module } from "@aardworx/wombat.shader/ir";

/** Pull the JSON template out of a `__wombat_stage(<template>, ...)` call site. */
function extractTemplate(code: string): Module {
  const idx = code.indexOf("__wombat_stage(");
  if (idx < 0) throw new Error("no __wombat_stage call found");
  // Walk balanced braces from the first `{` after the call name.
  let i = code.indexOf("{", idx);
  if (i < 0) throw new Error("no JSON in __wombat_stage call");
  const start = i;
  let depth = 0;
  for (; i < code.length; i++) {
    if (code[i] === "{") depth++;
    else if (code[i] === "}") {
      depth--;
      if (depth === 0) { i++; break; }
    }
  }
  return JSON.parse(code.slice(start, i)) as Module;
}

describe("inline shader: end-to-end via Stage runtime", () => {
  it("fragment with V3f closure capture compiles to WGSL with hole values inlined", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const tint: V3f = new V3f(0.25, 0.5, 0.75);
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    // Closure scope reference present in the template's IR JSON.
    expect(JSON.stringify(tmpl)).toContain('"scope":"Closure"');

    // Hand the template to the runtime stage/effect path with a value
    // that matches the V3f shape.
    const fx = effect(stage(tmpl, { tint: () => [0.25, 0.5, 0.75] }));
    const compiled = fx.compile({ target: "wgsl" });
    const wgsl = compiled.stages[0]!.source;
    expect(wgsl).not.toContain("tint");
    expect(wgsl).toMatch(/0\.25/);
    expect(wgsl).toMatch(/0\.5/);
    expect(wgsl).toMatch(/0\.75/);
  });

  it("re-sampling the getter on each compile picks up new values", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader";
      const tint: V3f = new V3f(1, 0, 0);
      const fs = fragment(() => ({
        outColor: new V4f(tint.x, tint.y, tint.z, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts")!;
    const tmpl = extractTemplate(r.code);
    const ref = { tint: [1, 0, 0] as [number, number, number] };
    const fx = effect(stage(tmpl, { tint: () => ref.tint }));

    const a = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(a).toMatch(/1\.0f?,\s*0\.0f?,\s*0\.0f?/);

    ref.tint = [0, 1, 0];
    const b = fx.compile({ target: "wgsl" }).stages[0]!.source;
    expect(b).toMatch(/0\.0f?,\s*1\.0f?,\s*0\.0f?/);
  });
});
