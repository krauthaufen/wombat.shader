// `rule(...)` marker — pure-expression IR via the inline-marker pipeline.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";

describe("rule() marker", () => {
  it("rewrites a rule call into __wombat_rule(...)", () => {
    const src = `
      import { rule } from "@aardworx/wombat.shader";
      const r = rule(() => 42);
    `;
    const out = transformInlineShaders(src, "/x/app.ts");
    expect(out).not.toBeNull();
    expect(out!.code).toContain("__wombat_rule(");
    expect(out!.code).toContain('ruleExpr as __wombat_rule');
    // The original rule(...) call is replaced (no longer present).
    expect(out!.code).not.toMatch(/\brule\s*\(\s*\(\)/);
  });

  it("preserves the return expression — no graphics-stage output mangling", () => {
    // For shader stages, the return value gets lifted into stage
    // outputs (gl_Position / @location(0) / etc.). For rule, we want
    // the return to stay as the closure's literal return — the
    // consumer extracts it from the entry's body.
    const src = `
      import { rule } from "@aardworx/wombat.shader";
      const r = rule(() => 42);
    `;
    const out = transformInlineShaders(src, "/x/app.ts")!;
    // No "ReturnRecord" / synthetic carrier in the emitted JSON.
    expect(out.code).not.toMatch(/ReturnRecord|_record/);
    // The emitted JSON contains a ReturnValue with the literal 42.
    expect(out.code).toMatch(/"kind":"ReturnValue"[\s\S]*?"value":42/);
  });

  it("lifts object-literal returns into a serialisable `__record:...` CallIntrinsic", () => {
    // For graphics stages the carrier is post-processed into
    // WriteOutput statements; for rule we want the OBJECT itself to
    // survive to the runtime — heap scene fingerprints distinct
    // returns and uses them directly as pipeline-mode values.
    const src = `
      import { rule } from "@aardworx/wombat.shader";
      const r = rule(() => ({ srcFactor: 1, dstFactor: 2, op: 0 }));
    `;
    const out = transformInlineShaders(src, "/x/app.ts")!;
    // The lifted form encodes field names in the op name.
    expect(out.code).toMatch(/"name":"__record:srcFactor\|dstFactor\|op"/);
    // The three field VALUES become CallIntrinsic args (consts).
    expect(out.code).toMatch(/"value":1[^0-9]/);
    expect(out.code).toMatch(/"value":2[^0-9]/);
    // Carrier is NOT a bare Const(Null) anymore in the return.
    expect(out.code).not.toMatch(/"kind":"ReturnValue"[\s\S]*?"value":\{\s*"kind":"Null"/);
  });

  it("supports control flow in the body (if / locals)", () => {
    // The free-identifier path needs the TS type checker (resolver)
    // for ambient `declare const u` lookups — that's how the real
    // build-time setup runs. To exercise control-flow lowering in a
    // resolver-less unit test, keep the body self-contained.
    const src = `
      import { rule } from "@aardworx/wombat.shader";
      const r = rule(() => {
        let acc: number = 7;
        if (acc < 0) {
          acc = 0;
        }
        return acc;
      });
    `;
    const out = transformInlineShaders(src, "/x/app.ts")!;
    expect(out.code).toMatch(/"kind":"If"/);
  });
});
