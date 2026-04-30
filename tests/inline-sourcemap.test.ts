// The plugin returns a v3 source map alongside the transformed code
// so DevTools and runtime errors map back to the user's TS source.

import { describe, expect, it } from "vitest";
import { transformInlineShaders } from "@aardworx/wombat.shader-vite";

describe("inline plugin source map", () => {
  it("transforms produce a v3 source map when at least one marker is rewritten", () => {
    const src = `
      import { fragment } from "@aardworx/wombat.shader-runtime";
      const fs = fragment((input: { v_uv: V2f }) => ({
        outColor: new V4f(input.v_uv.x, input.v_uv.y, 0.5, 1.0),
      }));
    `;
    const r = transformInlineShaders(src, "/x/app.ts");
    expect(r).not.toBeNull();
    expect(r!.map).not.toBeNull();
    const map = r!.map as { version: number; sources: string[]; mappings: string };
    expect(map.version).toBe(3);
    expect(map.sources).toContain("/x/app.ts");
    expect(typeof map.mappings).toBe("string");
    expect(map.mappings.length).toBeGreaterThan(0);
  });

  it("no map when no marker is rewritten", () => {
    const r = transformInlineShaders(`console.log("hi");`, "/x/app.ts");
    expect(r).toBeNull();
  });
});
