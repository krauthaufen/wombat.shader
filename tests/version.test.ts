// VERSION (src/version.ts, written by scripts/sync-version.mjs on build)
// matches the package version: consumers key caches of compiled shaders on it.
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import { VERSION } from "@aardworx/wombat.shader";

describe("VERSION", () => {
  it("is the package.json version", () => {
    const pkg = JSON.parse(fs.readFileSync(new URL("../packages/shader/package.json", import.meta.url), "utf8"));
    expect(VERSION).toBe(pkg.version);
  });
});
