// Smoke test: shader-types declarations type-check on user code that
// resembles real shader source. The test only inspects compile-time
// types; nothing in this file actually runs the declared functions.

import { describe, it, expect } from "vitest";
import {
  V2f, V3f, V4f, M44f,
  vec3, vec4,
  Sampler2D,
  texture, sin, mix, normalize, dot,
} from "@aardworx/wombat.shader-types";

describe("shader-types — type-only smoke", () => {
  it("user code referencing the declared classes type-checks", () => {
    // Pretend this is shader source the frontend would translate.
    function frag(uv: V2f, tex: Sampler2D, time: number): V4f {
      const base = texture(tex, uv);
      const wave = sin(time);
      const tinted = mix(base, vec4(wave, wave, wave, 1), 0.5);
      return tinted;
    }
    // The `frag` function above must type-check. We can't call it
    // without crashing (declared-only types), but TypeScript's
    // structural checks have already validated the signature.
    expect(typeof frag).toBe("function");
  });

  it("intrinsics generic over vector types", () => {
    function ndc(p: V3f): V3f {
      const n = normalize(p);
      const d = dot(n, vec3(0, 0, 1));
      return n.mul(d);
    }
    expect(typeof ndc).toBe("function");
  });

  it("M44f * V4f is a method, not an operator", () => {
    function project(m: M44f, p: V4f): V4f {
      return m.mul(p);
    }
    expect(typeof project).toBe("function");
  });
});
