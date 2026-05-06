// Type-level + runtime checks for `Semantic<T, N>` / `Builtin<T, K>`
// brands and their no-op constructors.

import { describe, expect, it } from "vitest";
import {
  BUILTIN_SLOTS,
  Color,
  ClipPosition,
  Depth,
  FragCoord,
  FragDepth,
  FrontFacing,
  GlobalInvocationId,
  LocalInvocationId,
  Normal,
  Position,
  Texcoord,
  VertexIndex,
  isBuiltinAllowed,
  type ClipPosition as ClipPositionT,
  type Color as ColorT,
  type FragCoord as FragCoordT,
  type Position as PositionT,
} from "../packages/shader/src/types/semantic.js";
import { V2f, V3f, V4f, V3ui } from "@aardworx/wombat.base";

describe("Semantic / Builtin runtime constructors", () => {
  it("constructors are no-op identity casts (preserve reference)", () => {
    const v3 = new V3f(1, 2, 3);
    const v4 = new V4f(0, 0, 0, 1);

    expect(Position(v4)).toBe(v4);
    expect(Normal(v3)).toBe(v3);
    expect(Color(v4)).toBe(v4);
    expect(FragCoord(v4)).toBe(v4);
    expect(ClipPosition(v4)).toBe(v4);
  });

  it("constructors return the same object instance — no allocation", () => {
    const c = new V4f(0.5, 0.5, 0.5, 1);
    const branded = Color(c);
    expect(branded).toBe(c);
    // The brand erases at runtime: branded.x still works.
    expect(branded.x).toBe(0.5);
  });

  it("scalar-valued brands work on plain numbers", () => {
    const d = Depth(0.7);
    expect(d).toBe(0.7);
    const fd = FragDepth(0.3);
    expect(fd).toBe(0.3);
    const ff = FrontFacing(true);
    expect(ff).toBe(true);
    const idx = VertexIndex(42);
    expect(idx).toBe(42);
  });

  it("compute builtins default to V3ui at the type level", () => {
    const id = new V3ui(0, 0, 0);
    const local = LocalInvocationId(id);
    expect(local).toBe(id);
    const global = GlobalInvocationId(id);
    expect(global).toBe(id);
  });

  // Type-level check (compile-only): if any of these break, the
  // build fails, the test never runs.
  it("type-level: branded types unify with their underlying T at .x", () => {
    const _v3 = new V3f(1, 2, 3);
    const _v4 = new V4f(0, 0, 0, 1);
    const p: PositionT<V3f> = Position(_v3);
    const c: ColorT<V4f> = Color(_v4);
    const cp: ClipPositionT<V4f> = ClipPosition(_v4);
    const fc: FragCoordT<V4f> = FragCoord(_v4);
    expect(p.x).toBe(1);
    expect(c.w).toBe(1);
    expect(cp.w).toBe(1);
    expect(fc.w).toBe(1);
  });

  it("supports Texcoord on V2f", () => {
    const uv = new V2f(0.25, 0.75);
    const t = Texcoord(uv);
    expect(t).toBe(uv);
    expect(t.x).toBe(0.25); expect(t.y).toBe(0.75);
  });

  it("`as`-cast form works alongside the constructor form", () => {
    // Both forms compile, both are zero-runtime — `as` is the
    // pure type-assertion path, the constructor is the same thing
    // at the value level (with arg-inferred generic).
    const v3 = new V3f(1, 2, 3);
    const v4 = new V4f(0, 0, 0, 1);

    // Cast form — explicit generic when the arg's type doesn't
    // match the alias's default `T`.
    const p1 = v3 as PositionT<V3f>;
    expect(p1).toBe(v3);
    expect(p1.x).toBe(1);

    // Constructor form — generic inferred from arg.
    const p2 = Position(v3);
    expect(p2).toBe(v3);

    // Default-generic cast form — works when arg type matches the
    // default. `Position` defaults to `Position<V4f>`.
    const p3 = v4 as PositionT;
    expect(p3).toBe(v4);
    expect(p3.w).toBe(1);

    // FragCoord too.
    const fc1 = v4 as FragCoordT<V4f>;
    const fc2 = FragCoord(v4);
    expect(fc1).toBe(v4);
    expect(fc2).toBe(v4);
  });
});

describe("BUILTIN_SLOTS / isBuiltinAllowed registry", () => {
  it("position is legal as VS output AND FS input", () => {
    expect(isBuiltinAllowed("position", "vertex", "out")).toBe(true);
    expect(isBuiltinAllowed("position", "fragment", "in")).toBe(true);
  });
  it("position is NOT legal as FS output", () => {
    expect(isBuiltinAllowed("position", "fragment", "out")).toBe(false);
  });
  it("vertex_index is legal only on VS input", () => {
    expect(isBuiltinAllowed("vertex_index", "vertex", "in")).toBe(true);
    expect(isBuiltinAllowed("vertex_index", "fragment", "in")).toBe(false);
    expect(isBuiltinAllowed("vertex_index", "vertex", "out")).toBe(false);
  });
  it("frag_depth is FS output only", () => {
    expect(isBuiltinAllowed("frag_depth", "fragment", "out")).toBe(true);
    expect(isBuiltinAllowed("frag_depth", "fragment", "in")).toBe(false);
    expect(isBuiltinAllowed("frag_depth", "vertex", "out")).toBe(false);
  });
  it("sample_mask is FS in AND out (the only dual-direction builtin)", () => {
    expect(isBuiltinAllowed("sample_mask", "fragment", "in")).toBe(true);
    expect(isBuiltinAllowed("sample_mask", "fragment", "out")).toBe(true);
    expect(isBuiltinAllowed("sample_mask", "vertex", "in")).toBe(false);
  });
  it("compute builtins are compute-input only", () => {
    for (const k of ["local_invocation_id", "local_invocation_index",
                     "global_invocation_id", "workgroup_id", "num_workgroups"]) {
      expect(isBuiltinAllowed(k, "compute", "in")).toBe(true);
      expect(isBuiltinAllowed(k, "fragment", "in")).toBe(false);
    }
  });
  it("unknown builtin names are conservatively allowed everywhere", () => {
    // Forward-compat with WGSL extensions the registry doesn't list yet.
    expect(isBuiltinAllowed("future_extension_builtin", "vertex", "in")).toBe(true);
    expect(isBuiltinAllowed("future_extension_builtin", "fragment", "out")).toBe(true);
  });

  it("registry covers every Builtin alias's `K`", () => {
    // Spot-check that the names declared on the type aliases exist
    // in the registry.
    const expected = [
      "position", "vertex_index", "instance_index",
      "front_facing", "sample_index", "sample_mask", "frag_depth",
      "local_invocation_id", "local_invocation_index",
      "global_invocation_id", "workgroup_id", "num_workgroups",
    ];
    for (const k of expected) {
      expect(BUILTIN_SLOTS[k]).toBeDefined();
    }
  });
});
