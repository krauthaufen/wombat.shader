// instanceUniforms — port of `Aardvark.SceneGraph.Semantics.Instancing.
// Effect.inlineTrafo`. Rewrites named uniform reads in a Module's
// vertex (and, where needed, fragment) entries so the named values
// come from per-instance vertex attributes instead of bind-group
// uniforms. Driven by an explicit `attrNames` set: scene-graph code
// supplies the names, this pass produces a rewritten Module.
//
// Special cases for trafo names (taken straight from Aardvark):
//
//  - `ModelTrafo` / `ModelViewTrafo` / `ModelViewProjTrafo`
//      → `uniform.X * input.InstanceTrafo`
//  - `ModelTrafoInv` / `ModelViewTrafoInv` / `ModelViewProjTrafoInv`
//      → `input.InstanceTrafoInv * uniform.X`
//  - `NormalMatrix`
//      → `uniform.NormalMatrix * transpose(m33(input.InstanceTrafoInv))`
//
// Trafo rewrites trigger only when `attrNames` contains the literal
// `"ModelTrafo"`. The runtime is expected to:
//  (1) pre-multiply the scope-accumulated `ModelTrafo` into each
//      `InstanceTrafo[i]` (and the inverses) on the CPU, and
//  (2) bind the leaf's `ModelTrafo` uniform to identity so the
//      `uniform.X * InstanceTrafo` product yields the expected value.
//
// Anything else in `attrNames` is rewritten as a plain
// `ReadInput("Uniform", X)` → `ReadInput("Input", X)`.
//
// FS reads of any rewritten name get a flat-interpolated varying
// synthesised in the VS so the per-primitive value is constant
// across the triangle.

import type {
  EntryDef, EntryParameter, Expr, Module, ParamDecoration, Stmt, Type, ValueDef,
} from "../ir/index.js";
import { Mat, Tf32, Vec } from "../ir/index.js";
import { mapStmt } from "./transform.js";

const FORWARD_TRAFOS = new Set(["ModelTrafo", "ModelViewTrafo", "ModelViewProjTrafo"]);
const INVERSE_TRAFOS = new Set(["ModelTrafoInv", "ModelViewTrafoInv", "ModelViewProjTrafoInv"]);

const TM44f: Type = Mat(Tf32, 4, 4);
const TM33f: Type = Mat(Tf32, 3, 3);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const Ti32:   Type = { kind: "Int", signed: true, width: 32 };

const INSTANCE_TRAFO     = "InstanceTrafo";
const INSTANCE_TRAFO_INV = "InstanceTrafoInv";

// WebGPU vertex attributes must be scalars or vec2/3/4 — matrices
// aren't allowed. A mat4x4 instance attribute therefore has to be
// uploaded as 4 contiguous vec4 columns; the shader reconstructs the
// matrix at the read site via `MatrixFromCols`. Naming convention:
// `<name>` → `_<name>_col0 .. _<name>_col3`.
const matCol = (matName: string, col: 0 | 1 | 2 | 3): string => `_${matName}_col${col}`;

const constI32 = (i: number): Expr => ({
  kind: "Const", value: { kind: "Int", signed: true, value: i }, type: Ti32,
});

interface RewriteRule {
  /** The expression that replaces `ReadInput("Uniform", name)`. */
  readonly buildExpr: (forStage: "vertex" | "fragment") => Expr;
  /** Type of the rewritten expression — matches the original uniform's type. */
  readonly type: Type;
}

/**
 * Rewrite the module so that every read of a uniform whose name is
 * in `attrNames` (plus the trafo aliases when `"ModelTrafo"` is in
 * the set) comes from a per-instance vertex attribute instead.
 *
 * `uniformTypes` carries the IR type for each name. Required for
 * plain (non-trafo) names — the new VS input parameter needs the
 * same type as the uniform it replaces.
 */
export function instanceUniforms(
  module: Module,
  attrNames: ReadonlySet<string>,
  uniformTypes: ReadonlyMap<string, Type>,
): Module {
  if (attrNames.size === 0) return module;

  const hasTrafo = attrNames.has("ModelTrafo");

  // Collect names that the rewrite actually replaces. Plain names
  // come straight from `attrNames` (sans "ModelTrafo"). Trafo aliases
  // are added when hasTrafo.
  const plainNames = new Set<string>();
  for (const n of attrNames) {
    if (hasTrafo && (FORWARD_TRAFOS.has(n) || INVERSE_TRAFOS.has(n) || n === "NormalMatrix")) continue;
    plainNames.add(n);
  }
  const trafoAliases = new Set<string>();
  if (hasTrafo) {
    for (const n of FORWARD_TRAFOS) trafoAliases.add(n);
    for (const n of INVERSE_TRAFOS) trafoAliases.add(n);
    trafoAliases.add("NormalMatrix");
  }

  // ─── helpers to build IR exprs ────────────────────────────────────

  const readUniform = (name: string, type: Type): Expr => ({
    kind: "ReadInput", scope: "Uniform", name, type,
  });
  const readInput = (name: string, type: Type): Expr => ({
    kind: "ReadInput", scope: "Input", name, type,
  });
  const mulMatMat = (a: Expr, b: Expr): Expr => ({
    kind: "MulMatMat", lhs: a, rhs: b, type: a.type,
  });
  const m33OfM44 = (m: Expr): Expr => {
    // WGSL doesn't have a `mat3x3(mat4x4)` constructor. Build from
    // the upper-3 of each column: `mat3x3(m[0].xyz, m[1].xyz, m[2].xyz)`.
    if (m.type.kind !== "Matrix" || m.type.rows !== 4 || m.type.cols !== 4) {
      return { kind: "ConvertMatrix", value: m, type: TM33f };
    }
    const col = (i: number): Expr => ({
      kind: "VecSwizzle",
      value: { kind: "MatrixCol", matrix: m, col: { kind: "Const", value: { kind: "Int", signed: true, value: i }, type: { kind: "Int", signed: true, width: 32 } }, type: Vec(Tf32, 4) },
      comps: ["x", "y", "z"] as const,
      type: Vec(Tf32, 3),
    });
    return { kind: "MatrixFromCols", cols: [col(0), col(1), col(2)], type: TM33f };
  };
  // Reconstruct a mat4x4 instance attribute from the four vec4 column
  // attributes the runtime uploaded. The CPU packs each M44 row-major
  // (`_data[r*4+c]`), so reading offsets 0/16/32/48 as vec4 attributes
  // pulls the four ROWS of M_cpu. We want the GPU-seen matrix to be
  // `transpose(M_cpu)` — same convention every uniform matrix follows
  // here, so `reverseMatrixOps`-rewritten downstream math composes
  // correctly. WGSL `mat4x4(rows)` with cols=rows yields exactly
  // `transpose(M_cpu)`. Emit-time we need the literal
  // `mat4x4(row0..row3)` (no extra transpose), which means the IR has
  // to use `MatrixFromRows` here — `reverseMatrixOps` then flips it to
  // `MatrixFromCols`, which emits without a transpose. (Using
  // `MatrixFromCols` directly would flip to `MatrixFromRows` and emit
  // `transpose(...)`, leaving the GPU seeing M_cpu instead — the
  // mismatch that produced screen-spanning triangles when composed
  // with uniform matrices that *do* live in transposed form.)
  const m44FromCols = (matName: string): Expr => ({
    kind: "MatrixFromRows",
    rows: [0, 1, 2, 3].map((i) => readInput(matCol(matName, i as 0 | 1 | 2 | 3), Tvec4f)),
    type: TM44f,
  });
  // Pad an M33 to an M44 by zero-extending each column and appending
  // a zero column (`mat4x4(vec4(m[0], 0), vec4(m[1], 0), vec4(m[2], 0),
  // vec4(0))`). Used when the source uniform was declared M44 (e.g.
  // wombat.shader's `UniformScope.NormalMatrix`) but the rewrite
  // computes a 3×3 normal-transform — for direction vectors with w=0
  // the padded M44 gives the same result.
  const const0: Expr = { kind: "Const", value: { kind: "Float", value: 0 }, type: Tf32 };
  const colInt = (i: number): Expr => ({ kind: "Const", value: { kind: "Int", signed: true, value: i }, type: Ti32 });
  const padM33toM44 = (m33: Expr): Expr => {
    const col4 = (i: 0 | 1 | 2): Expr => ({
      kind: "NewVector",
      components: [
        { kind: "MatrixCol", matrix: m33, col: colInt(i), type: Tvec3f },
        const0,
      ],
      type: Tvec4f,
    });
    const lastCol: Expr = {
      kind: "NewVector",
      components: [const0, const0, const0, const0],
      type: Tvec4f,
    };
    return { kind: "MatrixFromCols", cols: [col4(0), col4(1), col4(2), lastCol], type: TM44f };
  };

  // ─── per-name rewrite rules ───────────────────────────────────────

  const rules = new Map<string, RewriteRule>();
  for (const n of plainNames) {
    const t = uniformTypes.get(n);
    if (!t) {
      throw new Error(
        `instanceUniforms: missing IR type for uniform "${n}"; pass it in via uniformTypes.`,
      );
    }
    // mat4x4 attributes have to be split into four vec4 columns;
    // other matrix sizes aren't supported as vertex attributes
    // either, so reject them up front rather than emitting bad WGSL.
    if (t.kind === "Matrix") {
      if (t.rows !== 4 || t.cols !== 4 || t.element.kind !== "Float") {
        throw new Error(
          `instanceUniforms: matrix attribute "${n}" has unsupported shape ` +
          `${t.rows}x${t.cols}; only mat4x4<f32> is currently supported.`,
        );
      }
      rules.set(n, {
        type: t,
        buildExpr: () => m44FromCols(n),
      });
      continue;
    }
    rules.set(n, {
      type: t,
      buildExpr: () => readInput(n, t),
    });
  }
  if (hasTrafo) {
    for (const n of FORWARD_TRAFOS) {
      rules.set(n, {
        type: TM44f,
        buildExpr: () => mulMatMat(readUniform(n, TM44f), m44FromCols(INSTANCE_TRAFO)),
      });
    }
    for (const n of INVERSE_TRAFOS) {
      rules.set(n, {
        type: TM44f,
        buildExpr: () => mulMatMat(m44FromCols(INSTANCE_TRAFO_INV), readUniform(n, TM44f)),
      });
    }
    // `NormalMatrix` — Aardvark's
    // `uniform.NormalMatrix · transpose(m33(InstanceTrafoInv))`,
    // rebuilt entirely on the GPU. Only fires when user code reads
    // `uniform.NormalMatrix` explicitly — the default `trafo()` VS
    // takes the simpler `vec.mul(uniform.ModelTrafoInv)` row-vec
    // path (the upload trick gives the inverse-transpose for free),
    // so this rule is dead in the common case. We keep it for
    // backward-compat with user shaders that name `NormalMatrix`.
    //
    // Correctness depends on the parent-uniform-override in
    // `wombat.dom/scene/instancing.ts:applyInstancing`:
    // `uniform.NormalMatrix` at the leaf must carry the OUTER
    // (parent-of-instancing-scope) inv-transpose, not the inner
    // chain — otherwise the product `parentNM · transpose(m33(M_inst_inv))`
    // composes innerModel twice and drops the per-instance rotation.
    const nmDeclaredType: Type = uniformTypes.get("NormalMatrix") ?? TM33f;
    rules.set("NormalMatrix", {
      type: nmDeclaredType,
      buildExpr: () => {
        const nmExpr = readUniform("NormalMatrix", nmDeclaredType);
        const nmM33 = nmDeclaredType.kind === "Matrix" && nmDeclaredType.rows === 3 && nmDeclaredType.cols === 3
          ? nmExpr : m33OfM44(nmExpr);
        const instInvM33 = m33OfM44(m44FromCols(INSTANCE_TRAFO_INV));
        const transposed: Expr = { kind: "Transpose", value: instInvM33, type: TM33f };
        const rebuilt = mulMatMat(nmM33, transposed);
        return nmDeclaredType.kind === "Matrix" && nmDeclaredType.rows === 4 && nmDeclaredType.cols === 4
          ? padM33toM44(rebuilt) : rebuilt;
      },
    });
  }

  // ─── pass over each entry ─────────────────────────────────────────

  const newValues: ValueDef[] = [];

  // We need to know up-front which names the FS reads, so the VS can
  // emit synthesised varyings for them.
  const fsReadsByName = collectFsUniformReadsByName(module, rules);

  for (const v of module.values) {
    if (v.kind !== "Entry") {
      newValues.push(v);
      continue;
    }
    const e = v.entry;
    if (e.stage === "vertex") {
      newValues.push({ kind: "Entry", entry: rewriteVertex(e, rules, plainNames, hasTrafo, fsReadsByName) });
    } else if (e.stage === "fragment") {
      newValues.push({ kind: "Entry", entry: rewriteFragment(e, rules, fsReadsByName) });
    } else {
      newValues.push(v);
    }
  }

  return { ...module, values: newValues };
}

// ─────────────────────────────────────────────────────────────────────
// Vertex-stage rewrite
// ─────────────────────────────────────────────────────────────────────

function rewriteVertex(
  e: EntryDef,
  rules: ReadonlyMap<string, RewriteRule>,
  plainNames: ReadonlySet<string>,
  hasTrafo: boolean,
  fsReadsByName: ReadonlySet<string>,
): EntryDef {
  // Inline rewrite of each Uniform read.
  const body = mapStmt(e.body, {
    expr: (x) => x.kind === "ReadInput" && x.scope === "Uniform" && rules.has(x.name)
      ? rules.get(x.name)!.buildExpr("vertex")
      : x,
  });

  // Add VS inputs for every per-instance attribute we now read.
  const newInputs: EntryParameter[] = [...e.inputs];
  const existingInputNames = new Set(e.inputs.map((p) => p.name));
  const addInput = (name: string, type: Type): void => {
    if (existingInputNames.has(name)) return;
    newInputs.push({
      name, type, semantic: name, decorations: [{ kind: "Location", value: 0 }],
    });
    existingInputNames.add(name);
  };
  if (hasTrafo) {
    // mat4x4 isn't a legal vertex-attribute type in WebGPU, so each
    // per-instance trafo is uploaded as 4 vec4 columns. NormalMatrix
    // doesn't need its own buffer — the default `trafo()` VS reads
    // normals via `vec.mul(uniform.ModelTrafoInv)` which gives the
    // inv-transpose normal transform under the row/col-major
    // convention, and `ModelTrafoInv` is already rewritten via
    // INVERSE_TRAFOS to compose with InstanceTrafoInv.
    for (let i = 0; i < 4; i++) addInput(matCol(INSTANCE_TRAFO,     i as 0 | 1 | 2 | 3), Tvec4f);
    for (let i = 0; i < 4; i++) addInput(matCol(INSTANCE_TRAFO_INV, i as 0 | 1 | 2 | 3), Tvec4f);
  }
  for (const n of plainNames) {
    const r = rules.get(n);
    if (!r) continue;
    if (r.type.kind === "Matrix" && r.type.rows === 4 && r.type.cols === 4) {
      for (let i = 0; i < 4; i++) addInput(matCol(n, i as 0 | 1 | 2 | 3), Tvec4f);
    } else {
      addInput(n, r.type);
    }
  }

  // For every name the FS reads, synthesise flat-interpolated
  // varying(s) carrying the rewritten value. Vec / scalar types use
  // a single varying; matrix types split into one vec4 varying per
  // column because mat-typed varyings aren't legal in WGSL.
  const newOutputs: EntryParameter[] = [...e.outputs];
  const existingOutputNames = new Set(e.outputs.map((p) => p.name));
  // Pick the next free Location starting one above whatever the
  // existing outputs already pinned. Same approach `linkCrossStage`'s
  // `appendPositionPassThroughs` uses.
  let nextLoc = 0;
  for (const o of e.outputs) {
    for (const d of o.decorations) {
      if (d.kind === "Location" && d.value >= nextLoc) nextLoc = d.value + 1;
    }
  }
  const flatVarying = (name: string, type: Type): EntryParameter => ({
    name, type, semantic: name,
    decorations: [
      { kind: "Location", value: nextLoc++ },
      { kind: "Interpolation", mode: "flat" } as ParamDecoration,
    ],
  });
  const synthStmts: Stmt[] = [];
  for (const name of fsReadsByName) {
    const r = rules.get(name);
    if (!r) continue;
    const built = r.buildExpr("vertex");
    if (r.type.kind === "Matrix" && r.type.rows === 4 && r.type.cols === 4) {
      for (let i = 0; i < 4; i++) {
        const vname = varyingFor(`${name}_col${i}`);
        if (existingOutputNames.has(vname)) continue;
        newOutputs.push(flatVarying(vname, Tvec4f));
        existingOutputNames.add(vname);
        synthStmts.push({
          kind: "WriteOutput", name: vname,
          value: { kind: "Expr", value: { kind: "MatrixCol", matrix: built, col: constI32(i), type: Tvec4f } },
        });
      }
      continue;
    }
    const vname = varyingFor(name);
    if (existingOutputNames.has(vname)) continue;
    newOutputs.push(flatVarying(vname, r.type));
    existingOutputNames.add(vname);
    synthStmts.push({
      kind: "WriteOutput", name: vname,
      value: { kind: "Expr", value: built },
    });
  }
  const finalBody: Stmt = synthStmts.length === 0
    ? body
    : body.kind === "Sequential"
      ? { ...body, body: [...body.body, ...synthStmts] }
      : { kind: "Sequential", body: [body, ...synthStmts] };

  return { ...e, inputs: newInputs, outputs: newOutputs, body: finalBody };
}

// ─────────────────────────────────────────────────────────────────────
// Fragment-stage rewrite
// ─────────────────────────────────────────────────────────────────────

function rewriteFragment(
  e: EntryDef,
  rules: ReadonlyMap<string, RewriteRule>,
  fsReadsByName: ReadonlySet<string>,
): EntryDef {
  if (fsReadsByName.size === 0) return e;
  const body = mapStmt(e.body, {
    expr: (x) => {
      if (x.kind !== "ReadInput" || x.scope !== "Uniform" || !fsReadsByName.has(x.name)) return x;
      const r = rules.get(x.name)!;
      if (r.type.kind === "Matrix" && r.type.rows === 4 && r.type.cols === 4) {
        // Reconstruct from per-column varyings the VS exported.
        const cols: Expr[] = [];
        for (let i = 0; i < 4; i++) {
          cols.push({
            kind: "ReadInput", scope: "Input",
            name: varyingFor(`${x.name}_col${i}`),
            type: Tvec4f,
          });
        }
        return { kind: "MatrixFromCols", cols, type: r.type };
      }
      return { kind: "ReadInput", scope: "Input", name: varyingFor(x.name), type: r.type };
    },
  });
  // Add FS inputs (varyings) — single for non-matrix, four vec4
  // columns for matrix types.
  const newInputs: EntryParameter[] = [...e.inputs];
  const existingInputNames = new Set(e.inputs.map((p) => p.name));
  const addInput = (name: string, type: Type): void => {
    if (existingInputNames.has(name)) return;
    newInputs.push({
      name, type, semantic: name,
      // Drop the explicit Location — emit's auto-assign uses
      // declaration order, which keeps VS-output ↔ FS-input matched
      // when both sides see the same synth-varying list.
      decorations: [
        { kind: "Interpolation", mode: "flat" } as ParamDecoration,
      ],
    });
    existingInputNames.add(name);
  };
  for (const name of fsReadsByName) {
    const r = rules.get(name);
    if (!r) continue;
    if (r.type.kind === "Matrix" && r.type.rows === 4 && r.type.cols === 4) {
      for (let i = 0; i < 4; i++) addInput(varyingFor(`${name}_col${i}`), Tvec4f);
    } else {
      addInput(varyingFor(name), r.type);
    }
  }
  return { ...e, inputs: newInputs, body };
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function varyingFor(name: string): string {
  return `_inst_${name}`;
}

function collectFsUniformReadsByName(
  module: Module,
  rules: ReadonlyMap<string, RewriteRule>,
): Set<string> {
  const out = new Set<string>();
  for (const v of module.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    walkExprs(v.entry.body, (e) => {
      if (e.kind === "ReadInput" && e.scope === "Uniform" && rules.has(e.name)) out.add(e.name);
    });
  }
  return out;
}

function walkExprs(s: Stmt, fn: (e: Expr) => void): void {
  mapStmt(s, {
    expr: (e) => { fn(e); return e; },
  });
}
