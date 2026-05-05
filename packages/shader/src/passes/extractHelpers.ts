// extractHelpers — convert each fused stage entry into a sequence of
// helper Function calls plus a thin wrapper Entry.
//
// Used by `composeStages.fusePair`: when two same-stage entries are
// fused, naively concatenating their bodies breaks once either side
// uses imperative `if (cond) return X;` — A's `return` short-circuits
// B. The fix is to wrap each effect in its own function: WGSL/GLSL
// `return;` exits only that helper, the wrapper calls them in order,
// and B always gets to run.
//
// We use the **merged-state model**:
//
//   struct State { /* every input + every output across all helpers */ };
//
//   fn _h_A(state_in: State) -> State {
//     var out: State = state_in;
//     // body — `WriteOutput(X, v)` → `out.X = v`,
//     //         `ReadInput("Input", X)` → `out.X`
//     //         (because `out` was init'd from the merged state),
//     //         `Return` (lifted by liftReturns) → `return out;`
//     return out;  // synth tail
//   }
//
//   fn _h_B(state_in: State) -> State { … }
//
//   @vertex fn entry(in: WrapperIn) -> WrapperOut {
//     var s: State;
//     // Initialise state fields from wrapper inputs (entry attrs).
//     s.Positions = in.Positions; …
//     s = _h_A(s);
//     s = _h_B(s);
//     var out: WrapperOut;
//     out.gl_Position = s.gl_Position; …
//     return out;
//   }
//
// One State struct holds every "port" — wrapper inputs, surfaced
// outputs, and intra-stage carriers (a name written by helper A and
// read by helper B). Cross-helper DCE on the wrapper backward-
// propagates field reads to mark which State fields are live; the
// linker pass (`linkHelpers`) then drops dead WriteOutputs from
// helper bodies, removes dead State fields, and rebuilds a tightest-
// possible State struct + helper signatures.
//
// This pass only does the structural extraction. The follow-up
// `linkHelpers` pass handles liveness and pruning.

import type {
  EntryDef,
  EntryParameter,
  Expr,
  FunctionRef,
  Module,
  Stmt,
  StructField,
  Type,
  TypeDef,
  ValueDef,
  Var,
} from "../ir/index.js";
import { Tvoid, hashValue } from "../ir/index.js";
import { readInputs } from "./analysis.js";

export interface ExtractedFusion {
  /** State struct TypeDef to add to the module. */
  readonly state: TypeDef;
  /** Helper Function ValueDefs — one per fused operand, in order. */
  readonly helpers: readonly ValueDef[];
  /** Wrapper EntryDef (with body that calls helpers). */
  readonly wrapperEntry: EntryDef;
}

/**
 * Build a fused entry from N same-stage operands using the merged-
 * state model. Caller is responsible for adding `result.state` to
 * `module.types` and `result.helpers` to `module.values` (and using
 * `result.wrapperEntry` as the new entry).
 */
export function extractFusedEntry(
  operands: readonly EntryDef[],
  fusedName: string,
): ExtractedFusion {
  if (operands.length === 0) {
    throw new Error("extractFusedEntry: at least one operand required");
  }
  if (!operands.every((o) => o.stage === operands[0]!.stage)) {
    throw new Error("extractFusedEntry: all operands must share a stage");
  }
  const stage = operands[0]!.stage;

  // Merged ports: every input + every output across all operands,
  // keyed by name. Conflicts on name with different types throw —
  // a real type-check guard. Builtins (entry-level @builtin params)
  // stay outside the State struct; they're surfaced as wrapper-
  // entry params and forwarded into State if any helper reads them.
  const ports = new Map<string, { type: Type; isBuiltin: boolean }>();
  const recordPort = (p: EntryParameter): void => {
    const isBuiltin = (p.decorations ?? []).some((d) => d.kind === "Builtin");
    const prev = ports.get(p.name);
    if (prev) {
      if (!sameType(prev.type, p.type)) {
        throw new Error(
          `extractFusedEntry: port "${p.name}" has conflicting types ` +
          `across operands (${typeKindLabel(prev.type)} vs ${typeKindLabel(p.type)})`,
        );
      }
      // First operand wins on isBuiltin classification (consistent
      // with last-wins semantics elsewhere).
      return;
    }
    ports.set(p.name, { type: p.type, isBuiltin });
  };
  for (const op of operands) {
    for (const p of op.inputs) recordPort(p);
    for (const p of op.outputs) recordPort(p);
  }

  // Builtin ports stay outside State (they're @builtin params on the
  // wrapper). A helper that reads/writes a builtin gets its body
  // rewritten to read from / write to the wrapper's bookkeeping.
  // For now we surface builtins as State fields too (simpler) and
  // let downstream DCE drop them if unused; the wrapper just copies
  // them in/out.
  // TODO(linker): drop dead State fields after liveness analysis.

  const stateName = `${capitalise(fusedName)}State`;
  const state: TypeDef = {
    kind: "Struct",
    name: stateName,
    fields: [...ports.entries()].map(([name, info]): StructField => ({
      name,
      type: info.type,
    })),
  };
  const stateType: Type = {
    kind: "Struct", name: stateName, fields: state.fields,
  };

  // Build a helper Function per operand. The helper takes the merged
  // State as its only parameter and returns the (mutated) State.
  // Body is the operand's body verbatim — emit handles the
  // ReadInput/WriteOutput/Return rewriting via the
  // `merged_state_helper` attribute.
  const helpers: ValueDef[] = [];
  const helperRefs: FunctionRef[] = [];
  operands.forEach((op, i) => {
    const helperName = `_${fusedName}_${op.name}_${i}`;
    const stateParam: Var = {
      name: "s_in", type: stateType, mutable: false,
    };
    const helperBody: Stmt = op.body;
    const helperFn: ValueDef = {
      kind: "Function",
      signature: {
        name: helperName,
        returnType: stateType,
        parameters: [{
          name: stateParam.name,
          type: stateType,
          modifier: "in",
        }],
      },
      body: helperBody,
      attributes: ["merged_state_helper"],
    };
    helpers.push(helperFn);
    helperRefs.push({
      id: `helper:${helperName}:${hashValue({ stage, op: op.name })}`,
      signature: helperFn.signature,
      pure: false,
    });
  });

  // Wrapper outputs: a name X surfaces iff its LAST writer is the
  // final consumer in the chain — no operand strictly AFTER the last
  // writer reads X as an input. Otherwise X is a carrier: produced
  // by the last writer (or earlier), consumed within the fused
  // block, never observed by the wrapper.
  //
  // FShade semantics: a `B`-wins overwrite of an A-output makes the
  // pre-write value a carrier, but B's own write surfaces (it's the
  // last writer and B's body is self-consistent).
  // Per-operand actual body reads (what the body's `ReadInput("Input",
  // X)` calls reference). Declared inputs that the body never reads
  // don't count as consumption — that matches the legacy `fusePair`
  // which scanned the body rather than trusting `b.inputs`.
  const bodyReads: Set<string>[] = operands.map((o) => {
    const r = new Set<string>();
    for (const sn of readInputs(o.body).values()) {
      if (sn.scope === "Input") r.add(sn.name);
    }
    return r;
  });

  const lastWriter = new Map<string, number>();
  operands.forEach((o, i) => {
    for (const p of o.outputs) lastWriter.set(p.name, i);
  });
  const wrapperOutputs: EntryParameter[] = [];
  const seenOut = new Set<string>();
  operands.forEach((o, i) => {
    for (const p of o.outputs) {
      if (lastWriter.get(p.name) !== i) continue; // overwritten downstream
      // Is X consumed strictly after this writer?
      let consumedAfter = false;
      for (let j = i + 1; j < operands.length; j++) {
        if (bodyReads[j]!.has(p.name)) {
          consumedAfter = true;
          break;
        }
      }
      if (consumedAfter) continue; // demote to carrier
      if (seenOut.has(p.name)) continue;
      seenOut.add(p.name);
      wrapperOutputs.push(p);
    }
  });

  // Wrapper inputs: an input "matters" only if (a) the operand's
  // body actually reads it AND (b) no strictly upstream operand
  // wrote that name. Otherwise the reader picks it up from the
  // State struct, which is what the helper-call chain flows through
  // (and unread declared inputs don't surface either — matches the
  // legacy fusePair, which scanned the body instead of trusting
  // declared inputs).
  const writtenSoFar = new Set<string>();
  const wrapperInputs: EntryParameter[] = [];
  const seenIn = new Set<string>();
  operands.forEach((o, i) => {
    for (const inp of o.inputs) {
      if (writtenSoFar.has(inp.name)) continue;
      if (!bodyReads[i]!.has(inp.name)) continue;
      if (seenIn.has(inp.name)) continue;
      seenIn.add(inp.name);
      wrapperInputs.push(inp);
    }
    for (const p of o.outputs) writtenSoFar.add(p.name);
  });

  // Wrapper body: declare `var s: State;`, copy each wrapper input
  // into `s`, call helpers in order assigning the result back to s,
  // copy each wrapper output from `s.field`.
  const stateLocal: Var = { name: "s", type: stateType, mutable: true };
  const stmts: Stmt[] = [
    { kind: "Declare", var: stateLocal },
  ];
  // Init state from wrapper inputs (ReadInput("Input", X) → in.X
  // emits as expected; we just write s.X = in.X).
  for (const inP of wrapperInputs) {
    if ((inP.decorations ?? []).some((d) => d.kind === "Builtin")) {
      // Builtins are read via the entry's @builtin params, which the
      // emitter exposes as bare identifier reads. Fall back to a
      // ReadInput("Builtin", semantic) which the emitter resolves.
      stmts.push({
        kind: "Write",
        target: { kind: "LField", target: { kind: "LVar", var: stateLocal, type: stateType }, name: inP.name, type: inP.type },
        value: {
          kind: "ReadInput", scope: "Builtin",
          name: inP.semantic, type: inP.type,
        },
      });
      continue;
    }
    stmts.push({
      kind: "Write",
      target: { kind: "LField", target: { kind: "LVar", var: stateLocal, type: stateType }, name: inP.name, type: inP.type },
      value: {
        kind: "ReadInput", scope: "Input",
        name: inP.name, type: inP.type,
      },
    });
  }
  // Call each helper.
  for (const fn of helperRefs) {
    stmts.push({
      kind: "Write",
      target: { kind: "LVar", var: stateLocal, type: stateType },
      value: {
        kind: "Call",
        fn,
        args: [{ kind: "Var", var: stateLocal, type: stateType }],
        type: stateType,
      },
    });
  }
  // Surface each wrapper output: WriteOutput(X, s.X).
  for (const outP of wrapperOutputs) {
    stmts.push({
      kind: "WriteOutput",
      name: outP.name,
      value: {
        kind: "Expr",
        value: {
          kind: "Field",
          target: { kind: "Var", var: stateLocal, type: stateType },
          name: outP.name,
          type: outP.type,
        },
      },
    });
  }

  const wrapperEntry: EntryDef = {
    name: fusedName,
    stage,
    inputs: wrapperInputs,
    outputs: wrapperOutputs,
    arguments: operands[0]!.arguments,
    returnType: Tvoid,
    decorations: operands[0]!.decorations,
    body: { kind: "Sequential", body: stmts },
  };

  return { state, helpers, wrapperEntry };
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function capitalise(s: string): string {
  return s.length > 0 ? s[0]!.toUpperCase() + s.slice(1) : s;
}

function sameType(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "Vector" && b.kind === "Vector") {
    return a.dim === b.dim && sameType(a.element, b.element);
  }
  if (a.kind === "Matrix" && b.kind === "Matrix") {
    return a.rows === b.rows && a.cols === b.cols && sameType(a.element, b.element);
  }
  if (a.kind === "Float" && b.kind === "Float") return a.width === b.width;
  if (a.kind === "Int" && b.kind === "Int") {
    return a.signed === b.signed && a.width === b.width;
  }
  return a.kind === b.kind;
}

function typeKindLabel(t: Type): string {
  if (t.kind === "Vector") return `vec${t.dim}<${typeKindLabel(t.element)}>`;
  if (t.kind === "Matrix") return `mat${t.cols}x${t.rows}<${typeKindLabel(t.element)}>`;
  return t.kind;
}

function mergeParams(xs: readonly EntryParameter[]): EntryParameter[] {
  const seen = new Map<string, EntryParameter>();
  for (const p of xs) {
    // Last-wins on type / decorations (matches existing fusePair).
    seen.set(p.name, p);
  }
  return [...seen.values()];
}

/**
 * Standalone: extract a single Entry into a helper + thin wrapper.
 * Useful as a unit-test fixture for the helper machinery before
 * folding into composeStages.
 */
export function extractSingleEntry(entry: EntryDef): {
  state: TypeDef; helpers: readonly ValueDef[]; wrapperEntry: EntryDef;
} {
  return extractFusedEntry([entry], entry.name);
}
