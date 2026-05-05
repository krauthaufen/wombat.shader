// Human-readable IR dump. Used during pass development and from
// `Effect.dumpIR()` for runtime introspection.
//
// Format is intentionally flat / FShade-ish: each `ValueDef` prints
// with a header line + indented body, each `Stmt` on its own line,
// `Expr`s inline with type annotations only where they're load-bearing
// (the type checker isn't reading this — humans are).
//
// Coverage: every common IR node renders concretely. Less-common kinds
// fall through to a generic `(<kind> …)` form so the output is always
// readable even if not perfectly tuned.

import type {
  Expr,
  LExpr,
  Literal,
  Module,
  RExpr,
  Stmt,
  Type,
  ValueDef,
} from "./types.js";

export function prettyPrint(module: Module): string {
  const lines: string[] = [];
  lines.push("module {");
  for (const v of module.values) {
    for (const l of valueDef(v, 1)) lines.push(l);
    lines.push("");
  }
  // trim trailing blank
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  lines.push("}");
  return lines.join("\n");
}

// ─────────────────────────────────────────────────────────────────────

function valueDef(v: ValueDef, indent: number): string[] {
  const pad = "  ".repeat(indent);
  switch (v.kind) {
    case "Constant":
      return [`${pad}const ${v.name} : ${type(v.varType)} = ${rExpr(v.init)}`];
    case "Function": {
      const sig = v.signature;
      const params = sig.parameters.map((p) => `${p.name}: ${type(p.type)}`).join(", ");
      const out: string[] = [`${pad}fn ${sig.name}(${params}) -> ${type(sig.returnType)} {`];
      for (const l of stmt(v.body, indent + 1)) out.push(l);
      out.push(`${pad}}`);
      return out;
    }
    case "Entry": {
      const e = v.entry;
      const inputs = e.inputs.map((p) => `${p.name}: ${type(p.type)}`).join(", ");
      const outputs = e.outputs.map((p) => `${p.name}: ${type(p.type)}`).join(", ");
      const args = e.arguments.map((p) => `${p.name}: ${type(p.type)}`).join(", ");
      const decos = e.decorations.length === 0 ? ""
        : " [" + e.decorations.map(decoration).join(", ") + "]";
      const out: string[] = [];
      out.push(`${pad}entry ${e.stage} ${e.name}${decos}`);
      if (inputs)  out.push(`${pad}  inputs:    (${inputs})`);
      if (args)    out.push(`${pad}  arguments: (${args})`);
      if (outputs) out.push(`${pad}  outputs:   (${outputs})`);
      out.push(`${pad}  body {`);
      for (const l of stmt(e.body, indent + 2)) out.push(l);
      out.push(`${pad}  }`);
      return out;
    }
    case "Uniform": {
      const out = [`${pad}uniform {`];
      for (const u of v.uniforms) {
        const buf = u.buffer ? ` @buffer(${u.buffer})` : "";
        const slot = u.slot !== undefined ? ` @binding(${u.group ?? 0}.${u.slot})` : "";
        out.push(`${pad}  ${u.name}: ${type(u.type)}${buf}${slot}`);
      }
      out.push(`${pad}}`);
      return out;
    }
    case "Sampler":
      return [`${pad}sampler ${v.name}: ${type(v.type)} @binding(${v.binding.group}.${v.binding.slot})`];
    case "StorageBuffer":
      return [`${pad}storage<${v.access}> ${v.name}: ${type(v.layout)} @binding(${v.binding.group}.${v.binding.slot})`];
  }
}

function decoration(d: import("./types.js").EntryDecoration): string {
  switch (d.kind) {
    case "WorkgroupSize":
      return `workgroup_size(${d.x}${d.y !== undefined ? "," + d.y : ""}${d.z !== undefined ? "," + d.z : ""})`;
    case "OutputTopology":
      return `output_topology(${d.value})`;
  }
}

// ─────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────

function stmt(s: Stmt, indent: number): string[] {
  const pad = "  ".repeat(indent);
  switch (s.kind) {
    case "Nop":          return [];
    case "Return":       return [`${pad}return`];
    case "Break":        return [`${pad}break`];
    case "Continue":     return [`${pad}continue`];
    case "Discard":      return [`${pad}discard`];
    case "Barrier":      return [`${pad}barrier(${s.scope})`];
    case "Expression":   return [`${pad}${expr(s.value)}`];
    case "Declare": {
      const init = s.init ? ` = ${rExpr(s.init)}` : "";
      const mut = s.var.mutable ? "var" : "let";
      return [`${pad}${mut} ${s.var.name}: ${type(s.var.type)}${init}`];
    }
    case "Write":
      return [`${pad}${lExpr(s.target)} = ${expr(s.value)}`];
    case "WriteOutput":
      return [`${pad}out.${s.name}${s.index ? "[" + expr(s.index) + "]" : ""} = ${rExpr(s.value)}`];
    case "Increment":    return [`${pad}${lExpr(s.target)}${s.prefix ? "" : ""}++`];
    case "Decrement":    return [`${pad}${lExpr(s.target)}--`];
    case "ReturnValue":  return [`${pad}return ${expr(s.value)}`];
    case "Sequential":
    case "Isolated": {
      const out: string[] = [];
      for (const c of s.body) for (const l of stmt(c, indent)) out.push(l);
      return out;
    }
    case "If": {
      const out = [`${pad}if (${expr(s.cond)}) {`];
      for (const l of stmt(s.then, indent + 1)) out.push(l);
      if (s.else) {
        out.push(`${pad}} else {`);
        for (const l of stmt(s.else, indent + 1)) out.push(l);
      }
      out.push(`${pad}}`);
      return out;
    }
    case "For": {
      const out = [`${pad}for (${stmt(s.init, 0).join("; ")}; ${expr(s.cond)}; ${stmt(s.step, 0).join("; ")}) {`];
      for (const l of stmt(s.body, indent + 1)) out.push(l);
      out.push(`${pad}}`);
      return out;
    }
    case "While": {
      const out = [`${pad}while (${expr(s.cond)}) {`];
      for (const l of stmt(s.body, indent + 1)) out.push(l);
      out.push(`${pad}}`);
      return out;
    }
    case "DoWhile": {
      const out = [`${pad}do {`];
      for (const l of stmt(s.body, indent + 1)) out.push(l);
      out.push(`${pad}} while (${expr(s.cond)})`);
      return out;
    }
    case "Loop": {
      const out = [`${pad}loop {`];
      for (const l of stmt(s.body, indent + 1)) out.push(l);
      out.push(`${pad}}`);
      return out;
    }
    case "Switch": {
      const out = [`${pad}switch (${expr(s.value)}) {`];
      for (const c of s.cases) {
        out.push(`${pad}  case ${literal(c.literal)}:`);
        for (const l of stmt(c.body, indent + 2)) out.push(l);
      }
      if (s.default) {
        out.push(`${pad}  default:`);
        for (const l of stmt(s.default, indent + 2)) out.push(l);
      }
      out.push(`${pad}}`);
      return out;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────────────

function expr(e: Expr): string {
  switch (e.kind) {
    case "Const":         return literal(e.value);
    case "Var":           return e.var.name;
    case "ReadInput":     return `${e.scope}.${e.name}${e.index !== undefined ? "[" + expr(e.index) + "]" : ""}`;
    case "Call":          return `${e.fn.signature.name}(${e.args.map(expr).join(", ")})`;
    case "CallIntrinsic": return `${e.op.name}(${e.args.map(expr).join(", ")})`;
    case "Conditional":   return `(${expr(e.cond)} ? ${expr(e.ifTrue)} : ${expr(e.ifFalse)})`;
    case "Neg":           return `-${expr(e.value)}`;
    case "Not":           return `!${expr(e.value)}`;
    case "BitNot":        return `~${expr(e.value)}`;
    case "Add": case "Sub": case "Mul": case "Div": case "Mod":
    case "And": case "Or": case "BitAnd": case "BitOr": case "BitXor":
    case "ShiftLeft": case "ShiftRight":
    case "Eq": case "Neq": case "Lt": case "Le": case "Gt": case "Ge":
      return `(${expr(e.lhs)} ${binOp(e.kind)} ${expr(e.rhs)})`;
    case "MulMatMat": case "MulMatVec": case "MulVecMat":
      return `(${expr(e.lhs)} * ${expr(e.rhs)})`;
    case "Dot":           return `dot(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Cross":         return `cross(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Length":        return `length(${expr(e.value)})`;
    case "Transpose":     return `transpose(${expr(e.value)})`;
    case "Inverse":       return `inverse(${expr(e.value)})`;
    case "Determinant":   return `determinant(${expr(e.value)})`;
    case "VecSwizzle":    return `${expr(e.value)}.${e.comps.join("")}`;
    case "VecItem":       return `${expr(e.value)}[${expr(e.index)}]`;
    case "MatrixElement": return `${expr(e.matrix)}[${expr(e.row)}][${expr(e.col)}]`;
    case "MatrixRow":     return `row(${expr(e.matrix)}, ${expr(e.row)})`;
    case "MatrixCol":     return `${expr(e.matrix)}[${expr(e.col)}]`;
    case "NewVector":     return `${type(e.type)}(${e.components.map(expr).join(", ")})`;
    case "NewMatrix":     return `${type(e.type)}(${e.elements.map(expr).join(", ")})`;
    case "MatrixFromCols": return `${type(e.type)}.fromCols(${e.cols.map(expr).join(", ")})`;
    case "MatrixFromRows": return `${type(e.type)}.fromRows(${e.rows.map(expr).join(", ")})`;
    case "ConvertMatrix": return `${type(e.type)}(${expr(e.value)})`;
    case "Convert":       return `${type(e.type)}(${expr(e.value)})`;
    case "VecAny":        return `any(${expr(e.value)})`;
    case "VecAll":        return `all(${expr(e.value)})`;
    case "VecLt": case "VecLe": case "VecGt": case "VecGe": case "VecEq": case "VecNeq":
      return `${e.kind.toLowerCase()}(${expr(e.lhs)}, ${expr(e.rhs)})`;
    case "Field":         return `${expr(e.target)}.${e.name}`;
    case "Item":          return `${expr(e.target)}[${expr(e.index)}]`;
    case "DebugPrintf":   return `printf(${expr(e.format)}, ${e.args.map(expr).join(", ")})`;
  }
}

function binOp(k: string): string {
  switch (k) {
    case "Add": return "+";
    case "Sub": return "-";
    case "Mul": return "*";
    case "Div": return "/";
    case "Mod": return "%";
    case "And": return "&&";
    case "Or":  return "||";
    case "BitAnd": return "&";
    case "BitOr":  return "|";
    case "BitXor": return "^";
    case "ShiftLeft":  return "<<";
    case "ShiftRight": return ">>";
    case "Eq":  return "==";
    case "Neq": return "!=";
    case "Lt":  return "<";
    case "Le":  return "<=";
    case "Gt":  return ">";
    case "Ge":  return ">=";
    default: return k;
  }
}

function lExpr(l: LExpr): string {
  switch (l.kind) {
    case "LVar":          return l.var.name;
    case "LField":        return `${lExpr(l.target)}.${l.name}`;
    case "LItem":         return `${lExpr(l.target)}[${expr(l.index)}]`;
    case "LSwizzle":      return `${lExpr(l.target)}.${l.comps.join("")}`;
    case "LMatrixElement": return `${lExpr(l.matrix)}[${expr(l.row)}][${expr(l.col)}]`;
    case "LInput":         return `${l.scope}.${l.name}${l.index !== undefined ? "[" + expr(l.index) + "]" : ""}`;
  }
}

function rExpr(r: RExpr): string {
  if (r.kind === "Expr") return expr(r.value);
  return `[${r.values.map(expr).join(", ")}]`;
}

function literal(l: Literal): string {
  switch (l.kind) {
    case "Bool":  return l.value ? "true" : "false";
    case "Int":   return l.signed ? `${l.value}i` : `${l.value}u`;
    case "Float": return Number.isInteger(l.value) ? `${l.value}.0` : String(l.value);
    case "Null":  return "null";
  }
}

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

function type(t: Type): string {
  switch (t.kind) {
    case "Void":        return "void";
    case "Bool":        return "bool";
    case "Int":         return t.signed ? "i32" : "u32";
    case "Float":       return "f32";
    case "Vector":      return `vec${t.dim}<${type(t.element)}>`;
    case "Matrix":      return t.rows === t.cols ? `mat${t.rows}<${type(t.element)}>` : `mat${t.rows}x${t.cols}<${type(t.element)}>`;
    case "Array":       return `array<${type(t.element)}, ${t.length}>`;
    case "Struct":      return `struct ${t.name}`;
    case "Sampler":     return `sampler${t.target}${t.comparison ? "_compare" : ""}`;
    case "Texture":     return `texture${t.target}`;
    case "StorageTexture":
      return `texture_storage${t.target}<${t.format}, ${t.access}>`;
    case "AtomicI32":   return "atomic<i32>";
    case "AtomicU32":   return "atomic<u32>";
    case "Intrinsic":   return `intrinsic<${t.name}>`;
  }
}
