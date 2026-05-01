// Storage-buffer access-mode and atomic-type inference.
//
// For each `StorageBuffer` ValueDef, scan every entry's body to decide:
//
//   - `access`: `read` if the body only reads the buffer, `read_write`
//     if anything writes to it (`Write` target, `Increment`/`Decrement`,
//     compound assignment lowered to `Write`, or any atomic op).
//
//   - `layout.element`: promoted to `AtomicI32`/`AtomicU32` if any
//     atomic intrinsic touches the buffer. Walks one level of `Array`
//     so the typical `u32[]` storage layout gets its element wrapped;
//     scalar storage (`u32`) is wrapped directly.
//
// Runs before `legaliseTypes` so the WGSL emitter sees the right
// element kinds when laying out the binding.

import type {
  Expr,
  LExpr,
  Module,
  Stmt,
  Type,
  ValueDef,
} from "../ir/index.js";
import { mapStmt } from "./transform.js";

interface BufferUsage {
  written: boolean;
  atomic: boolean;
}

export function inferStorageAccess(module: Module): Module {
  const buffers = new Map<string, BufferUsage>();
  for (const v of module.values) {
    if (v.kind === "StorageBuffer") {
      buffers.set(v.name, { written: false, atomic: false });
    }
  }
  if (buffers.size === 0) return module;

  for (const v of module.values) {
    if (v.kind === "Entry") walkStmt(v.entry.body, buffers);
    else if (v.kind === "Function") walkStmt(v.body, buffers);
  }

  let changed = false;
  const values = module.values.map((v): ValueDef => {
    if (v.kind !== "StorageBuffer") return v;
    const usage = buffers.get(v.name);
    if (!usage) return v;
    const access = usage.written || usage.atomic ? "read_write" : "read";
    const layout = usage.atomic ? promoteAtomic(v.layout) : v.layout;
    if (access === v.access && layout === v.layout) return v;
    changed = true;
    return { ...v, access, layout };
  });
  return changed ? { ...module, values } : module;
}

// ─────────────────────────────────────────────────────────────────────

function promoteAtomic(t: Type): Type {
  if (t.kind === "Array") {
    return { ...t, element: promoteAtomic(t.element) };
  }
  if (t.kind === "Int") {
    return t.signed ? { kind: "AtomicI32" } : { kind: "AtomicU32" };
  }
  return t;
}

function rootBufferNameLExpr(l: LExpr): string | undefined {
  switch (l.kind) {
    case "LVar": return l.var.name;
    case "LField":
    case "LSwizzle":
    case "LItem":
      return rootBufferNameLExpr(l.target);
    case "LMatrixElement": return rootBufferNameLExpr(l.matrix);
    case "LInput": return undefined;
  }
}

function rootBufferNameExpr(e: Expr): string | undefined {
  switch (e.kind) {
    case "Var": return e.var.name;
    // Free identifiers (uniforms, samplers, storage buffers) translate
    // to ReadInput; storage-buffer reads keep the source-level name.
    case "ReadInput": return e.name;
    case "Item":
    case "Field":
      return rootBufferNameExpr(e.target);
    case "VecSwizzle":
    case "VecItem":
      return rootBufferNameExpr(e.value);
    case "MatrixCol":
    case "MatrixRow":
      return rootBufferNameExpr(e.matrix);
    default:
      return undefined;
  }
}

function walkStmt(s: Stmt, buffers: Map<string, BufferUsage>): void {
  // mapStmt's `stmt` callback only fires on child statements; the
  // top-level Stmt itself is never passed through. So we wrap our
  // observers in an explicit pre-order pass: handle this Stmt's
  // direct effects, then recurse via mapStmt for its children.
  observeStmt(s, buffers);
  mapStmt(s, {
    expr: (e) => {
      if (e.kind === "CallIntrinsic" && e.op.atomic && e.args.length > 0) {
        const name = rootBufferNameExpr(e.args[0]!);
        if (name) {
          const u = buffers.get(name);
          if (u) {
            u.atomic = true;
            u.written = true;
          }
        }
      }
      return e;
    },
    lexpr: (l) => l,
    stmt: (c) => {
      observeStmt(c, buffers);
      return c;
    },
  });
}

function observeStmt(c: Stmt, buffers: Map<string, BufferUsage>): void {
  if (c.kind === "Write" || c.kind === "Increment" || c.kind === "Decrement") {
    const name = rootBufferNameLExpr(c.target);
    if (name) {
      const u = buffers.get(name);
      if (u) u.written = true;
    }
  }
}
