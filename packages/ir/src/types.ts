// wombat.shader IR — type definitions.
// See ../../docs/IR.md for the canonical specification.
//
// The IR is the single contract between frontend, passes, and emitters.
// Every node is `{ kind: string, ... }` so it serialises directly to JSON.

// ─────────────────────────────────────────────────────────────────────
// Source spans
// ─────────────────────────────────────────────────────────────────────

export interface Span {
  readonly file: string;
  readonly start: number;
  readonly end: number;
}

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export type SamplerTarget =
  | "1D" | "2D" | "3D" | "Cube" | "2DArray" | "CubeArray";

export type SampledType =
  | { readonly kind: "Float" }
  | { readonly kind: "Int"; readonly signed: boolean };

export interface StructField {
  readonly type: Type;
  readonly name: string;
}

export type Type =
  | { readonly kind: "Void" }
  | { readonly kind: "Bool" }
  | { readonly kind: "Int"; readonly signed: boolean; readonly width: 32 }
  | { readonly kind: "Float"; readonly width: 32 }
  | { readonly kind: "Vector"; readonly element: Type; readonly dim: 2 | 3 | 4 }
  | { readonly kind: "Matrix"; readonly element: Type; readonly rows: 2 | 3 | 4; readonly cols: 2 | 3 | 4 }
  | { readonly kind: "Array"; readonly element: Type; readonly length: number | "runtime" }
  | { readonly kind: "Struct"; readonly name: string; readonly fields: readonly StructField[] }
  | {
      readonly kind: "Sampler";
      readonly target: SamplerTarget;
      readonly sampled: SampledType;
      readonly comparison: boolean;
    }
  | {
      readonly kind: "Texture";
      readonly target: SamplerTarget;
      readonly sampled: SampledType;
      readonly arrayed: boolean;
      readonly multisampled: boolean;
    }
  | { readonly kind: "AtomicI32" }
  | { readonly kind: "AtomicU32" }
  | { readonly kind: "Intrinsic"; readonly name: string; readonly tag?: unknown };

// Convenience constructors / common instances.
export const Tvoid: Type = { kind: "Void" };
export const Tbool: Type = { kind: "Bool" };
export const Ti32: Type = { kind: "Int", signed: true, width: 32 };
export const Tu32: Type = { kind: "Int", signed: false, width: 32 };
export const Tf32: Type = { kind: "Float", width: 32 };
export const Vec = (element: Type, dim: 2 | 3 | 4): Type => ({ kind: "Vector", element, dim });
export const Mat = (element: Type, rows: 2 | 3 | 4, cols: 2 | 3 | 4): Type => ({
  kind: "Matrix", element, rows, cols,
});

// ─────────────────────────────────────────────────────────────────────
// Variables and literals
// ─────────────────────────────────────────────────────────────────────

export interface Var {
  readonly name: string;
  readonly type: Type;
  readonly mutable: boolean;
}

export type Literal =
  | { readonly kind: "Bool"; readonly value: boolean }
  | { readonly kind: "Int"; readonly signed: boolean; readonly value: number }
  | { readonly kind: "Float"; readonly value: number }
  | { readonly kind: "Null" };

// ─────────────────────────────────────────────────────────────────────
// Function references
// ─────────────────────────────────────────────────────────────────────

export interface Parameter {
  readonly name: string;
  readonly type: Type;
  readonly modifier: "in" | "inout";
  readonly semantic?: string;
  readonly decorations?: readonly ParamDecoration[];
}

export interface FunctionSignature {
  readonly name: string;
  readonly returnType: Type;
  readonly parameters: readonly Parameter[];
}

export interface FunctionRef {
  readonly id: string;
  readonly signature: FunctionSignature;
  readonly pure: boolean;
}

export interface IntrinsicRef {
  readonly name: string;
  /** Computed from arg types; concrete result of each call site is on the Expr. */
  readonly returnTypeOf: (args: readonly Type[]) => Type;
  readonly pure: boolean;
  readonly samplerBinding?: boolean;
  readonly emit: { readonly glsl: string; readonly wgsl: string };
}

// ─────────────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────────────

export type VecComp = "x" | "y" | "z" | "w";
export type InputScope = "Input" | "Output" | "Uniform" | "Builtin";

type _ExprBase = { readonly type: Type; readonly span?: Span };

export type Expr = _ExprBase & ExprBody;

export type ExprBody =
  | { readonly kind: "Var"; readonly var: Var }
  | { readonly kind: "Const"; readonly value: Literal }
  | {
      readonly kind: "ReadInput";
      readonly scope: InputScope;
      readonly name: string;
      readonly index?: Expr;
    }
  | { readonly kind: "Call"; readonly fn: FunctionRef; readonly args: readonly Expr[] }
  | { readonly kind: "CallIntrinsic"; readonly op: IntrinsicRef; readonly args: readonly Expr[] }
  | {
      readonly kind: "Conditional";
      readonly cond: Expr;
      readonly ifTrue: Expr;
      readonly ifFalse: Expr;
    }
  | { readonly kind: "Neg"; readonly value: Expr }
  | { readonly kind: "Not"; readonly value: Expr }
  | { readonly kind: "BitNot"; readonly value: Expr }
  | { readonly kind: "Add" | "Sub" | "Mul" | "Div" | "Mod"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "MulMatMat"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "MulMatVec"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "MulVecMat"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "Transpose"; readonly value: Expr }
  | { readonly kind: "Inverse"; readonly value: Expr }
  | { readonly kind: "Determinant"; readonly value: Expr }
  | { readonly kind: "Dot"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "Cross"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "Length"; readonly value: Expr }
  | { readonly kind: "VecSwizzle"; readonly value: Expr; readonly comps: readonly VecComp[] }
  | { readonly kind: "VecItem"; readonly value: Expr; readonly index: Expr }
  | { readonly kind: "MatrixElement"; readonly matrix: Expr; readonly row: Expr; readonly col: Expr }
  | { readonly kind: "MatrixRow"; readonly matrix: Expr; readonly row: Expr }
  | { readonly kind: "MatrixCol"; readonly matrix: Expr; readonly col: Expr }
  | { readonly kind: "NewVector"; readonly components: readonly Expr[] }
  | { readonly kind: "NewMatrix"; readonly elements: readonly Expr[] }
  | { readonly kind: "MatrixFromRows"; readonly rows: readonly Expr[] }
  | { readonly kind: "MatrixFromCols"; readonly cols: readonly Expr[] }
  | { readonly kind: "ConvertMatrix"; readonly value: Expr }
  | { readonly kind: "Convert"; readonly value: Expr }
  | { readonly kind: "And" | "Or"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "BitAnd" | "BitOr" | "BitXor"; readonly lhs: Expr; readonly rhs: Expr }
  | { readonly kind: "ShiftLeft" | "ShiftRight"; readonly lhs: Expr; readonly rhs: Expr }
  | {
      readonly kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge";
      readonly lhs: Expr;
      readonly rhs: Expr;
    }
  | { readonly kind: "VecAny" | "VecAll"; readonly value: Expr }
  | {
      readonly kind: "VecLt" | "VecLe" | "VecGt" | "VecGe" | "VecEq" | "VecNeq";
      readonly lhs: Expr;
      readonly rhs: Expr;
    }
  | { readonly kind: "Field"; readonly target: Expr; readonly name: string }
  | { readonly kind: "Item"; readonly target: Expr; readonly index: Expr }
  | { readonly kind: "DebugPrintf"; readonly format: Expr; readonly args: readonly Expr[] };

// ─────────────────────────────────────────────────────────────────────
// L-expressions
// ─────────────────────────────────────────────────────────────────────

export type LExpr = _ExprBase & LExprBody;

export type LExprBody =
  | { readonly kind: "LVar"; readonly var: Var }
  | { readonly kind: "LField"; readonly target: LExpr; readonly name: string }
  | { readonly kind: "LItem"; readonly target: LExpr; readonly index: Expr }
  | { readonly kind: "LSwizzle"; readonly target: LExpr; readonly comps: readonly VecComp[] }
  | {
      readonly kind: "LMatrixElement";
      readonly matrix: LExpr;
      readonly row: Expr;
      readonly col: Expr;
    }
  | {
      readonly kind: "LInput";
      readonly scope: InputScope;
      readonly name: string;
      readonly index?: Expr;
    };

// ─────────────────────────────────────────────────────────────────────
// R-expressions (initialisers / output writes)
// ─────────────────────────────────────────────────────────────────────

export type RExpr =
  | { readonly kind: "Expr"; readonly value: Expr }
  | { readonly kind: "ArrayLiteral"; readonly arrayType: Type; readonly values: readonly Expr[] };

// ─────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────

export type BarrierScope = "workgroup" | "storage" | "subgroup";

export interface SwitchCase {
  readonly literal: Literal;
  readonly body: Stmt;
}

export type Stmt =
  | { readonly kind: "Nop" }
  | { readonly kind: "Expression"; readonly value: Expr }
  | { readonly kind: "Declare"; readonly var: Var; readonly init?: RExpr }
  | { readonly kind: "Write"; readonly target: LExpr; readonly value: Expr }
  | {
      readonly kind: "WriteOutput";
      readonly name: string;
      readonly index?: Expr;
      readonly value: RExpr;
    }
  | { readonly kind: "Increment"; readonly target: LExpr; readonly prefix: boolean }
  | { readonly kind: "Decrement"; readonly target: LExpr; readonly prefix: boolean }
  | { readonly kind: "Sequential"; readonly body: readonly Stmt[] }
  | { readonly kind: "Isolated"; readonly body: readonly Stmt[] }
  | { readonly kind: "Return" }
  | { readonly kind: "ReturnValue"; readonly value: Expr }
  | { readonly kind: "Break" }
  | { readonly kind: "Continue" }
  | { readonly kind: "If"; readonly cond: Expr; readonly then: Stmt; readonly else?: Stmt }
  | {
      readonly kind: "For";
      readonly init: Stmt;
      readonly cond: Expr;
      readonly step: Stmt;
      readonly body: Stmt;
    }
  | { readonly kind: "While"; readonly cond: Expr; readonly body: Stmt }
  | { readonly kind: "DoWhile"; readonly cond: Expr; readonly body: Stmt }
  | {
      readonly kind: "Switch";
      readonly value: Expr;
      readonly cases: readonly SwitchCase[];
      readonly default?: Stmt;
    }
  | { readonly kind: "Discard" }
  | { readonly kind: "Barrier"; readonly scope: BarrierScope };

// ─────────────────────────────────────────────────────────────────────
// Definitions and module
// ─────────────────────────────────────────────────────────────────────

export type FnAttr = "inline" | "no_inline" | "must_use";

export interface BindingPoint {
  readonly group: number;
  readonly slot: number;
}

export type BuiltinSemantic =
  | "position" | "vertex_index" | "instance_index"
  | "front_facing" | "frag_depth"
  | "global_invocation_id" | "local_invocation_id" | "local_invocation_index"
  | "workgroup_id" | "num_workgroups";

export type ParamDecoration =
  | {
      readonly kind: "Interpolation";
      readonly mode: "smooth" | "flat" | "centroid" | "sample" | "no-perspective";
    }
  | { readonly kind: "Builtin"; readonly value: BuiltinSemantic }
  | { readonly kind: "Location"; readonly value: number }
  | { readonly kind: "Binding"; readonly group: number; readonly slot: number };

export type EntryDecoration =
  | { readonly kind: "WorkgroupSize"; readonly x: number; readonly y?: number; readonly z?: number }
  | {
      readonly kind: "OutputTopology";
      readonly value: "triangle-list" | "line-list" | "point-list";
    };

export type Stage = "vertex" | "fragment" | "compute";

export interface EntryParameter {
  readonly name: string;
  readonly type: Type;
  readonly semantic: string;
  readonly decorations: readonly ParamDecoration[];
}

export interface EntryDef {
  readonly name: string;
  readonly stage: Stage;
  readonly inputs: readonly EntryParameter[];
  readonly outputs: readonly EntryParameter[];
  readonly arguments: readonly EntryParameter[];
  readonly returnType: Type;
  readonly body: Stmt;
  readonly decorations: readonly EntryDecoration[];
}

export interface UniformDecl {
  readonly name: string;
  readonly type: Type;
  readonly group?: number;
  readonly slot?: number;
  /** GLSL UBO name; same name = same block. */
  readonly buffer?: string;
}

export type ValueDef =
  | { readonly kind: "Constant"; readonly varType: Type; readonly name: string; readonly init: RExpr }
  | {
      readonly kind: "Function";
      readonly signature: FunctionSignature;
      readonly body: Stmt;
      readonly attributes?: readonly FnAttr[];
    }
  | { readonly kind: "Entry"; readonly entry: EntryDef }
  | { readonly kind: "Uniform"; readonly uniforms: readonly UniformDecl[] }
  | {
      readonly kind: "StorageBuffer";
      readonly binding: BindingPoint;
      readonly name: string;
      readonly layout: Type;
      readonly access: "read" | "read_write";
    }
  | {
      readonly kind: "Sampler";
      readonly binding: BindingPoint;
      readonly name: string;
      readonly type: Type;
    };

export type TypeDef =
  | { readonly kind: "Struct"; readonly name: string; readonly fields: readonly StructField[] };

export interface ModuleMeta {
  readonly target?: "glsl-es-300" | "wgsl";
  readonly debug?: boolean;
}

export interface Module {
  readonly types: readonly TypeDef[];
  readonly values: readonly ValueDef[];
  readonly meta?: ModuleMeta;
}
