# wombat.shader IR — concrete plan

This document specifies the C-style imperative IR that sits between the
TypeScript frontend and the GLSL/WGSL emitters. It mirrors FShade's
[`Ast.fs`](../../fshade/src/Libs/FShade.Imperative/Ast.fs) (`CType`,
`CExpr`, `CLExpr`, `CRExpr`, `CStatement`, `CValueDef`, `CModule`)
with the deliberate departures listed at the end.

The IR is the **single contract** of the project. Every other piece
(frontend, passes, emitters, runtime) only sees IR — they don't share
types with each other.

## Design rules

1. **Discriminated unions on `kind`.** Every node is
   `{ kind: string, ... }` so it serialises straight to JSON, and so
   that a build-time-emitted IR can be persisted and restored by the
   runtime composer without re-parsing TS.
2. **Types on every expression node.** Every `Expr` carries its result
   `Type` inline. Costs a field, saves an inferring pass.
3. **Side-effect tag on every callable.** Intrinsics, user functions,
   atomics, image stores. DCE keys off this — pure nodes can vanish
   when their value is unused; impure nodes can't.
4. **Address spaces are first-class.** `Type` distinguishes `uniform`,
   `storage`, `workgroup`, `private`, `function`. WGSL needs them; GLSL
   ignores them at emit time.
5. **Source-map link on every node.** `span?: { file, start, end }` —
   the frontend fills it in, emitters propagate to GLSL/WGSL line
   maps, error messages can point at the original `.tsx` source.
6. **Aardvark naming for the shipped public types.** `V2f`/`M44f`/etc.
   live in `@aardworx/wombat.shader-types/index.d.ts`. The IR's `Type` representation
   stays structural: `{ kind: "Vector", element, dim }`. The frontend
   maps `V3f` → `{ kind: "Vector", element: F32, dim: 3 }`.

## Top-level shape

```ts
interface Module {
  readonly types: TypeDef[];     // user-defined struct decls
  readonly values: ValueDef[];   // constants, fns, entry points, uniforms
  readonly meta?: ModuleMeta;    // backend hints, debug info
}

type ValueDef =
  | ConstantDef        // global const
  | FunctionDef        // user shader-callable function
  | EntryDef           // a stage entry point: vertex / fragment / compute
  | UniformDef         // a group of uniforms (single uniform block)
  | StorageBufferDef   // ssbo (compute / WebGPU storage buffer)
  | SamplerDef;        // sampler binding (combined for GLSL, separated for WGSL)
```

The compose pipeline operates on `Module`s; the optimiser passes are
all `Module → Module` (or `EntryDef → EntryDef`).

## Type system — `Type`

```ts
type Type =
  | { kind: "Void" }
  | { kind: "Bool" }
  | { kind: "Int", signed: boolean, width: 32 }
  | { kind: "Float", width: 32 }
  | { kind: "Vector", element: Type, dim: 2 | 3 | 4 }
  | { kind: "Matrix", element: Type, rows: 2 | 3 | 4, cols: 2 | 3 | 4 }
  | { kind: "Array", element: Type, length: number | "runtime" }
  | { kind: "Struct", name: string, fields: readonly StructField[] }
  | { kind: "Sampler", target: SamplerTarget, sampled: SampledType, comparison: boolean }
  | { kind: "Texture", target: SamplerTarget, sampled: SampledType, arrayed: boolean, multisampled: boolean }
  | { kind: "AtomicI32" } | { kind: "AtomicU32" }
  | { kind: "Intrinsic", name: string, tag?: unknown };

interface StructField { readonly type: Type; readonly name: string }

type SamplerTarget = "1D" | "2D" | "3D" | "Cube" | "2DArray" | "CubeArray";
type SampledType = { kind: "Float" } | { kind: "Int", signed: boolean };
```

**Notes vs FShade `CType`:**

- No `Color` (use `Vector(Float, n)` + a semantic decoration).
- No `Pointer` / `PointerModifier`. Web shaders don't expose pointers
  in user code; functions are pass-by-value or marked `@inline`.
- `Float` width fixed at 32. WebGPU's f16 is opt-in via a precision
  qualifier (`@precision("low")`) on parameters/locals, not a
  separate type.
- `Array.length` is `number` for fixed-size or `"runtime"` for the
  WGSL runtime-sized arrays used at the end of storage structs.
- `Atomic*` are explicit IR types (WGSL semantics — atomic ops are
  only valid on these).
- Samplers and textures are split (WGSL has separate samplers; GLSL's
  `sampler2D` is a combined view).

## Variable — `Var`

```ts
interface Var {
  readonly name: string;
  readonly type: Type;
  readonly mutable: boolean;       // declared with `let` (GLSL) / `var` (WGSL)
}
```

Variable identity is by reference — a `Var` is interned per scope by
the frontend so that two `Var` references with the same `name` *are*
the same variable. Passes that rename use a fresh interned object.

## Literals — `Literal`

```ts
type Literal =
  | { kind: "Bool", value: boolean }
  | { kind: "Int", signed: boolean, value: number }    // i32 / u32
  | { kind: "Float", value: number }                   // f32
  | { kind: "Null" };                                  // for sampler defaults, etc.
```

## Expressions — `Expr`

Every `Expr` carries `type` and optional `span`. The `kind` discriminator
is at most one word.

```ts
type Expr = ExprBody & { type: Type; span?: Span };

type ExprBody =
  /* terminals */
  | { kind: "Var", var: Var }
  | { kind: "Const", value: Literal }
  | { kind: "ReadInput", scope: InputScope, name: string, index?: Expr }

  /* call shapes */
  | { kind: "Call", fn: FunctionRef, args: readonly Expr[] }
  | { kind: "CallIntrinsic", op: IntrinsicRef, args: readonly Expr[] }
  | { kind: "Conditional", cond: Expr, ifTrue: Expr, ifFalse: Expr }   // ternary

  /* unary numeric / boolean */
  | { kind: "Neg", value: Expr }
  | { kind: "Not", value: Expr }
  | { kind: "BitNot", value: Expr }

  /* binary numeric */
  | { kind: "Add" | "Sub" | "Mul" | "Div" | "Mod", lhs: Expr, rhs: Expr }

  /* matrix-aware multiplications. Frontend lowers `*` to whichever applies. */
  | { kind: "MulMatMat", lhs: Expr, rhs: Expr }
  | { kind: "MulMatVec", lhs: Expr, rhs: Expr }
  | { kind: "MulVecMat", lhs: Expr, rhs: Expr }

  /* matrix unary */
  | { kind: "Transpose", value: Expr }
  | { kind: "Inverse",   value: Expr }
  | { kind: "Determinant", value: Expr }

  /* vector specialised */
  | { kind: "Dot", lhs: Expr, rhs: Expr }
  | { kind: "Cross", lhs: Expr, rhs: Expr }
  | { kind: "Length", value: Expr }
  | { kind: "VecSwizzle", value: Expr, comps: readonly VecComp[] }       // a.xyz, a.brga
  | { kind: "VecItem", value: Expr, index: Expr }                        // a[i]
  | { kind: "MatrixElement", matrix: Expr, row: Expr, col: Expr }
  | { kind: "MatrixRow", matrix: Expr, row: Expr }
  | { kind: "MatrixCol", matrix: Expr, col: Expr }
  | { kind: "NewVector", components: readonly Expr[] }                   // V3f(x, y, z)  also V3f(v2, z)
  | { kind: "NewMatrix", elements: readonly Expr[] }                     // column-major flat
  | { kind: "MatrixFromRows", rows: readonly Expr[] }
  | { kind: "MatrixFromCols", cols: readonly Expr[] }
  | { kind: "ConvertMatrix", value: Expr }                               // M44f → M33f truncate, etc.

  /* conversions */
  | { kind: "Convert", value: Expr }                                     // float(i), int(f), etc.

  /* logic / comparisons */
  | { kind: "And" | "Or", lhs: Expr, rhs: Expr }                         // short-circuit
  | { kind: "BitAnd" | "BitOr" | "BitXor", lhs: Expr, rhs: Expr }
  | { kind: "ShiftLeft" | "ShiftRight", lhs: Expr, rhs: Expr }
  | { kind: "Eq" | "Neq" | "Lt" | "Le" | "Gt" | "Ge", lhs: Expr, rhs: Expr }

  /* component-wise vector comparisons. Result is a Bool/BoolN. */
  | { kind: "VecAny" | "VecAll", value: Expr }                           // collapses BVec3 → Bool
  | { kind: "VecLt"  | "VecLe"  | "VecGt"  | "VecGe"
        | "VecEq"  | "VecNeq", lhs: Expr, rhs: Expr }                    // result: BVec*

  /* memory access */
  | { kind: "Field", target: Expr, name: string }
  | { kind: "Item", target: Expr, index: Expr }

  /* debug */
  | { kind: "DebugPrintf", format: Expr, args: readonly Expr[] };

type VecComp = "x" | "y" | "z" | "w";
type Span = { file: string; start: number; end: number };

type InputScope = "Input" | "Output" | "Uniform" | "Builtin";
```

**Departures from FShade `CExpr`:**

- All "any/all" lifted to a single `VecAny` / `VecAll` reduction (FShade
  has nine variants — `CVecAnyEqual`, `CVecAllNotEqual`, etc.). The
  composition `VecAny(VecLt(a,b))` is the same shape and emits to the
  same GLSL `any(lessThan(a,b))`.
- `Inverse` and `Determinant` are explicit nodes (FShade puts them in
  intrinsics; we surface them so passes can see them).
- No `AddressOf`. Web shaders don't have user-visible pointers.
- `Conditional` only for ternary (no `if-expr` form — branch
  statements live in `Stmt`).
- `Convert` covers all numeric conversions; matrix dim-changes get
  their own `ConvertMatrix`.

## Function references — `FunctionRef` / `IntrinsicRef`

```ts
interface FunctionRef {
  readonly id: string;            // stable name; user fn or generated
  readonly signature: FunctionSignature;
  readonly pure: boolean;
}

interface FunctionSignature {
  readonly name: string;
  readonly returnType: Type;
  readonly parameters: readonly Parameter[];
}

interface Parameter {
  readonly name: string;
  readonly type: Type;
  readonly modifier: "in" | "inout";    // `out` lowered to inout for emit; pure-by-value functions use "in"
  readonly semantic?: string;
  readonly decorations?: readonly ParamDecoration[];
}

interface IntrinsicRef {
  readonly name: string;          // e.g. "sin", "mix", "texture"
  readonly returnTypeOf: (args: readonly Type[]) => Type;
  readonly pure: boolean;
  readonly samplerBinding?: boolean;   // calls that need a sampler+texture binding
  readonly emit: { glsl: string; wgsl: string };
}
```

`pure: false` on either keeps DCE from removing the call. `samplerBinding`
flags `texture(...)` and friends so the emitter can wire up sampler
slots correctly.

## L-expressions — `LExpr`

The l-value side: targets of writes, increments, and `out` parameter
binding sites. FShade has the same split (`CLExpr`).

```ts
type LExpr = LExprBody & { type: Type; span?: Span };

type LExprBody =
  | { kind: "LVar", var: Var }
  | { kind: "LField", target: LExpr, name: string }
  | { kind: "LItem", target: LExpr, index: Expr }
  | { kind: "LSwizzle", target: LExpr, comps: readonly VecComp[] }       // v.xy = ...
  | { kind: "LMatrixElement", matrix: LExpr, row: Expr, col: Expr }
  | { kind: "LInput", scope: InputScope, name: string, index?: Expr };   // `out_color = ...`
```

Note: `LExpr` builds out of `LExpr` recursively, with `Expr` for
indices only — `arr[expr].field.xy` is `LSwizzle(LField(LItem(LVar(arr), expr), "field"), [x, y])`.

## R-expressions — `RExpr`

For variable initialisers and output writes that may carry an array
literal directly (FShade's `CRExpr`).

```ts
type RExpr =
  | { kind: "Expr", value: Expr }
  | { kind: "ArrayLiteral", arrayType: Type, values: readonly Expr[] };
```

## Statements — `Stmt`

```ts
type Stmt =
  | { kind: "Nop" }
  | { kind: "Expression", value: Expr }                          // `f()`; — for side-effecting calls
  | { kind: "Declare", var: Var, init?: RExpr }
  | { kind: "Write", target: LExpr, value: Expr }
  | { kind: "WriteOutput", name: string, index?: Expr, value: RExpr }   // gl_Position = …
  | { kind: "Increment", target: LExpr, prefix: boolean }
  | { kind: "Decrement", target: LExpr, prefix: boolean }
  | { kind: "Sequential", body: readonly Stmt[] }                // brace block
  | { kind: "Isolated", body: readonly Stmt[] }                  // scope barrier (no var leak across)
  | { kind: "Return" }
  | { kind: "ReturnValue", value: Expr }
  | { kind: "Break" }
  | { kind: "Continue" }
  | { kind: "If", cond: Expr, then: Stmt, else?: Stmt }
  | { kind: "For", init: Stmt, cond: Expr, step: Stmt, body: Stmt }
  | { kind: "While", cond: Expr, body: Stmt }
  | { kind: "DoWhile", cond: Expr, body: Stmt }
  | { kind: "Switch", value: Expr, cases: readonly SwitchCase[], default?: Stmt }
  | { kind: "Discard" }                                          // fragment-only
  | { kind: "Barrier", scope: BarrierScope };                    // compute

interface SwitchCase { readonly literal: Literal; readonly body: Stmt }
type BarrierScope = "workgroup" | "storage" | "subgroup";
```

**Departures from FShade `CStatement`:**

- Explicit `Discard` (frag) and `Barrier` (compute) nodes.
- No `CIsolated` semantics tied to F#-quotation peculiarities — kept
  the name but it's just a scope-barrier hint for hoisting passes.
- `If.else` optional (FShade always has both branches; we'll lower).

## Definitions — `ValueDef` / `TypeDef`

```ts
type ValueDef =
  | { kind: "Constant", varType: Type, name: string, init: RExpr }
  | { kind: "Function", signature: FunctionSignature, body: Stmt, attributes?: readonly FnAttr[] }
  | { kind: "Entry", entry: EntryDef }
  | { kind: "Uniform", uniforms: readonly UniformDecl[] }
  | { kind: "StorageBuffer", binding: BindingPoint, name: string, layout: Type, access: "read" | "read_write" }
  | { kind: "Sampler", binding: BindingPoint, name: string, type: Type };

type FnAttr = "inline" | "no_inline" | "must_use";

type TypeDef =
  | { kind: "Struct", name: string, fields: readonly StructField[] };

interface EntryDef {
  readonly name: string;
  readonly stage: Stage;
  readonly inputs: readonly EntryParameter[];
  readonly outputs: readonly EntryParameter[];
  readonly arguments: readonly EntryParameter[];      // only used for compute (workgroup args)
  readonly returnType: Type;                          // usually Void; vertex/fragment write outputs
  readonly body: Stmt;
  readonly decorations: readonly EntryDecoration[];
}

type Stage = "vertex" | "fragment" | "compute";

interface EntryParameter {
  readonly name: string;
  readonly type: Type;
  readonly semantic: string;                          // "Position" / "Color" / "WorldPos" / etc.
  readonly decorations: readonly ParamDecoration[];
}

type EntryDecoration =
  | { kind: "WorkgroupSize", x: number, y?: number, z?: number }
  | { kind: "OutputTopology", value: "triangle-list" | "line-list" | "point-list" };

type ParamDecoration =
  | { kind: "Interpolation", mode: "smooth" | "flat" | "centroid" | "sample" | "no-perspective" }
  | { kind: "Builtin", value: BuiltinSemantic }
  | { kind: "Location", value: number }
  | { kind: "Binding", group: number, slot: number };

type BuiltinSemantic =
  | "position" | "vertex_index" | "instance_index"
  | "front_facing" | "frag_depth"
  | "global_invocation_id" | "local_invocation_id" | "local_invocation_index"
  | "workgroup_id" | "num_workgroups";

interface UniformDecl {
  readonly name: string;
  readonly type: Type;
  readonly group?: number;                            // WGSL bind group
  readonly slot?: number;
  readonly buffer?: string;                           // GLSL UBO name; same name = same block
}

interface BindingPoint { group: number; slot: number }
```

**Departures from FShade:**

- `EntryDef.stage` is explicit; FShade discovers it from the
  builder name. Surface it on the IR so passes can dispatch.
- `Sampler` and `StorageBuffer` are top-level `ValueDef`s; they were
  woven through the `CUniform` / `CRaytracingDataDef` machinery in
  FShade in a way specific to .NET reflection. Cleaner for us to
  separate.
- No `CConditionalDef` (`#ifdef`-style branches in the module).
  Web targets don't need it; if we want feature gates, we generate
  separate `Module`s.

## Use-def helpers (lives in `@aardworx/wombat.shader-ir/visit.ts`)

The minimum surface the optimiser needs:

```ts
// Walks every Expr, LExpr, Stmt within a Stmt, calling visitors.
function visitStmt(s: Stmt, v: StmtVisitor): void;
function mapStmt(s: Stmt, v: StmtMapper): Stmt;

// Free variables / read inputs / written outputs of a node.
function freeVars(s: Stmt | Expr): ReadonlySet<Var>;
function readInputs(s: Stmt | Expr): ReadonlySet<{scope: InputScope; name: string}>;
function writtenOutputs(s: Stmt): ReadonlySet<string>;

// Substitution.
function substVar(s: Stmt, oldVar: Var, newExpr: Expr): Stmt;
function substInput(s: Stmt, scope: InputScope, name: string, expr: Expr): Stmt;

// Side-effect inquiry.
function isPure(e: Expr): boolean;          // true iff no impure call/intrinsic anywhere in subtree
function isSideEffectStmt(s: Stmt): boolean; // true iff Stmt has any side-effecting child
```

Everything in `@aardworx/wombat.shader-passes` is built on these.

## Pass list (lives in `@aardworx/wombat.shader-passes/`)

Order matters. Passes return new IR; nothing mutates in place.

1. `inline(module, policy)` — inlines `@inline` functions and copy-
   propagates trivial bindings.
2. `foldConstants(module)` — evaluates constant subtrees on
   pure-only paths.
3. `cse(module)` — common-subexpression elimination within an
   `EntryDef`.
4. `dce(module)` — drops dead bindings and pure unread results.
5. `composeStages(module)` — given a list of `EntryDef`s, fuses
   same-stage entries by semantic and pipelines vertex→fragment.
6. **`pruneCrossStage(module)` — the load-bearing pass.** Walks the
   fragment entry, collects every `ReadInput` it makes; strips
   unread fields from the vertex output struct; reruns DCE on the
   vertex; iterates to fixed point. Produces an IR-level Aardvark
   `EffectInputLayout`-equivalent as a side product, recording the
   final shape per stage.
7. `reduceUniforms(module)` — drops `Uniform` / `Sampler` /
   `StorageBuffer` decls that no surviving stage references.
8. `legaliseTypes(module, target)` — target-specific lowering. WGSL
   requires explicit address spaces; GLSL doesn't. Splits combined
   samplers for WGSL or combines them for GLSL.

Each pass is `(Module) → Module` (or `(Module, args) → Module`) so
they can be composed in any order during testing, and so the
optimiser is just `pipe(passes, module)`.

## Emitter contract

```ts
interface Emitter {
  emit(module: Module): EmitResult;
}

interface EmitResult {
  readonly source: string;
  readonly sourceMap: SourceMap;
  readonly bindings: BindingMap;       // semantic → location/group/slot
  readonly meta: BackendMeta;          // workgroup size, etc.
}
```

`@aardworx/wombat.shader-glsl` emits GLSL ES 3.00 (WebGL2 target). `@aardworx/wombat.shader-wgsl` emits
WGSL. Both consume a fully-legalised module — `legaliseTypes` runs
last in the pass list so the emitter doesn't need to know about
target subtleties.

## Test harness

Every IR node and pass ships with property tests. The shape:

```ts
// 1. hand-write a small Module.
const m = exampleModule(/* … */);
// 2. run a pass.
const m2 = dce(m);
// 3. assert structural invariants.
expect(freeVars(m2.values)).toEqual(/* … */);
// 4. emit and snapshot the GLSL string.
expect(emitGlsl(m2).source).toMatchSnapshot();
```

The cross-stage prune pass gets its own dedicated test fixture: a
v+f IR program with deliberately-unused fragment inputs and assertions
that the corresponding vertex outputs and their compute chains drop
out cleanly.

## Naming reminder — public types in `@aardworx/wombat.shader-types`

| IR shape | Public TS name |
| --- | --- |
| `Vector(Int(true,32), 2/3/4)` | `V2i` / `V3i` / `V4i` |
| `Vector(Int(false,32), 2/3/4)` | `V2u` / `V3u` / `V4u` |
| `Vector(Float(32), 2/3/4)` | `V2f` / `V3f` / `V4f` |
| `Vector(Bool, 2/3/4)` | `V2b` / `V3b` / `V4b` |
| `Matrix(Float(32), R, C)` | `M{R}{C}f` (e.g. `M44f`, `M23f`) |
| `Matrix(Int(true,32), R, C)` | `M{R}{C}i` (rare; ship if WebGPU adopts) |
| `Sampler(2D, Float)` | `Sampler2D` |
| `Sampler(2D, Int)` | `ISampler2D` |
| `Sampler(Cube, Float)` | `SamplerCube` |

The frontend recognises these by name (or, more robustly, by the
brand symbol on the declared class) and emits the structural IR.
