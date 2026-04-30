# Shader intrinsics

This is the canonical surface of free-function intrinsics shipped in
`@aardworx/wombat.shader-types`. Anything with a wombat.base method
equivalent has been **removed from the free-function surface** for
vectors — users call the method form (`v.length()`, `a.dot(b)`, …);
the shader frontend translates it to identical IR. Scalars retain
the free-function form where applicable (`abs(0.5)`, `min(a, b)`
where both are `number`).

The pre-Phase-9 surface that exposed both forms is gone. There's
exactly **one way** to spell each operation now.

---

## What's shipped (free-function)

### Scalar / vector trig + transcendental

```
sin / cos / tan
asin / acos / atan / atan2
sinh / cosh / tanh / asinh / acosh / atanh
exp / exp2 / log / log2
pow / sqrt / inversesqrt
degrees / radians / trunc
```

Vector overloads kept — wombat.base doesn't expose `V*f.sin()` etc.
(per-component trig isn't typically a vector method).

### Scalar element-wise math

```
abs / sign / floor / ceil / round / fract / mod
min / max / clamp / mix
```

**Scalar overloads only.** For vectors, use methods:
`v.abs()`, `a.min(b)`, `v.clamp(lo, hi)`, `a.lerp(b, t)`.

### Scalar / vector helpers

```
step(edge, x)
smoothstep(edge0, edge1, x)
```

Both scalar and vector overloads kept (no wombat.base method form).

### Geometric utilities (vector-only, no method form)

```
reflect(I, N) → V*f
refract(I, N, eta) → V*f
faceforward(N, I, Nref) → V*f
```

Could move to V*f methods later; FShade keeps them as free functions.

### Vector predicates

```
any(boolVec) → bool
all(boolVec) → bool
```

Operate on `V*b`. wombat.base's V*b doesn't expose `any`/`all`
methods today.

### Fragment-only

```
dFdx / dFdy / fwidth      // both scalar and vector overloads kept
discard()
```

### Compute-only — atomics (WGSL only)

```
atomicLoad / atomicStore
atomicAdd / atomicSub
atomicMin / atomicMax
atomicAnd / atomicOr / atomicXor
atomicExchange / atomicCompareExchangeWeak
```

The frontend infers atomic typing on the buffer from these calls;
the WGSL emitter prepends `&` to the first argument. GLSL ES 3.00
has no surface for them — the emitter throws.

### Compute-only — barriers

```
workgroupBarrier()
storageBarrier()
```

### Texture sampling

```
texture(sampler, coord)
textureLod(sampler, coord, lod)
textureGrad(sampler, coord, dPdx, dPdy)
texelFetch(sampler, ij, lod_or_sample)
textureLoad(sampler, coord)            // alias of texelFetch on WGSL
textureStore(storageTex, coord, value) // storage-texture write, WGSL only
textureGather(sampler, coord)
textureSampleCompare(shadowSampler, coord, depthRef)
textureSize(sampler, lod) → V*i
```

Sampler/texture handles are GPU-only; wombat.base doesn't ship
runtime classes for them.

### Bit operations (integer math, scalar only)

```
countOneBits         // GLSL: bitCount
firstLeadingBit      // GLSL: findMSB
firstTrailingBit     // GLSL: findLSB
extractBits          // GLSL: bitfieldExtract
insertBits           // GLSL: bitfieldInsert
reverseBits          // GLSL: bitfieldReverse
```

### Pack / unpack

```
pack2x16float / unpack2x16float
pack2x16unorm / unpack2x16unorm
pack2x16snorm / unpack2x16snorm
pack4x8unorm / unpack4x8unorm
pack4x8snorm / unpack4x8snorm
```

WGSL-style names. The GLSL emitter maps to ES 3.00 spellings
(`packHalf2x16`, `packUnorm4x8`, …).

---

## What's NO LONGER shipped

These intrinsics were removed from `@aardworx/wombat.shader-types`
because wombat.base provides equivalent methods:

| Removed | Use this |
|---|---|
| `length(v)` | `v.length()` |
| `dot(a, b)` | `a.dot(b)` |
| `cross(a, b)` | `a.cross(b)` (V3f only) |
| `distance(a, b)` | `a.distance(b)` |
| `normalize(v)` | `v.normalize()` |
| `abs(vec)` | `v.abs()` (scalar `abs(x)` still works) |
| `sign(vec)` | `v.sign()` (scalar `sign(x)` still works) |
| `floor(vec)` | `v.floor()` (scalar still works) |
| `ceil(vec)` | `v.ceil()` (scalar still works) |
| `round(vec)` | `v.round()` (scalar still works) |
| `fract(vec)` | `v.fract()` (scalar still works) |
| `mod(vec, …)` | `v.mod(…)` (scalar still works) |
| `min(a, b)` for vectors | `a.min(b)` (scalar still works) |
| `max(a, b)` for vectors | `a.max(b)` (scalar still works) |
| `clamp(v, lo, hi)` for vectors | `v.clamp(lo, hi)` (scalar still works) |
| `mix(a, b, t)` for vectors | `a.lerp(b, t)` |

The shader frontend's method translator routes each method call to
the same IR the free-function form used to produce, so emitted
shader source is unchanged.

---

## Style guide

Use the **method form** for any vector or matrix expression:

```ts
const c = a.add(b).normalize().lerp(d, t);
const len = (p1.sub(p0)).length();
const m = M44f.fromCols(c0, c1, c2, c3).mul(view);
```

Use the **free-function form** for scalar expressions:

```ts
const angle = atan2(y, x);
const t = clamp(scalar, 0, 1);
const v = sin(time) * 0.5 + 0.5;
const d = step(0.5, scalar);
```

The IR is identical either way; this is purely a readability and
consistency choice — *and* the only way to spell vector ops since
the redundant free-function declarations were removed.
