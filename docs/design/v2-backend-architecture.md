# v2 Backend Architecture

**Date:** 2026-04-02
**Status:** Draft
**Repos:** tenferro-rs, tidu-rs
**Related:** `v2-ad-architecture.md`

---

## I. Overview

**All computation — primal and derivatives — flows through the same pipeline:**

```
Graph (primal, and/or AD-transformed)
  → compile → TenferroIR (flat SSA, common to all algebras)
      │
      ├── Standard → StableHLO → faer/LAPACK (default) or XLA (optional)
      └── Custom   → Custom backend (Semiring Core only)
```

AD is a graph transformation (differentiate, transpose), not a separate
execution mode. A primal-only computation follows the same path — just
without graph transformations. Even eager execution (`apply_op`-equivalent)
is internally a single-node graph compiled and evaluated through this
pipeline.

### Current GPU backend status

| Algebra | CPU | GPU |
|---------|-----|-----|
| Standard | DefaultBackend → BLAS/LAPACK | **Deprecated** — unoptimized CUDA kernels removed. CPU fallback until IREE. |
| Tropical | DefaultBackend → custom kernels | Custom CUDA kernels (**retained** — optimized, production-ready) |

Standard GPU will be reintroduced via StableHLO → IREE (Phase 3 of AD
architecture roadmap). Tropical GPU kernels are hand-optimized and remain
as-is.

---

## II. Primitive Set

### Tier 1 — Semiring Core

Sufficient for einsum-based computation. Compatible with custom algebraic
backends (tropical, p-adic, polynomial rings).

| Primitive | Inputs | StableHLO equivalent |
|-----------|--------|---------------------|
| `Add` | 2 | `stablehlo.add` |
| `Mul` | 2 | `stablehlo.multiply` |
| `Neg` | 1 | `stablehlo.negate` |
| `Scale(c)` | 1 | `stablehlo.multiply` (with broadcast constant) |
| `Conj` | 1 | `stablehlo.complex` (swap imag sign) |
| `Einsum` / `DotGeneral` | 2 | `stablehlo.dot_general` |
| `ReduceAdd` | 1 | `stablehlo.reduce` (sum) |
| `Transpose` | 1 | `stablehlo.transpose` |
| `Reshape` | 1 | `stablehlo.reshape` |
| `BroadcastInDim` | 1 | `stablehlo.broadcast_in_dim` |
| `Dup` | 1 | `stablehlo.broadcast_in_dim` (same) |

### Tier 2 — Standard = Core + JAX prims

Full JAX-compatible set for general-purpose differentiable programming.

**Transcendental:**

| Primitive | StableHLO |
|-----------|-----------|
| `Exp` | `stablehlo.exponential` |
| `Log` | `stablehlo.log` |
| `Sin` | `stablehlo.sine` |
| `Cos` | `stablehlo.cosine` |
| `Tanh` | `stablehlo.tanh` |
| `Sqrt` | `stablehlo.sqrt` |
| `Rsqrt` | `stablehlo.rsqrt` |
| `Pow` | `stablehlo.power` |
| `Expm1` | `stablehlo.exponential_minus_one` |
| `Log1p` | `stablehlo.log_plus_one` |

**Comparison & Selection:**

| Primitive | StableHLO |
|-----------|-----------|
| `Maximum` | `stablehlo.maximum` |
| `Minimum` | `stablehlo.minimum` |
| `Compare(dir)` | `stablehlo.compare` |
| `Select` | `stablehlo.select` |
| `Clamp` | `stablehlo.clamp` |
| `Abs` | `stablehlo.abs` |
| `Sign` | `stablehlo.sign` |

**Indexing & Structure:**

| Primitive | StableHLO |
|-----------|-----------|
| `Gather` | `stablehlo.gather` |
| `Scatter` | `stablehlo.scatter` |
| `Slice` | `stablehlo.slice` |
| `DynamicSlice` | `stablehlo.dynamic_slice` |
| `Pad` | `stablehlo.pad` |
| `Concatenate` | `stablehlo.concatenate` |
| `Reverse` | `stablehlo.reverse` |

**Reduction:**

| Primitive | StableHLO |
|-----------|-----------|
| `ReduceSum` | `stablehlo.reduce` (add) |
| `ReduceProd` | `stablehlo.reduce` (multiply) |
| `ReduceMax` | `stablehlo.reduce` (max) |
| `ReduceMin` | `stablehlo.reduce` (min) |

**Linalg (custom_call in StableHLO):**

| Primitive | Backend call |
|-----------|-------------|
| `SVD` | LAPACK/cuSOLVER `gesvd` |
| `Cholesky` | LAPACK/cuSOLVER `potrf` |
| `QR` | LAPACK/cuSOLVER `geqrf` |
| `Eigh` | LAPACK/cuSOLVER `syevd` |
| `Solve` | LAPACK/cuSOLVER `getrf` + `getrs` |

Linalg ops map to `stablehlo.custom_call` with a target name. The backend
dispatches to the appropriate LAPACK/cuSOLVER routine.

**Control flow (future):**

| Primitive | StableHLO |
|-----------|-----------|
| `Cond` | `stablehlo.if` |
| `Scan` | `stablehlo.while` (unrolled or looped) |
| `While` | `stablehlo.while` |

### tenferro-prims v1 → v2 mapping

tenferro v1 has ~134 ops. v2 reorganizes into Tier 1 + Tier 2:

- **Keep**: ops with direct StableHLO mapping (~85 ops, aligned with JAX lax)
- **Merge**: `Square` → `Mul(x,x)`, `Reciprocal` → `Div(1,x)` (compose from primitives)
- **Keep as-is**: `SemiringCore::BatchedGemm` → `DotGeneral`
- **Add**: `Reshape`, `Transpose`, `BroadcastInDim`, `Pad` (critical gaps from v1)

### Key gaps to fill from v1

| Op | Why needed |
|----|-----------|
| `Reshape` | Transpose rules, broadcasting |
| `Transpose` (permutation) | Transpose of DotGeneral |
| `BroadcastInDim` | Transpose of reduction, Dup |
| `Pad` | Transpose of Slice |
| `Sign`, `IsFinite` | Numerical stability in JVP rules |

---

## III. StableHLO Lowering

### Principle: 1:1 mapping

Each `Instruction<Op>` in `CompiledProgram` maps to exactly one StableHLO op.
No multi-op lowering, no pattern matching. This keeps lowering trivial.

```rust
fn lower_instruction(inst: &Instruction<TensorOp>) -> StableHloOp {
    match &inst.op {
        TensorOp::Add => stablehlo::add(inst.inputs, inst.outputs),
        TensorOp::Mul => stablehlo::multiply(inst.inputs, inst.outputs),
        TensorOp::Exp => stablehlo::exponential(inst.inputs, inst.outputs),
        TensorOp::DotGeneral(config) => stablehlo::dot_general(config, ...),
        TensorOp::SVD => stablehlo::custom_call("lapack_gesvd", ...),
        // ...
    }
}
```

### Type support

| Rust type | StableHLO type | Supported |
|-----------|---------------|-----------|
| `f32` | `f32` | ✅ |
| `f64` | `f64` | ✅ |
| `Complex<f32>` | `complex<f32>` | ✅ |
| `Complex<f64>` | `complex<f64>` | ✅ |
| `Tropical<f64>` | — | ❌ (Tier 1 only, DefaultBackend) |

### Linalg as custom_call

Linalg ops (SVD, QR, etc.) are not in StableHLO's standard op set.
They map to `stablehlo.custom_call` with:
- `call_target_name`: e.g., `"lapack_gesvd"`, `"cusolver_gesvd"`
- Backend-specific dispatch at runtime

JVP rules for linalg ops are expressed in Tier 1 + Tier 2 primitives
(matmul, add, div, etc.) — these DO map to StableHLO. So the JVP
computation is fully compilable even though the primal is a custom_call.

---

## IV. Backend Dispatch

### Architecture: two-level compilation

All algebras first compile to **TenferroIR** (Semiring Core + JAX Prims).
This is the common intermediate representation. Further lowering depends
on the algebra.

```
Graph → differentiate / transpose → merge → compile
    │
    ▼
TenferroIR  (= CompiledProgram<TenferroOp>)
    │         Semiring Core (Tier 1) + JAX Prims (Tier 2)
    │         common to ALL algebras
    │
    ├── Standard algebra (uses Tier 1 + Tier 2):
    │     → lower to StableHLO (Tier 2 ops map 1:1)
    │     ├── faer/LAPACK backend (default, lightweight, no XLA dep)
    │     │     interprets StableHLO, dispatches op-by-op to BLAS
    │     │     no fusion, but cached. fast path addable later.
    │     └── XLA backend (optional, ~200MB, GPU support)
    │           JIT compiles StableHLO → LLVM/PTX, fusion, optimization
    │
    └── Custom algebra — Tropical, p-adic, etc. (Tier 1 only):
          → Custom backend (implements Semiring Core ops only)
          ├── CPU: custom kernels
          └── GPU: hand-optimized CUDA kernels
```

**Custom algebra programs may only use Tier 1 (Semiring Core) ops.**
Using Tier 2 ops (Exp, SVD, etc.) with a custom algebra is a compile-time
error — these ops have no meaning outside Standard algebra.

### Type erasure boundary

| Layer | Scalar type T | Algebra Alg |
|-------|--------------|-------------|
| Graph + AD (tidu2) | generic (via `Operand`) | generic (via `PrimitiveOp`) |
| TenferroIR | generic (preserved) | generic (preserved) |
| Custom backend | generic (preserved) | custom (preserved) |
| StableHLO + faer/XLA | erased to DType enum | Standard only |

Type erasure happens **only at the StableHLO boundary**.

### faer/LAPACK backend (default for Standard CPU)

Interprets StableHLO IR instruction-by-instruction, dispatching each op
to the corresponding faer/BLAS/LAPACK routine. No XLA dependency.

```
StableHLO instruction    →   faer/BLAS dispatch
  stablehlo.add          →   elementwise add
  stablehlo.dot_general   →   faer::mat_mul / dgemm
  stablehlo.custom_call("gesvd") → LAPACK dgesvd
  ...
```

No fusion, no JIT. Op-by-op execution. Sufficient for most CPU workloads.
If performance is insufficient for specific patterns, **fast paths** can
be added that bypass StableHLO for hot operations (e.g., direct BLAS call
for fused matmul chains).

### Custom backend (Tropical / custom algebra)

Executes TenferroIR (Tier 1 ops only) directly:

```rust
fn eval<Op: PrimitiveOp>(prog: &CompiledProgram<Op>, inputs: &[Op::Operand]) -> Vec<Op::Operand> {
    let mut slots = vec![None; prog.n_slots];
    for (i, val) in prog.input_slots.iter().zip(inputs) {
        slots[*i] = Some(val.clone());
    }
    for inst in &prog.instructions {
        let ins: Vec<&Op::Operand> = inst.inputs.iter()
            .map(|&i| slots[i].as_ref().unwrap())
            .collect();
        let outs = inst.op.eval(&ins);
        for (slot, val) in inst.outputs.iter().zip(outs) {
            slots[*slot] = Some(val);
        }
    }
    prog.output_slots.iter().map(|&i| slots[i].take().unwrap()).collect()
}
```

---

## V. GPU Backend: XLA first, IREE later

### Strategy: XLA → IREE migration

Both XLA and IREE accept StableHLO as input. We use **XLA first** (mature,
prebuilt binaries available) and migrate to **IREE later** (future runtime).

```
TenferroIR → StableHLO → XLA (Phase 3) → IREE (Phase 4+)
                           ↑                ↑
                           available now     future replacement
```

StableHLO is the stable interface. Backend swap is transparent.

### Phase 3: XLA via xla-rs

**Prebuilt binaries**: `elixir-nx/xla` distributes precompiled XLA shared
libraries (~200MB) for all major platforms:

| Binary | Target |
|--------|--------|
| `xla_extension-*-x86_64-linux-gnu-cpu.tar.gz` | Linux x86_64 CPU |
| `xla_extension-*-aarch64-darwin-cpu.tar.gz` | macOS Apple Silicon |
| `xla_extension-*-x86_64-linux-gnu-cuda12.tar.gz` | Linux CUDA 12 |
| `xla_extension-*-x86_64-linux-gnu-cuda13.tar.gz` | Linux CUDA 13 |

**Rust bindings**: `xla` crate (xla-rs, by Laurent Mazare / Hugging Face)
wraps XLA's PjRt C API. Uses elixir-nx/xla binaries directly.

```rust
use xla::{PjRtClient, XlaBuilder};

let client = PjRtClient::gpu()?;       // or ::cpu()
let builder = XlaBuilder::new("my_program");
// ... build HLO from CompiledProgram ...
let executable = client.compile(&computation)?;
let result = executable.execute::<xla::Literal>(&[input_buffers])?;
```

### XLA's JIT compilation design

XLA does NOT bundle precompiled CUDA kernels. It **JIT-compiles at runtime**:

```
HLO → LLVM IR → PTX → cubin (at runtime)
```

CUDA libraries (`libcudart`, `libcublas`, `libcudnn`) are loaded via `dlopen`
from the user's system installation. This keeps the XLA binary compact (~200MB
vs libtorch's ~2GB).

| | libtorch CUDA | XLA CUDA |
|---|---|---|
| Size | ~2GB | ~200MB |
| CUDA kernels | precompiled, bundled | JIT at runtime |
| CUDA libraries | partially bundled | `dlopen` from system |

### JIT cache

```rust
// Cache key: (program hash, input shapes, dtypes)
let compiled = xla_cache.get_or_compile(prog, input_shapes);
compiled.execute(input_buffers);
```

Same program with same input shapes → cached compiled executable.

### Future: IREE migration

IREE is officially replacing XLA's runtime (OpenXLA project). When mature:
- Same StableHLO input — no change to our lowering code
- Additional targets: Vulkan, Metal (Apple Silicon native)
- CMake build (lighter than XLA's Bazel)
- C API for Rust FFI
- Compilation path: StableHLO → linalg → vector → LLVM IR / SPIR-V

### Memory space compatibility

| tenferro device | XLA / IREE | Apple Silicon |
|----------------|------------|---------------|
| MainMemory | HOST_LOCAL | MTLBuffer shared |
| PinnedMemory | HOST_LOCAL \| DEVICE_VISIBLE | (same) |
| GpuMemory | DEVICE_LOCAL | MTLBuffer shared |
| ManagedMemory | DEVICE_LOCAL \| HOST_VISIBLE | (hw managed) |

---

## VI. Tensor Types and Operand

### Tensor<T> enum

```rust
enum Tensor<T: Scalar> {
    Eager(EagerTensor<T>),     // holds data, T preserved
    Traced(TracedTensor),      // no data, T erased to DType
}
```

- `Eager`: immediate execution via DefaultBackend
- `Traced`: records ops into a graph (used for JIT compilation path)

### Operand implementation

`Tensor<T>` implements the `Operand` trait with StableHLO-aligned ops:

```rust
impl<T: Scalar> Operand for Tensor<T> {
    fn reshape(&self, shape: &[usize]) -> Self { ... }
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self { ... }
    fn add(&self, other: &Self) -> Self { ... }
    fn multiply(&self, other: &Self) -> Self { ... }
    fn dot_general(&self, other: &Self, config: &DotConfig) -> Self { ... }
}
```

### DynTensor (scalar type erased)

```rust
enum DynTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    C32(Tensor<Complex<f32>>),
    C64(Tensor<Complex<f64>>),
}
```

Type erasure at the user API level. Internally dispatches to typed tensors.

---

## VII. Roadmap

### Phase 1: Minimal vertical slice

- `DotGeneral`, `Add`, `Mul`, `Neg`, `ReduceAdd` (Tier 1 core)
- `SVD` primal (custom_call, LAPACK)
- DefaultBackend (CPU, BLAS)
- Tropical: reuse existing CPU + CUDA kernels
- **Milestone**: einsum works for Standard + Tropical, SVD works

### Phase 2: Expand primitives

- Add Tier 2 ops: `Exp`, `Log`, `Sin`, `Sqrt`, `Reshape`, `Transpose`,
  `BroadcastInDim`, `Pad`, `Gather`, `Scatter`, `Slice`
- Linalg JVP rules in traced primitives (SVD, QR, Cholesky, Eigh)
- DefaultBackend for all new ops (CPU)
- **Milestone**: full AD for linalg works (JVP + auto-transpose VJP)

### Phase 3: XLA backend

- `tenferro2-stablehlo`: primitive → StableHLO 1:1 lowering
- `tenferro2-xla-backend`: XLA via xla-rs + elixir-nx/xla prebuilt binaries
- JIT cache (program hash + shapes → compiled executable)
- **Milestone**: Standard GPU execution via XLA, replacing deprecated CUDA kernels

### Phase 4: Optimization + IREE

- Operator fusion (XLA compiler handles this)
- Memory optimization
- IREE migration: swap XLA backend for IREE (same StableHLO input)
- Tropical GPU: retain hand-optimized CUDA kernels (not via XLA/IREE)

---

## Superseded Issues (partially)

- tenferro-rs#616: Traced Tensor + StableHLO IR (AD portions → `v2-ad-architecture.md`)
- tenferro-rs#618: tenferro v2 roadmap (backend portions here, AD portions → `v2-ad-architecture.md`)
