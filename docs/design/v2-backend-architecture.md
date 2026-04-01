# v2 Backend Architecture

**Date:** 2026-04-02
**Status:** Draft
**Repos:** tenferro-rs, tidu-rs
**Related:** `v2-ad-architecture.md`

---

## I. Overview

The AD engine (tidu2) produces `CompiledProgram<Op>` — a flat SSA IR of
primitive operations. This document describes how that IR is executed.

```
CompiledProgram<Op>
    │
    ├── Tier 1 only (Semiring Core)
    │     ├── Standard: DefaultBackend (CPU, BLAS)
    │     └── Tropical: DefaultBackend (CPU) or CUDA kernels (GPU, optimized)
    │
    └── Tier 1 + Tier 2 (Standard)
          ├── Phase 1: DefaultBackend (CPU, BLAS + LAPACK)
          └── Phase 2: StableHLO → IREE (CPU/GPU/TPU)
```

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

### Architecture

```
CompiledProgram<Op>
    │
    ├── check: all ops in Tier 1?
    │     yes → SemiringBackend<Alg>
    │             ├── Standard → DefaultBackend (BLAS)
    │             └── Tropical → DefaultBackend (CPU) or CUDA kernels (GPU)
    │
    └── no (has Tier 2 ops)
          └── StandardBackend
                ├── Phase 1: DefaultBackend (CPU, eager per-op)
                └── Phase 2: StableHLO → IREE (compiled, fused)
```

### Type erasure boundary

| Layer | Scalar type T | Algebra Alg |
|-------|--------------|-------------|
| Graph + AD (tidu2) | generic (via `Operand`) | generic (via `PrimitiveOp`) |
| DefaultBackend | generic (preserved) | generic (preserved) |
| IREE Backend | erased to DType enum | Standard only |

Type erasure happens **only at the IREE boundary**. Everything above is
fully generic.

### DefaultBackend (Phase 1)

Executes `CompiledProgram` instruction-by-instruction:

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

No optimization, no fusion. Correct and simple.

---

## V. IREE Integration

### Why IREE, not XLA directly

- IREE is officially replacing XLA's runtime (OpenXLA project)
- CMake build (lighter than Bazel)
- C API for Rust FFI
- Targets: CPU (LLVM), GPU (Vulkan/CUDA/ROCm), Apple Silicon (Metal)
- Same StableHLO input — XLA optimizations flow into IREE

### Compilation path

```
CompiledProgram
  → StableHLO IR (MLIR textual or bytecode)
  → IREE compiler:
      StableHLO → linalg dialect → vector dialect → LLVM IR / SPIR-V
                   ↑ tiling         ↑ vectorization    ↑ codegen
                   fusion here      SIMD here           target-specific
  → IREE runtime: execute compiled module
```

### JIT cache

```rust
// Cache key: (program hash, input shapes, dtypes)
let compiled = iree_cache.get_or_compile(prog, input_shapes);
compiled.execute(input_buffers);
```

Same program with same input shapes → cached compiled module.
Different shapes → recompile (like JAX's tracing).

### Memory space compatibility

| tenferro device | IREE HAL | Apple Silicon |
|----------------|----------|---------------|
| MainMemory | HOST_LOCAL | MTLBuffer shared |
| PinnedMemory | HOST_LOCAL \| DEVICE_VISIBLE | (same) |
| GpuMemory | DEVICE_LOCAL | MTLBuffer shared |
| ManagedMemory | DEVICE_LOCAL \| HOST_VISIBLE | (hw managed) |

Zero-copy possible when tenferro buffer and IREE buffer share the
same device memory.

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

### Phase 3: StableHLO + IREE

- `tenferro2-stablehlo`: primitive → StableHLO 1:1 lowering
- `tenferro2-iree-backend`: IREE compiler + runtime via C API
- JIT cache (program hash + shapes → compiled module)
- **Milestone**: Standard GPU execution via IREE, replacing deprecated CUDA kernels

### Phase 4: Optimization

- Operator fusion in StableHLO (IREE handles this)
- Memory optimization (IREE tiling + buffer reuse)
- Tropical GPU: retain hand-optimized CUDA kernels (not via IREE)

---

## Superseded Issues (partially)

- tenferro-rs#616: Traced Tensor + StableHLO IR (AD portions → `v2-ad-architecture.md`)
- tenferro-rs#618: tenferro v2 roadmap (backend portions here, AD portions → `v2-ad-architecture.md`)
