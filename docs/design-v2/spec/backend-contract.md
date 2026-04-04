# v2 Backend Architecture

**Date:** 2026-04-04
**Status:** Draft
**Repos:** tenferro-rs
**Related:** `../architecture/ad-pipeline.md`, `primitive-catalog.md`, `../reference/stablehlo-primitives.md`, `../reference/jax-primitives.md`

---

## I. Overview

**All computation — primal and derivatives — flows through the same pipeline:**

```
MaterializedGraph (after resolve + materialize_merge)
  │
  │ compile
  ▼
CompiledProgram (flat SSA of StableHLO IR)
  │
  │                     ┌─── single cut point ───┐
  │                     │                        │
  ├── XLA backend ──────┤  takes StableHLO IR    │
  │   (serialize to     │  directly              │
  │    MLIR, JIT)       └────────────────────────┘
  │
  └── All other backends
        │
        │ optimizing compiler (TransposeFolding, DotDecomposer,
        │                      LinalgCustomCallPassthrough)
        ▼
      Execution IR (StableHLO ops + BatchedGemm - DotGeneral)
        │               engine-produced data is column-major
        │
        │ generic execution engine
        │   structural ops    → common infrastructure
        │   elementwise ops   → standard kernels (faer/libm)
        │   indexing ops      → standard kernels
        │   semiring ops      → SemiringCore / SemiringFastPath trait
        │   linalg/custom     → kernel registry
        │
        ├── faer/BLAS backend (CPU, default)
        ├── Custom GPU backend (GPU, op-by-op, dynamic shapes)
        └── Custom algebra backends (Tropical, etc.)
              implement SemiringCore (batched_gemm + reduce_sum)
```

AD is a graph transformation (differentiate, transpose), not a separate
execution mode. AD-side transforms may remain fragmented until a final
`materialize_merge` produces the compile-time `MaterializedGraph`. A
primal-only computation follows the same path — just without AD transforms.
Even eager execution (`apply_op`-equivalent) is internally a single-node
materialized graph compiled and evaluated through this pipeline.

### 2-level IR architecture

The **StableHLO IR** is the single cut point between graph/AD and
all backends. For **standard algebra**, it contains the full Tenferro IR
vocabulary (`StdTensorOp` variants, most mapping 1:1 to a StableHLO op;
exceptions include composite lowerings like `Conj` and multi-output linalg
ops like `Svd`). This IR is serializable to StableHLO MLIR; XLA takes it
directly. For **custom algebra**, the same `StableHloOp` types are used but
ops have semiring-specific semantics (Add=⊕, Mul=⊗) — this IR is **not**
serializable to StableHLO MLIR and always goes through the optimizing
compiler → Execution IR path. The StableHLO program is
**layout-independent** -- input layout normalization is a pure runtime
concern handled by the execution engine, not an IR transformation.

The **Execution IR** is the output of the optimizing compiler. Input operands
may be contiguous with arbitrary axis ordering; the engine inspects strides at
dispatch time. Engine-produced intermediates and outputs are column-major
contiguous. Its op vocabulary is the **same as StableHLO, with one
substitution**: `DotGeneral` is replaced by `BatchedGemm` (produced by
DotDecomposer). All other ops -- elementwise, reductions, indexing,
structural, and `CustomCall` -- pass through from StableHLO unchanged.
The key optimization is DotGeneral decomposition; everything else is
executed as-is.

```
                  StableHLO IR
                  (the single cut point)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
       XLA backend                optimizing compiler
       (direct)                        │
                                       ▼
                                 Execution IR
                                       │
                              generic execution engine
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                           faer   custom GPU  custom algebra
```

### Backend comparison

| | faer/CPU | Custom GPU | XLA | Custom algebra |
|---|---|---|---|---|
| Input IR | Execution IR | Execution IR | StableHLO (direct) | Execution IR |
| Execution | op-by-op interpret | op-by-op interpret | JIT compile | op-by-op interpret |
| Dynamic shapes | yes | yes | no (recompile) | yes |
| Kernel fusion | none | none | yes | none |
| Tensor networks | yes | **primary target** | needs padding | yes |
| Dependencies | faer (Rust) | CUDA kernels | xla-rs (~200MB) | user kernels |

**Custom GPU** is the key backend for tensor network computations where
bond dimensions change dynamically. It interprets Execution IR op-by-op,
dispatching to CUDA kernels (reused from tenferro-rs v1). No JIT
compilation, so no recompile on shape change. Individual ops (matmul, SVD)
are large enough that fusion is not critical.

**XLA** is for workloads with stable shapes (ML training loops, fixed
batch size) where kernel fusion provides significant speedup. It takes
StableHLO IR directly (no optimizing compiler pass).

### Custom algebra backends

| Algebra | CPU | GPU |
|---------|-----|-----|
| Tropical | SemiringCore → custom kernels | SemiringCore → optimized CUDA kernels |

Custom algebras receive the same Execution IR as standard backends. They
implement the `SemiringCore` trait (`batched_gemm` + `reduce_sum`).
Structural ops (`Permute`, `Reshape`) are handled by common
infrastructure. This means a new algebra backend needs only two method
implementations to support any einsum-derived program.

---

## II. Primitive Vocabulary

**Owner:** [`primitive-catalog.md`](primitive-catalog.md) is the sole source
of truth for the Tenferro IR vocabulary, StableHLO lowering rules, Execution
IR dispatch categories, and backend trait contracts.

This document does not re-state op tables. See primitive-catalog.md for:
- Tenferro IR vocabulary (Section IV)
- StableHLO lowering rules (Section VI)
- Execution IR dispatch categories (Section III.3)
- Backend trait contracts: `SemiringCore`, `SemiringFastPath` (Section III.4)
- Frontend sugar and canonical lowering (Section VII)

---

## III. StableHLO Format and Tooling

### Serialization format

StableHLO uses **MLIR bytecode** as its portable format. 5-month forward/backward
compatibility guarantee. Also available as MLIR textual assembly (`.mlir`).

### Consuming StableHLO

| Approach | Dependencies | Notes |
|----------|-------------|-------|
| Full MLIR + StableHLO C API | LLVM/MLIR + StableHLO libs | Official. Heavy build (~75k LOC C++) |
| `melior` crate (Rust MLIR bindings) | LLVM/MLIR | Extend with StableHLO C API via FFI |
| Text `.mlir` output → xla-rs | xla-rs only | Let XLA parse. Simplest for XLA backend |
| Own Rust IR (no MLIR) | None | Map ~100 ops directly. Lightest. |

**Our approach**: CompiledProgram is our own Rust IR (StableHLO IR).
For non-XLA backends (faer, custom GPU, custom algebra), the optimizing
compiler lowers to Execution IR and the generic engine interprets it — no
MLIR needed. For the XLA backend, we emit `.mlir` text or use the xla-rs
builder API. Full MLIR dependency is avoided.

StableHLO's reference interpreter (~8.7k LOC) is a useful reference for
implementing our faer backend.

---

## IV. StableHLO Lowering

### Principle: mostly 1:1 mapping

Most `Instruction<Op>` variants in `CompiledProgram` map to exactly one
StableHLO op. `StdTensorOp` is flat — no `Semiring(SemiringOpKind::...)`
wrapping — which keeps lowering trivial. Documented exceptions include
composite lowerings (e.g., `Conj` -> 4 ops) and multi-output linalg ops
(e.g., `Svd` -> `custom_call` + `get_tuple_element` x N).

```rust
fn lower_instruction(inst: &Instruction<StdTensorOp>) -> StableHloOp {
    match &inst.op {
        StdTensorOp::Add => stablehlo::add(inst.inputs, inst.outputs),
        StdTensorOp::Mul => stablehlo::multiply(inst.inputs, inst.outputs),
        StdTensorOp::Neg => stablehlo::negate(inst.inputs, inst.outputs),
        StdTensorOp::DotGeneral(c) => stablehlo::dot_general(c, inst.inputs, inst.outputs),
        StdTensorOp::ReduceSum { axes } => stablehlo::reduce_sum(axes, inst.inputs, inst.outputs),
        StdTensorOp::Transpose { perm } => stablehlo::transpose(perm, inst.inputs, inst.outputs),
        StdTensorOp::Reshape { shape } => stablehlo::reshape(shape, inst.inputs, inst.outputs),
        StdTensorOp::BroadcastInDim { shape, dims } =>
            stablehlo::broadcast_in_dim(shape, dims, inst.inputs, inst.outputs),
        StdTensorOp::Exp => stablehlo::exponential(inst.inputs, inst.outputs),
        StdTensorOp::Cholesky => stablehlo::cholesky(inst.inputs, inst.outputs),
        StdTensorOp::Svd => stablehlo::custom_call("lapack_gesvd", inst.inputs, inst.outputs),
        StdTensorOp::Qr => stablehlo::custom_call("lapack_geqrf_orgqr", inst.inputs, inst.outputs),
        StdTensorOp::Eigh => stablehlo::custom_call("lapack_syevd", inst.inputs, inst.outputs),
        StdTensorOp::Solve => stablehlo::custom_call("lapack_getrf_getrs", inst.inputs, inst.outputs),
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
| `Tropical<f64>` | — | ❌ (StableHLO path unavailable; use custom backend on semiring-compatible subset) |

### Linalg lowering

See [`primitive-catalog.md` Section V](primitive-catalog.md) for the linalg
primitive table and StableHLO lowering rules. Key point: `Cholesky` is a
direct StableHLO op; all others (`SVD`, `QR`, `Eigh`, `Solve`) lower to
`stablehlo.custom_call`.

---

## V. Backend Dispatch

### Architecture: 2-level IR compilation

All programs first compile to a **CompiledProgram** containing
StableHLO IR. This is the single cut point between graph/AD
and all backends. From here, execution diverges:

Terminology: **Tier 1** = AD-closed graph core (`primitive-catalog.md` IV),
**Tier 2** = Standard arithmetic only (`primitive-catalog.md` V).

```
Graph → differentiate / transpose → merge → compile
    │
    ▼
CompiledProgram<Op>  (StableHLO IR — the single cut point)
    │
    ├── XLA backend (takes StableHLO IR directly)
    │     serialize to MLIR → JIT compile → LLVM/PTX → fused kernels
    │
    └── All other backends
          │
          │  optimizing compiler (algebra-agnostic passes)
          │    TransposeFolding  — fold Transpose into DotGeneral dim numbers
          │    DotDecomposer     — break multi-dim DotGeneral → BatchedGemm
          │    LinalgCustomCallPassthrough — pass linalg custom_call ops through
          │
          ▼
        Execution IR (stride-aware engine dispatch)
          │  StableHLO ops + BatchedGemm - DotGeneral
          │
          │  generic execution engine
          │    1. structural ops (Permute, Reshape, BroadcastInDim) → common infrastructure
          │    2. semiring elementwise (Add, Mul) → algebra-dependent: standard kernel OR Operand::add()/multiply()
          │    3. semiring ops (BatchedGemm, ReduceSum) → check SemiringFastPath, fall back to SemiringCore
          │    4. elementwise (Neg, Div, Exp, Log, ...) → standard kernels (faer/libm)
          │    5. comparison/selection (Compare, Select, Clamp) → standard kernels
          │    6. indexing (Gather, Scatter, Slice, ...) → standard kernels
          │    7. additional reductions (ReduceProd, ReduceMax, ReduceMin) → standard kernels
          │    8. CustomCall, Cholesky → kernel registry (LAPACK / cuSOLVER)
          │
          ├── faer backend (Standard CPU, default)
          │     SemiringCore → faer::mat_mul / BLAS dgemm
          │     Linalg custom_call → LAPACK routines
          │
          ├── Custom GPU backend (Standard GPU, dynamic shapes)
          │     SemiringCore → CUDA kernels (reused from v1)
          │
          └── Custom algebra backends (Tropical, p-adic, etc.)
                SemiringCore → user-provided kernels
                (semiring-compatible subset of Tier 1 only)
```

**Custom algebra programs may only use the semiring-compatible subset of
Tier 1.** Using Tier 2 ops (`Exp`, `SVD`, etc.) with a custom algebra is a
compile-time error, and even some Tier-1 ops (for example `Neg` or `Conj`)
may be unavailable when the algebra does not define them.

### Optimizing compiler

The optimizing compiler transforms StableHLO IR into Execution IR through
a sequence of algebra-agnostic passes:

| Pass | Purpose |
|------|---------|
| **TransposeFolding** | Fold chains of `Transpose` + `DotGeneral` into a single instruction with permuted dimension numbers |
| **DotDecomposer** | Break multi-contracting-dim `DotGeneral` into sequences that map to `BatchedGemm` |
| **LinalgCustomCallPassthrough** | Pass linalg `CustomCall` ops through to the Execution IR as-is |

Note: input contiguity checking is **not** a compiler pass. It happens at
eval() time as a runtime pre-processing step (see Memory layout section
below). Only truly non-contiguous data (memory gaps) is physically copied.
Contiguous data with arbitrary axis ordering is passed through as-is; the
engine handles stride differences at dispatch time.

These passes operate on shape metadata and instruction structure, not on
element values. They are shared by all non-XLA backends (faer, custom GPU,
custom algebra).

### Execution IR

The output of the optimizing compiler is a flat sequence of Execution IR
instructions. Input operands may be contiguous with arbitrary axis ordering.
The engine inspects strides and adjusts dispatch accordingly (e.g., BLAS
trans flags for transposed inputs, v1-style fusability checks on dimension
groups for BatchedGemm). Engine-produced intermediates and outputs are
column-major contiguous.

The Execution IR uses the **same op vocabulary as the StableHLO
IR**, with one substitution: `DotGeneral` is replaced by `BatchedGemm`
(produced by DotDecomposer). All other ops pass through from StableHLO
unchanged. The key optimization is DotGeneral decomposition; everything
else is executed as-is.

| Category | Ops | Engine dispatch |
|----------|-----|----------------|
| Semiring contraction | `BatchedGemm` | `SemiringCore::batched_gemm()` |
| Semiring reduction | `ReduceSum` | `SemiringCore::reduce_sum()` |
| Semiring elementwise | `Add`, `Mul` | Algebra-dependent: standard kernel for standard algebra, `Operand::add()`/`multiply()` for custom algebra |
| Elementwise arithmetic | `Neg`, `Conj`, `Div`, `Abs`, `Sign`, `Maximum`, `Minimum` | Standard kernel (faer) |
| Elementwise analytic | `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Pow`, `Expm1`, `Log1p` | Standard kernel |
| Comparison & selection | `Compare`, `Select`, `Clamp` | Standard kernel |
| Additional reductions | `ReduceProd`, `ReduceMax`, `ReduceMin` | Standard kernel |
| Indexing | `Gather`, `Scatter`, `Slice`, `DynamicSlice`, `Pad`, `Concatenate`, `Reverse` | Standard kernel |
| Structural | `Permute`, `Reshape`, `BroadcastInDim` | Common infrastructure |
| Linalg / extensibility | `Cholesky`, `CustomCall` | Kernel registry |

Structural ops (`Permute`, `Reshape`, `BroadcastInDim`) are handled by
**common infrastructure** shared across all backends. They are not part of
the custom backend contract. Note that `Copy` is not an Execution IR
instruction; only truly non-contiguous input data (memory gaps) is
physically copied at eval() pre-processing before IR entry.

### Backend traits

Backend traits follow the v1 pattern of required core + optional fast paths.

**`SemiringCore`** (required):

| Method | Execution IR instruction(s) covered |
|--------|----------------------------------|
| `batched_gemm` | `BatchedGemm` |
| `reduce_sum` | `ReduceSum` |

This is the **minimum contract** for a custom algebra backend. Implementing
only these two methods is sufficient to execute any einsum-derived program.

**`SemiringFastPath`** (optional):

| Method | Purpose |
|--------|---------|
| `contract` | Direct binary contraction; avoids decomposition to `BatchedGemm` + `ReduceSum` |
| `elementwise_mul` | Fast path for Hadamard products |
| `elementwise_add` | Fast path for semiring accumulation / fused patterns |

Fast-path methods can **absorb multiple Execution IR instructions** into a
single kernel call. The Execution IR and backend trait methods need **not** be
1:1; a fast-path method may pattern-match a subgraph of Execution IR instructions
and execute them as one fused operation.

### Generic execution engine

The generic execution engine is a simple interpreter that walks the Execution
IR instruction sequence and dispatches each instruction:

1. **Structural ops** (`Permute`, `Reshape`, `BroadcastInDim`) → common
   infrastructure.
2. **Semiring elementwise** (`Add`, `Mul`) → algebra-dependent: standard
   kernel for standard algebra, `Operand::add()`/`multiply()` for custom.
3. **Semiring contraction/reduction** (`BatchedGemm`, `ReduceSum`) →
   engine's `prepare` step inspects input strides, then dispatches via
   `SemiringFastPath` (if available) or `SemiringCore`.
4. **Standard elementwise** (`Neg`, `Div`, `Abs`, `Exp`, `Log`, etc.),
   **comparison/selection** (`Compare`, `Select`, `Clamp`), **additional
   reductions** (`ReduceProd`, `ReduceMax`, `ReduceMin`), and **indexing**
   (`Gather`, `Scatter`, `Slice`, etc.) → standard kernels (faer/libm).
5. **Linalg / extensibility** (`CustomCall`, `Cholesky`) → registered
   kernel registry (LAPACK/cuSOLVER).

For the full dispatch table, see
[`primitive-catalog.md`](primitive-catalog.md#iii3-execution-ir).
5. The engine first checks `SemiringFastPath` for an applicable pattern match.
6. If no fast path fires, the engine falls back to `SemiringCore` methods.

This design means a backend author can start with the minimum two-method
contract and add fast paths incrementally as performance needs arise.

### Buffer lifecycle: liveness analysis + buffer pool

The Execution IR is SSA (each slot is written once), but the execution engine
must manage Rust buffer ownership efficiently. The key mechanism is
**liveness analysis**: the compiler annotates each instruction input with
whether it is the **last use** of that slot.

```rust
struct ExecInstruction {
    op: ExecOp,
    inputs: Vec<usize>,     // input slot indices
    outputs: Vec<usize>,    // output slot indices
    last_use: Vec<bool>,    // last_use[i] = true if inputs[i] is consumed here
}
```

At execution time, the engine uses this annotation to decide consume vs borrow:

- **`last_use = true`**: the engine calls `slots[i].take()` (Rust ownership
  transfer). The operation can reuse the input buffer for its output if the
  layout is compatible, or return the buffer to a **buffer pool** for later
  reuse.
- **`last_use = false`**: the engine borrows `slots[i].as_ref()`. The buffer
  stays alive for downstream consumers.

```rust
for inst in &program.instructions {
    let input = if inst.last_use[0] {
        slots[inst.inputs[0]].take().unwrap()   // consume: buffer reusable
    } else {
        slots[inst.inputs[0]].as_ref().unwrap() // borrow: buffer stays
    };
    // ... execute op, possibly reusing input buffer ...
}
```

**Buffer pool**: freed buffers are returned to a per-engine pool
(`TensorBufferPool` in v1) and recycled for future allocations. This avoids
repeated heap allocation in tight loops (e.g., einsum over many contraction
steps).

The IR semantics remain purely functional (SSA). Consume/borrow decisions are
an execution engine implementation detail, invisible to the IR and backend
traits.

### Type erasure boundary

| Layer | Scalar type T | Algebra Alg |
|-------|--------------|-------------|
| Graph + AD (tidu) | generic (via `Operand`) | generic (via `PrimitiveOp`) |
| CompiledProgram (StableHLO IR) | generic (preserved) | generic (preserved) |
| Execution IR + custom backend | generic (preserved) | custom (preserved) |
| XLA backend | erased to DType enum | Standard only |

Type erasure happens **only at the XLA/StableHLO serialization boundary**.

### faer backend (default for Standard CPU)

Interprets Execution IR instruction-by-instruction, dispatching each compute
op through `SemiringCore` to the corresponding faer/BLAS routine. Linalg
`CustomCall` instructions in the Execution IR are dispatched to LAPACK
routines via a registered kernel registry. No XLA dependency.

```
Execution IR instruction  →   faer/BLAS dispatch
  BatchedGemm             →   faer::mat_mul / dgemm
  ReduceSum               →   elementwise sum
  Add, Mul, Neg, ...      →   elementwise kernel
  Exp, Log, Sin, ...      →   libm / faer analytic
  Compare, Select, ...    →   elementwise kernel
  ReduceProd, ReduceMax,  →   elementwise kernel
    ReduceMin
  Gather, Scatter, ...    →   indexing kernel
  Permute/Reshape/...     →   common infrastructure (memory ops)

CustomCall (Execution IR) →   LAPACK kernel registry dispatch
  Cholesky                →   dpotrf
  SVD                     →   dgesvd
  QR                      →   dgeqrf + dorgqr (or dungqr for complex)
  Eigh                    →   dsyevd
  Solve                   →   dgetrf + dgetrs
```

No fusion, no JIT. Op-by-op execution. Sufficient for most CPU workloads.

**Why faer is the default, not XLA CPU**: XLA's CPU backend (via
elixir-nx/xla) uses oneDNN only on x86_64 Linux. All other platforms
fall back to Eigen (unoptimized). faer has optimized kernels everywhere:

| Platform | XLA CPU backend | faer |
|----------|----------------|------|
| x86_64 Linux | oneDNN (fast) | fast (AVX/AVX-512) |
| aarch64 Linux | Eigen (slow) | fast (NEON/SVE) |
| aarch64 macOS | Eigen (slow) | fast (NEON) |
| x86_64 macOS | Eigen (slow) | fast (AVX) |

XLA is optional, primarily for GPU. Its CPU path is useful only for
fusion optimization, not raw kernel performance.

### Memory layout: stride-aware execution

`Tensor` allows **arbitrary strides** at the user level (zero-copy views for
permute, slice, reshape). At eval() time, input pre-processing checks memory
contiguity:

1. **Contiguous data** (including permuted-contiguous views from
   `tensor.permute()` or `.t()`, and contiguous slices): passed as-is with
   zero copy. The strides are preserved.
2. **Non-contiguous data** (memory gaps from slicing): physically copied to
   a contiguous buffer before execution.

No StableHLO ops are inserted for input normalization. The StableHLO program
is layout-independent -- the same compiled program is used regardless of input
strides. The compile cache needs no layout signature in its key.

The engine is stride-aware. Contiguous-but-permuted inputs are handled via
BLAS trans flags and v1-style fusability checks (see `prepare_one_operand`).
Only non-contiguous data (memory gaps) is physically copied.

```
User Tensor:      may have arbitrary strides
                          │
               eval() pre-processing:
                 contiguity check:
                   contiguous (any axis order) → pass directly, zero copy
                   non-contiguous (gaps)       → physical copy
                          │
                   program execution
                          │
                   StableHLO IR (layout-independent)
                          │
                   optimizing compiler (TransposeFolding, DotDecomposer, etc.)
                          │
                   Execution IR → stride-aware engine dispatch
                          │
                   engine-produced intermediates/outputs: column-major
```

**XLA backend input contract**: XLA accepts only dense contiguous buffers
(no stride concept). The XLA backend's eval() pre-processing therefore
**always copies to column-major contiguous** before uploading:

- Column-major contiguous → upload directly (zero host-side copy)
- Contiguous-but-permuted (e.g., from `.t()`) → copy to column-major, then upload
- Non-contiguous → copy to column-major, then upload

This is stricter than the Execution IR engine (which is stride-aware and
avoids copies for permuted views). The extra host-side reorder is negligible
because XLA is primarily for GPU, where the host→device PCIe transfer
dominates.

**Contract**: the final output of any backend is always a dense `Tensor` with
some runtime `Placement`. Internal intermediates may use backend-specific
layout, but this is invisible to the caller.

### Device management

CompiledProgram is **placement-agnostic** — it contains no `Placement`
information. Runtime placement is carried by `Tensor`.

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}
```

`resident_device` tells us which device owns or primarily addresses the memory.
`preferred_compute_device` is still a separate runtime execution hint on
`Tensor`.

**Phase 1: all inputs must share a resident device domain.**

The backend asserts this at `eval` time (not compile time, since
CompiledProgram has no placement info):

```rust
fn eval(&self, prog: &CompiledProgram, inputs: &[Tensor]) -> Vec<Tensor> {
    let resident = inputs[0].resident_device();
    for input in inputs {
        assert_eq!(
            input.resident_device(),
            resident,
            "all inputs must share the same resident device"
        );
    }
    // backend may normalize memory kinds internally if needed
}
```

**Cross-device execution** requires splitting into multiple programs:

```rust
let result_gpu = gpu_backend.eval(&prog1, &[a_gpu, b_gpu]);
let result_cpu = result_gpu.to_cpu();                          // explicit transfer to UnpinnedHost
let result2 = cpu_backend.eval(&prog2, &[result_cpu, c_cpu]);
```

**Phase 2 (future): Transfer op in CompiledProgram.**

CompiledProgram may include `Transfer(tensor, target_placement)` as a primitive
op. When lowering to StableHLO, Transfer ops split the program into chunks:

```
CompiledProgram:
  [op0, op1, Transfer(Device@cuda:0 → UnpinnedHost), op2, op3,
   Transfer(UnpinnedHost → Device@cuda:0), op4]

StableHLO lowering:
  chunk 1: [op0, op1]     → GPU backend
  transfer: Device@cuda:0 → UnpinnedHost
  chunk 2: [op2, op3]     → CPU backend
  transfer: UnpinnedHost → Device@cuda:0
  chunk 3: [op4]          → GPU backend
```

Transfer is NOT a StableHLO op — it exists only in CompiledProgram.
tenferro handles the insertion (manual or automatic placement planning)
and the splitting during StableHLO lowering.

tidu never knows about placement. It is purely symbolic.

### Backends live inside tenferro

All backends use tenferro's placement and transfer infrastructure
(allocation, placement-aware transfer, kernel dispatch). They cannot be
separated into standalone crates without duplicating that runtime layer:

```
tenferro/
  ├── tensor/          Tensor, placement model, transfer helpers
  ├── prims/           PrimitiveOp implementations
  ├── ir/
  │   ├── stablehlo/   StableHLO IR types + lowering
  │   ├── exec/        Execution IR types (StableHLO ops + BatchedGemm - DotGeneral)
  │   └── compiler/    Optimizing compiler passes
  │                      TransposeFolding, DotDecomposer, LinalgCustomCallPassthrough
  ├── engine/          Generic execution engine (walks Execution IR)
  ├── backend_cpu/     faer backend (SemiringCore impl, LAPACK dispatch)
  ├── backend_gpu/     Custom GPU backend (SemiringCore impl, CUDA kernels)
  └── backend_xla/     XLA backend (takes StableHLO directly, optional feature flag)
```

### Custom algebra backend (Tropical / custom algebra)

Custom algebra backends receive **Execution IR** (after the optimizing
compiler), not StableHLO IR directly. They implement `SemiringCore` (canonical
signature in [`primitive-catalog.md` Section III.4](primitive-catalog.md)).

The generic execution engine handles the dispatch loop, using liveness
annotations for buffer management (see "Buffer lifecycle" above):

```rust
fn eval_exec_ir<B: SemiringCore>(
    backend: &B,
    program: &ExecProgram,
    inputs: Vec<B::Operand>,
    pool: &mut BufferPool,
) -> Vec<B::Operand> {
    let mut slots: Vec<Option<B::Operand>> = vec![None; program.n_slots];
    for (&slot, val) in program.input_slots.iter().zip(inputs) {
        slots[slot] = Some(val);
    }
    for inst in &program.instructions {
        // Helper: consume (take) or borrow based on last_use annotation
        let get = |slots: &mut Vec<Option<B::Operand>>, idx: usize, last: bool| {
            if last { slots[idx].take().unwrap() }
            else    { slots[idx].as_ref().unwrap().clone() }
        };

        let result = match &inst.op {
            // Structural ops — common infrastructure
            ExecOp::Permute { .. } | ExecOp::Reshape { .. } |
            ExecOp::BroadcastInDim { .. } => {
                let input = get(&mut slots, inst.inputs[0], inst.last_use[0]);
                dispatch_structural(&inst.op, input, pool)
            }

            // Semiring contraction — SemiringCore dispatch
            ExecOp::BatchedGemm { .. } => {
                let lhs = get(&mut slots, inst.inputs[0], inst.last_use[0]);
                let rhs = get(&mut slots, inst.inputs[1], inst.last_use[1]);
                backend.batched_gemm(&lhs, &rhs, pool)
            }
            // Semiring reduction — SemiringCore dispatch
            ExecOp::ReduceSum { .. } => {
                let input = get(&mut slots, inst.inputs[0], inst.last_use[0]);
                backend.reduce_sum(&input, axes, pool)
            }

            // Semiring elementwise — algebra-dependent dispatch
            ExecOp::Add => dispatch_semiring_add(backend, &inst, &mut slots, pool),
            ExecOp::Mul => dispatch_semiring_mul(backend, &inst, &mut slots, pool),
            // (standard algebra → faer kernel; custom algebra → Operand::add()/multiply())

            // Elementwise — standard kernel (faer); standard algebra only
            ExecOp::Neg | ExecOp::Div | ExecOp::Abs | ExecOp::Sign |
            ExecOp::Maximum | ExecOp::Minimum | ... =>
                dispatch_elementwise(&inst.op, &inst, &mut slots, pool),

            // Analytic — standard kernel (libm/faer)
            ExecOp::Exp | ExecOp::Log | ExecOp::Sin |
            ExecOp::Cos | ExecOp::Tanh | ... =>
                dispatch_analytic(&inst.op, &inst, &mut slots, pool),

            // Comparison & selection
            ExecOp::Compare(_) | ExecOp::Select |
            ExecOp::Clamp =>
                dispatch_comparison(&inst.op, &inst, &mut slots, pool),

            // Additional reductions
            ExecOp::ReduceProd { .. } | ExecOp::ReduceMax { .. } |
            ExecOp::ReduceMin { .. } =>
                dispatch_reduction(&inst.op, &inst, &mut slots, pool),

            // Indexing
            ExecOp::Gather(_) | ExecOp::Scatter(_) |
            ExecOp::Slice(_) | ExecOp::DynamicSlice |
            ExecOp::Pad(_) | ExecOp::Concatenate { .. } |
            ExecOp::Reverse { .. } =>
                dispatch_indexing(&inst.op, &inst, &mut slots, pool),

            // Linalg / extensibility — kernel registry
            ExecOp::Cholesky | ExecOp::CustomCall { .. } => {
                dispatch_custom_call(&inst.op, &inst, &mut slots, pool)
            }
        };
        slots[inst.outputs[0]] = Some(result);
    }
    program.output_slots.iter().map(|&i| slots[i].take().unwrap()).collect()
}
```

A custom algebra backend needs only `batched_gemm` + `reduce_sum`.
Everything else is provided by tenferro's common infrastructure.

---

## VI. GPU Backend: XLA first, IREE later

### Strategy: XLA → IREE migration

Both XLA and IREE accept StableHLO as input. We use **XLA first** (mature,
prebuilt binaries available) and migrate to **IREE later** (future runtime).

```
CompiledProgram → StableHLO → XLA (Phase 3) → IREE (Phase 4+)
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

### Two-level caching

Graph construction and AD transforms are expensive. Caching happens at
two levels, both outside tidu:

```
Level 1 — CompiledProgram (user / tenferro responsibility):
  Expensive:  Graph → differentiate → merge → compile → CompiledProgram
  Cheap:      retain CompiledProgram, call eval() many times
  tidu does NOT cache. Caller retains the CompiledProgram.

Level 2 — XLA executable (tenferro-xla-backend):
  Expensive:  CompiledProgram → StableHLO → XLA compile → executable
  Cheap:      cache executable, call execute() many times
  Cache key:  (program hash, input shapes, dtypes)
  Note: no layout info in key — inputs are always column-major contiguous
```

```rust
// Level 1: user retains CompiledProgram
let prog = graph.compile(...);   // expensive, do once
prog.eval(&[2.0, 3.0]);         // cheap, do many times

// Level 2: XLA backend caches compiled executable
let xla_exec = xla_cache.get_or_compile(&prog, input_shapes);
xla_exec.execute(input_buffers); // cheap
```

### XLA limitations

XLA requires **fully static shapes** at compile time. Known issues:

- **Recompilation on shape change**: every new shape combination triggers
  a full re-trace + re-compile (seconds to minutes). Cache grows with the
  number of distinct shapes.
- **Dynamic shapes are experimental**: StableHLO supports bounded dynamic
  dimensions (`tensor<? x f64>`), but XLA resolves them to static shapes
  before compilation. Truly symbolic compilation (like PyTorch's SymPy
  approach) is not planned for XLA.
- **Tensor networks**: DMRG, TCI, adaptive rank methods change bond
  dimensions at each iteration → recompilation per step → impractical.
- **Workarounds** (all with tradeoffs):
  - Padding to max size (wastes compute and memory)
  - Shape bucketing (compile a few canonical shapes)
  - Drop to eager mode (lose XLA optimization)

This is why the Custom GPU backend exists: op-by-op execution on Execution
IR with no compilation step handles dynamic shapes natively, at the cost of
no fusion.

### Future: IREE migration

IREE is officially replacing XLA's runtime (OpenXLA project). When mature:
- Same StableHLO input — no change to our lowering code
- Additional targets: Vulkan, Metal (Apple Silicon native)
- CMake build (lighter than XLA's Bazel)
- C API for Rust FFI
- Compilation path: StableHLO → linalg → vector → LLVM IR / SPIR-V

### Memory kind compatibility

`resident_device` is tracked separately from `memory_kind`. The table below
describes only the canonical public memory kind names.

| Canonical tenferro memory kind | JAX / XLA / IREE realization | Apple Silicon notes |
|-------------------------------|------------------------------|---------------------|
| `UnpinnedHost` | `unpinned_host` or backend default host memory | `MTLBuffer` shared / host-visible allocation |
| `PinnedHost` | `pinned_host` | driver-pinned host allocation |
| `Device` | `device` | device-local or unified GPU allocation depending on backend |
| `Other("managed")` | backend-specific managed/unified memory kind | hardware-managed unified-memory realization |

---

## VII. Tensor Types and Operand

### Tensor (concrete data type)

`Tensor` is the user-facing type-erased tensor. Internally it is an enum
over concrete typed storage for each supported scalar type. `Tensor` allows
**arbitrary strides** — at eval() time, input pre-processing checks memory
contiguity: contiguous data (including permuted views) is passed as-is with
zero copy; only truly non-contiguous data (memory gaps) is physically copied.
No StableHLO ops are inserted for input normalization.

```rust
struct Placement {
    memory_kind: MemoryKind,
    resident_device: Option<ComputeDevice>,
}

enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

struct TypedTensor<T: Scalar> {
    buffer: Buffer<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,       // arbitrary strides allowed
    placement: Placement,
    preferred_compute_device: Option<ComputeDevice>,
}

enum Buffer<T> {
    Host(HostBuffer<T>),
    Backend(BufferHandle<T>),
}

enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}
```

### TracedTensor (graph-aware wrapper)

`TracedTensor` wraps `Tensor` with graph tracking for lazy evaluation and AD.
All operations are lazy — there is no eager mode.

```rust
struct TracedTensor {
    shape: Vec<usize>,
    dtype: DType,
    fragment: Arc<Fragment<StdTensorOp>>,  // graph info (always present)
    val: LocalValId,
    data: Option<Tensor>,               // Some for inputs / eval'd results
}
```

- `TracedTensor::from(Tensor)` creates a Fragment input node with `data = Some(...)`.
- Operations (einsum, exp, ...) build graph, return `TracedTensor` with `data = None`.
- `eval()` triggers compile (cached) + execute, fills in `data`, returns `&Tensor`.

See `../examples/tensor-api-pseudocode.md` for full usage examples.

### Operand and TensorData traits

The `Operand` trait and `TensorData` trait are **separate concerns**:

- **`Operand`** — pure algebra: the methods a tensor type needs for
  semiring-compatible computation (add, multiply, dot_general, reduce_sum).
  This is what the graph/AD stack and einsum are generic over.
- **`TensorData`** — buffer access: shape, strides, raw data access,
  construction from data. This is what the backend infrastructure uses
  for structural ops (permute, reshape, copy).

```rust
/// Pure algebra — what einsum and AD are generic over
trait Operand {
    fn zero(shape: &[usize]) -> Self;
    fn one(shape: &[usize]) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn dot_general(&self, other: &Self, config: &DotGeneralConfig) -> Self;
    fn reduce_sum(&self, axes: &[usize]) -> Self;
}

/// Buffer access — what backends use for structural ops
trait TensorData {
    type Scalar;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn data(&self) -> &[Self::Scalar];
    fn from_data(shape: Vec<usize>, data: Vec<Self::Scalar>) -> Self;
}
```

`Tensor` implements both. Custom algebra types (e.g., `TropicalTensor`)
implement both. The split means the graph/AD stack never needs to know
about buffer layout, and the backend structural-op infrastructure never
needs to know about algebra.

---

## VIII. Roadmap

### Phase 1: Minimal vertical slice

- `DotGeneral`, `Add`, `Mul`, `Neg`, `ReduceSum` (Tier 1 core)
- `SVD` primal (custom_call, LAPACK)
- 2-level IR: StableHLO IR + optimizing compiler + Execution IR
- `SemiringCore` trait: `batched_gemm` + `reduce_sum`
- faer backend (CPU, implements `SemiringCore`)
- Tropical: implement `SemiringCore` with existing CPU + CUDA kernels
- **Milestone**: einsum works for Standard + Tropical, SVD works

### Phase 2: Expand primitives

- Add Tier 2 ops: `Exp`, `Log`, `Sin`, `Sqrt`, `Reshape`, `Transpose`,
  `BroadcastInDim`, `Pad`, `Gather`, `Scatter`, `Slice`
- Linalg JVP rules in traced primitives (SVD, QR, Cholesky, Eigh)
- faer backend for all new ops (CPU)
- `SemiringFastPath` for hot patterns
- **Milestone**: full AD for linalg works (JVP + auto-transpose VJP)

### Phase 3: GPU backends

- Custom GPU backend: `SemiringCore` impl with CUDA kernels (reused from v1)
  - Op-by-op execution on Execution IR, no fusion, dynamic shapes
  - **Milestone**: tensor network on GPU with dynamic bond dimensions
- XLA backend: takes StableHLO IR directly (no optimizing compiler)
  - JIT compile, kernel fusion, static shapes
  - **Milestone**: ML-style workloads on GPU with fusion

### Phase 4: Optimization + IREE

- Optimizing compiler improvements (better transpose folding, memory reuse)
- Memory optimization in custom GPU engine
- IREE as future alternative to XLA (same StableHLO input)
- Tropical GPU: `SemiringCore` with hand-optimized CUDA kernels

---

## Superseded Issues (partially)

- tenferro-rs#616: Traced Tensor + StableHLO IR (AD portions → `../architecture/ad-pipeline.md`)
- tenferro-rs#618: tenferro v2 roadmap (backend portions here, AD portions → `../architecture/ad-pipeline.md`)
