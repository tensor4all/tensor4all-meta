# v2 Backend Architecture

**Date:** 2026-04-02
**Status:** Draft
**Repos:** tenferro-rs
**Related:** `v2-ad-architecture.md`

---

## I. Overview

**All computation — primal and derivatives — flows through the same pipeline:**

```
MaterializedGraph (after resolve + materialize_merge)
  │
  │ compile
  ▼
CompiledProgram (flat SSA, common to all algebras)
  │
  ├─ Standard ──→ StableHLO ──┬→ (1) faer/BLAS engine  (CPU, default)
  │                            ├→ (2) Custom GPU engine (GPU, op-by-op, dynamic shapes)
  │                            └→ (3) XLA              (GPU/TPU, JIT, static shapes)
  │
  └─ Custom ────→ Custom backend (Semiring Core only)
                   ├── CPU: custom kernels
                   └── GPU: optimized CUDA kernels
```

AD is a graph transformation (differentiate, transpose), not a separate
execution mode. AD-side transforms may remain fragmented until a final
`materialize_merge` produces the compile-time `MaterializedGraph`. A
primal-only computation follows the same path — just without AD transforms.
Even eager execution (`apply_op`-equivalent) is internally a single-node
materialized graph compiled and evaluated through this pipeline.

### Three Standard backends (all accept StableHLO)

```
CompiledProgram → StableHLO
                      │
                      ├── (1) faer/BLAS engine (CPU, default)
                      ├── (2) Custom GPU engine (GPU, op-by-op)
                      └── (3) XLA (GPU/TPU, JIT compiled)
```

| | (1) faer/CPU | (2) Custom GPU | (3) XLA |
|---|---|---|---|
| Execution | op-by-op interpret | op-by-op interpret | JIT compile |
| Dynamic shapes | ✅ | ✅ | ❌ recompile |
| Kernel fusion | none | none | yes |
| Tensor networks | ✅ | **✅ primary target** | △ padding needed |
| Large static shapes | slower | fast | fastest |
| Dependencies | faer (Rust) | CUDA kernels (Rust) | xla-rs (~200MB) |

**(2) Custom GPU** is the key backend for tensor network computations where
bond dimensions change dynamically. It interprets StableHLO op-by-op,
dispatching to custom CUDA kernels (reused from tenferro-rs v1). No JIT
compilation → no recompile on shape change. Individual ops (matmul, SVD)
are large enough that fusion is not critical.

**(3) XLA** is for workloads with stable shapes (ML training loops, fixed
batch size) where kernel fusion provides significant speedup.

### Custom algebra backend (separate, not StableHLO)

| Algebra | CPU | GPU |
|---------|-----|-----|
| Tropical | Custom backend → custom kernels | Custom backend → optimized CUDA kernels |

Tropical and other custom algebras bypass StableHLO entirely. They use
CompiledProgram (Tier 1 only) dispatched directly to custom kernels.

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
| `Dup` | 1 (→ 2 outputs) | `stablehlo.broadcast_in_dim` (duplicate) |
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

## III. StableHLO Format and Tooling

### Serialization format

StableHLO uses **MLIR bytecode** as its portable format. 5-month forward/backward
compatibility guarantee. Also available as MLIR textual assembly (`.mlir`).

### Consuming StableHLO

| Approach | Dependencies | Notes |
|----------|-------------|-------|
| Full MLIR + StableHLO C API | LLVM/MLIR + StableHLO libs | Official. Heavy build (~75k LOC C++) |
| `melior` crate (Rust MLIR bindings) | LLVM/MLIR | Extend with StableHLO C API via FFI |
| Text `.mlir` output → xla-rs | xla-rs only | Let XLA parse. Simplest for backend (3) |
| Own Rust IR (no MLIR) | None | Map ~100 ops directly. Lightest. |

**Our approach**: CompiledProgram is our own Rust IR. For backends (1) and (2),
we interpret CompiledProgram directly — no MLIR needed. For backend (3) XLA,
we emit `.mlir` text or use xla-rs builder API. Full MLIR dependency is
avoided.

StableHLO's reference interpreter (~8.7k LOC) is a useful reference for
implementing our faer backend.

---

## IV. StableHLO Lowering

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

## V. Backend Dispatch

### Architecture: two-level compilation

All algebras first compile to a **CompiledProgram** (Semiring Core + JAX Prims).
This is the common intermediate representation. Further lowering depends
on the algebra.

```
Graph → differentiate / transpose → merge → compile
    │
    ▼
CompiledProgram<Op>
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
| Graph + AD (tidu) | generic (via `Operand`) | generic (via `PrimitiveOp`) |
| CompiledProgram | generic (preserved) | generic (preserved) |
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
If performance is insufficient for specific patterns, **fast paths** can
be added that bypass StableHLO for hot operations (e.g., direct BLAS call
for fused matmul chains).

### Memory layout: CompiledProgram knows nothing about strides

CompiledProgram is purely logical — it describes WHAT to compute, not HOW
memory is laid out. Operations like `Transpose`, `Slice`, `BroadcastInDim`
are logical transformations, not memory operations.

`MakeContiguous` MAY exist in CompiledProgram as an optional hint, but is
**ignored on the Standard (StableHLO) path** — both faer and XLA backends
treat all tensors as contiguous. It is only meaningful on the **Custom
backend path** (e.g., Tropical) where strides are used internally:

```
CompiledProgram:   Transpose(A)     ← logical operation
                  │
       ┌─────────┴─────────┐
       ▼                    ▼
  StableHLO             faer backend
  all contiguous        stride-based (lazy)
  XLA optimizes:        faer optimizes:
   - layout assignment   - strides for transpose (zero-copy)
   - transpose folding   - contiguous only when needed
     into dot/conv         (internal, not in IR)
   - copy removal
```

Each backend independently optimizes memory layout. The user and CompiledProgram
see only logical operations. This is a clean separation of concerns.

**Contract**: the final output of any backend is always contiguous. Internal
intermediates may be non-contiguous (e.g., faer may defer transpose via
strides), but this is invisible to the caller.

### Device management

CompiledProgram is **device-agnostic** — it contains no
device or memory space information. Device placement is a runtime concern.

**Phase 1: all inputs must be on the same device.**

The backend asserts this at `eval` time (not compile time, since
CompiledProgram has no device info):

```rust
fn eval(&self, prog: &CompiledProgram, inputs: &[Tensor]) -> Vec<Tensor> {
    let device = inputs[0].device();
    for input in inputs {
        assert_eq!(input.device(), device, "all inputs must be on same device");
    }
    // execute all ops on `device`
}
```

**Cross-device execution** requires splitting into multiple programs:

```rust
let result_gpu = gpu_backend.eval(&prog1, &[a_gpu, b_gpu]);
let result_cpu = result_gpu.to_cpu();                          // explicit transfer
let result2 = cpu_backend.eval(&prog2, &[result_cpu, c_cpu]);
```

**Phase 2 (future): Transfer op in CompiledProgram.**

CompiledProgram may include `Transfer(tensor, target_device)` as a primitive op.
When lowering to StableHLO, Transfer ops split the program into chunks:

```
CompiledProgram:  [op0, op1, Transfer(GPU→CPU), op2, op3, Transfer(CPU→GPU), op4]

StableHLO lowering:
  chunk 1: [op0, op1]     → GPU backend
  transfer: GPU → CPU
  chunk 2: [op2, op3]     → CPU backend
  transfer: CPU → GPU
  chunk 3: [op4]          → GPU backend
```

Transfer is NOT a StableHLO op — it exists only in CompiledProgram.
tenferro handles the insertion (manual or automatic device placement)
and the splitting during StableHLO lowering.

tidu never knows about devices. It is purely symbolic.

### Backends live inside tenferro

All backends use tenferro's device management infrastructure (memory
allocation, CPU↔GPU transfer, kernel dispatch). They cannot be
separated into standalone crates without duplicating device management:

```
tenferro/
  ├── tensor/          Tensor, device management, memory spaces
  ├── prims/           PrimitiveOp implementations
  ├── stablehlo/       CompiledProgram → StableHLO lowering + chunk splitting
  ├── backend_cpu/     (1) faer engine (uses tenferro Tensor directly)
  ├── backend_gpu/     (2) Custom GPU engine (uses tenferro GpuTensor)
  └── backend_xla/     (3) XLA engine (optional feature flag)
```

### Custom backend (Tropical / custom algebra)

Executes CompiledProgram (Tier 1 ops only) directly:

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

This is why Custom GPU engine (2) exists: op-by-op execution with no
compilation step handles dynamic shapes natively, at the cost of no fusion.

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

## VII. Tensor Types and Operand

### Tensor (concrete data type)

`Tensor` is a concrete data struct holding a buffer, shape, strides, and dtype.
It is the runtime representation used by backends for actual computation.

```rust
struct Tensor {
    buffer: Buffer,         // owned or shared data
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,           // F32, F64, C32, C64
}
```

### TracedTensor (graph-aware wrapper)

`TracedTensor` is a separate struct that wraps graph node references for
tracing. It records ops into a graph for deferred compilation (JIT path).
It does not hold data — only shape and dtype metadata plus a graph node ID.

```rust
struct TracedTensor {
    node_id: NodeId,        // reference into the trace graph
    shape: Vec<usize>,
    dtype: DType,
}
```

- `Tensor`: immediate execution via DefaultBackend (eager mode)
- `TracedTensor`: records ops into a graph (used for JIT compilation path)

### Operand implementation

Both `Tensor` and `TracedTensor` implement the `Operand` trait with
StableHLO-aligned ops:

```rust
impl Operand for Tensor {
    fn reshape(&self, shape: &[usize]) -> Self { ... }
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self { ... }
    fn add(&self, other: &Self) -> Self { ... }
    fn multiply(&self, other: &Self) -> Self { ... }
    fn dot_general(&self, other: &Self, config: &DotConfig) -> Self { ... }
}
```

---

## VIII. Roadmap

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

### Phase 3: StableHLO backends

- `tenferro-stablehlo`: primitive → StableHLO 1:1 lowering
- `tenferro-stablehlo-cpu`: (1) faer/BLAS StableHLO interpreter (default CPU)
- `tenferro-stablehlo-gpu`: (2) Custom GPU StableHLO interpreter
  - Reuse CUDA kernels from tenferro-rs v1
  - Op-by-op execution, no fusion, dynamic shapes ✅
  - **Milestone**: tensor network on GPU with dynamic bond dimensions
- `tenferro-xla-backend`: (3) XLA via xla-rs (optional, ~200MB)
  - JIT compile, kernel fusion, static shapes
  - **Milestone**: ML-style workloads on GPU with fusion

### Phase 4: Optimization + IREE

- Memory optimization in custom GPU engine
- IREE as future alternative to XLA (same StableHLO input)
- Tropical GPU: retain hand-optimized CUDA kernels (not via StableHLO)

---

## Superseded Issues (partially)

- tenferro-rs#616: Traced Tensor + StableHLO IR (AD portions → `v2-ad-architecture.md`)
- tenferro-rs#618: tenferro v2 roadmap (backend portions here, AD portions → `v2-ad-architecture.md`)
