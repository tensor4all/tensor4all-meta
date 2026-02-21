# Why Rewrite First-Principles Codes in Rust

Large-scale legacy first-principles codes written in C and Fortran (hundreds of thousands of lines, hundreds of global variables) can be redesigned and reimplemented in Rust using AI agents. This solves the problems facing the first-principles computation community.

We use [OpenMX](https://www.openmx-square.org/) (a DFT code, ~390,000 lines of C) as a case study, but the strategy applies to any large-scale computational science code with similar characteristics.

No prior knowledge of Rust is required to read this document.

> **The hardest part of rewriting a first-principles code is not the rewriting itself, but finding the right abstractions.** Where to draw crate (module) boundaries. Which operations belong to a shared computational foundation and which are DFT-specific logic. What the Hamiltonian construction interface should look like. These are physics questions, not programming questions. Finding the right answers requires a fusion of three things: the intuition of first-principles physicists, the insights of researchers with cross-cutting experience spanning first-principles to tensor networks (such as Shinaoka et al.), and the rapid materialisation of candidate designs by AI agents.

---

## Part I: The Problem with Legacy First-Principles Codes

### Does this sound familiar?

If you maintain or use a large first-principles code, some of the following will ring true:

- **Only one person understands the build system.** Dependencies on MPI, BLAS/LAPACK, ScaLAPACK, FFTW, and ELPA require tacit knowledge to compile on new machines.
- **Nobody dares to touch the core.** The SCF loop, force calculation, and Poisson solver work, but the code is so entangled that a fix in one place can break another.
- **Tests are slow and shallow.** Only full integration tests exist ("does the total energy of bulk Si match?"). Unit tests do not exist because components cannot be isolated.
- **Adding new physics is painful.** Adding a new exchange-correlation functional requires editing 7 files and understanding 50 global variables.
- **The original developer is about to retire.** The bus factor is 1. When they leave, the code dies.

### Concrete example: OpenMX 3.9

OpenMX is a versatile DFT code based on pseudo-atomic localised basis functions, supporting collinear and non-collinear magnetism, O(N) methods, quantum transport (NEGF), LDA+U, hybrid functionals, and molecular dynamics. It is a serious production code used worldwide.

| Metric | Value |
|--------|-------|
| C source files | 346 |
| Header files | 36 |
| Fortran files | 4 (ELPA bindings) |
| Total C lines | 390,782 |
| Largest single file | `DFTD3vdW_init.c` (33,727 lines) |
| Global variables in `openmx_common.h` | 330+ |

All 346 source files include `openmx_common.h`. This header declares over 330 `extern` global variables covering atomic coordinates, basis function data, grid information, MPI distribution tables, SCF control flags, spin parameters, and Hubbard U values. Any function in any file can freely read or write them. Reading a function signature tells you nothing about what state it depends on or modifies.

The build system is a hand-written Makefile linking Intel MKL, ScaLAPACK, FFTW3, ELPA, MPI, and OpenMP. Seven different supercomputer configurations are commented out; getting it to compile on your machine requires manual editing.

This is not a criticism of OpenMX. It is the standard state of large first-principles codes that have evolved over decades. The physics is excellent. The software engineering is a product of its era.

#### GPU version: OpenMX 3.9.9

An analysis of the GPU-enabled variant (OpenMX 3.9.9 GPU) reveals the computational primitives that a DFT code demands from its linear algebra backend:

| Priority | Kernel | OpenMX usage | Call frequency |
|----------|--------|-------------|----------------|
| **Highest** | Symmetric/Hermitian eigenvalue | SCF bottleneck (`dsyev`/`zheev` on CPU, `cusolverDnXsyevd` on GPU) | 1000s per SCF |
| **Highest** | Matrix multiply (GEMM) | H-C, S-C, C^H-H-C etc. (`dgemm`/`zgemm`, `cublasDgemm`/`cublasZgemm`) | 100s per SCF |
| **High** | 3D FFT | Poisson equation (rho -> V_H via FFTW3/cuFFT) | A few per SCF |
| **High** | LU decomposition + inversion | Overlap regularisation, Green's functions | 10s per SCF |
| **Medium** | Cholesky decomposition | Overlap matrix (`dpotrf`) | A few per SCF |
| **Medium** | Sparse block operations | Hamiltonian/overlap construction | Every SCF step |
| **Medium** | MPI distributed parallel | Node-level distribution (`pdsyev`, `pzgemm`, ScaLAPACK) | All computation |
| **Low** | NEGF (Green's functions) | Transport properties | Special purpose |

The GPU version uses cuBLAS for GEMM, cuSOLVER for eigenvalue problems, OpenACC directives for data management, and CUDA streams for asynchronous execution. GPU acceleration targets the two most expensive kernels: eigenvalue decomposition and matrix multiplication.

### The status quo is unsustainable

The first-principles computation community faces these problems simultaneously:

1. **Code keeps growing.** Transport, optical properties, topological invariants, and machine-learning potentials are added one after another.
2. **Maintainers keep shrinking.** Original developers retire or move on.
3. **Hardware keeps diversifying.** CPU, GPU, ARM, and heterogeneous nodes must all be supported.
4. **User expectations keep rising.** Python interfaces, reproducible workflows, and cloud deployment are demanded.

Continuing to patch 390,000 lines of C code with 330 global variables is not a viable path.

---

## Part II: Why Rust (and Why Not C++)

### You do not need to learn Rust

The most important point, stated first.

In the age of AI agents (Claude Code, Cursor, GitHub Copilot, etc.), **humans do not need to write Rust line by line**. AI agents handle the mechanical complexity of ownership annotations, lifetime bounds, and trait implementations, while humans focus on **physics, algorithms, abstractions, and correctness**.

The Rust compiler verifies the code written by agents, catching memory bugs, data races, and type errors at compile time. It is a tireless automated reviewer that never misses an entire class of bugs.

What physicists **need to be able to do**:

- Read Rust function signatures (types, inputs, outputs) -- they resemble mathematical notation
- Judge whether an abstraction is physically correct
- Run `cargo test` and read the output
- Tell the AI agent what should change

What physicists **do not need to be able to do**:

- Memorise ownership rules
- Write lifetime annotations
- Debug borrow checker errors
- Understand the internals of trait dispatch

These are handled by AI and verified by the compiler.

### Why not C++

The question every computational physicist asks.

**C code compiles as C++ unchanged, so nothing improves.** Compiling OpenMX's 346 C files with a C++ compiler produces identical behaviour. C++ has safety features (RAII, smart pointers, `std::vector`), but the compiler does not enforce their use. Safety is opt-in. Legacy C code ported to C++ remains unsafe C code with a `.cpp` extension.

**Rust's safety is the default.** All code is memory-safe, and only places that need unsafe operations are marked with `unsafe {}`. All unsafe blocks are visible and auditable. As refactoring progresses, `unsafe` decreases and compiler guarantees increase.

**Mutable global state is a compile error.** In C/C++, 330 global variables are legal. In Rust, mutable global state requires `unsafe` or synchronisation primitives. Splitting into crates makes the 330-global-variable pattern physically unable to survive. The compiler eliminates it.

**Crate boundaries enforce modularity.** A Rust workspace consists of crates with explicit dependency declarations. Crate A cannot access crate B's private internals, and cyclic dependencies are forbidden. Splitting into 12 crates means the compiler **enforces** clean interfaces. In C++, header and library boundaries are build-system conventions.

**Incremental compilation works at crate granularity.** Changing one crate recompiles only that crate and its dependents. Modifying the exchange-correlation crate does not trigger recompilation of the Poisson solver or I/O layer. In C/C++ with a shared header like `openmx_common.h`, changing that header recompiles all 346 files.

**C++ lacks a standard package system.** C++ dependency management often relies on CMake, but there is no standard for how CMakeLists.txt should be written, and every project does it differently. Mechanically verifying that dependencies are correctly resolved is difficult. AI can verify this, but it takes time. With Rust's `cargo`, you declare dependencies in `Cargo.toml` and resolution, building, and version consistency verification complete instantly. A workspace of 12 crates builds with a single `cargo build`.

**AI agents are more effective in Rust than C++.** The cost of "writing" code is the same for AI in both languages. The difference is in "verification." When AI-written C++ code compiles, undefined behaviour and use-after-free may remain. When Rust code compiles, the entire class of memory safety bugs is eliminated. The productivity of the AI's edit-compile-test cycle improves dramatically.

---

## Part III: The Ecosystem is Ready

### External libraries

Every external library needed by a first-principles code is available from Rust.

| Library | Rust crate | Notes |
|---------|-----------|-------|
| BLAS/LAPACK | [blas](https://crates.io/crates/blas) / [lapack](https://crates.io/crates/lapack) | Supports OpenBLAS, MKL, etc. Runtime injection via [cblas-inject](https://crates.io/crates/cblas-inject) |
| MPI | [rsmpi](https://github.com/rsmpi/rsmpi) (crate: [mpi](https://crates.io/crates/mpi)) | MPI-3.1 compliant. Runtime loading via [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) |
| HDF5 | [hdf5](https://crates.io/crates/hdf5) | Thread-safe wrapper. Runtime loading via [hdf5-rt](https://github.com/tensor4all/hdf5-rt) |
| FFT | [fftw](https://crates.io/crates/fftw) / [rustfft](https://crates.io/crates/rustfft) | FFTW3 bindings or pure Rust implementation |
| CUDA | [cudarc](https://crates.io/crates/cudarc) | Safe wrapper for cuBLAS, cuSOLVER, cuSPARSE, cuRAND, etc. |
| OpenMP equivalent | [rayon](https://crates.io/crates/rayon) | Data-parallel library with nested parallelism and thread pool separation |

BLAS/LAPACK, MPI, HDF5, FFT, and CUDA are thin wrappers calling existing C/Fortran libraries. Rayon is a pure Rust implementation providing functionality equivalent to OpenMP's `parallel for`. Unlike OpenMP, nested parallelism is natural. Thread pools can be separated by purpose -- "MPI communication," "matrix operations," "I/O" -- enabling structured concurrency.

Custom CUDA kernels can also be embedded and compiled with nvcc, and parts can be written in C or C++ (including OpenMP parallelism) and called from Rust. Since Rust achieves equivalent performance this is normally unnecessary, but the option remains for reusing highly optimised existing kernels.

### Tensor operations: tenferro-rs

Dense tensor operations are the computational core that every first-principles code depends on. [tenferro-rs](https://github.com/tensor4all/tenferro-rs) is the Rust workspace being developed to provide this core.

- **Dense tensors**: strided views with zero-copy slicing
- **Einsum**: with contraction tree optimisation
- **Batched linear algebra**: SVD, QR, LU, Cholesky, eigenvalue decomposition, linear solve, matrix inversion
- **Automatic differentiation primitives** (VJP/JVP): with correct differentiation rules for complex SVD, QR, etc.
- **C-FFI**: DLPack interop
- **Device abstraction**: CPU and GPU (CUDA/HIP)

tenferro-rs is structured as a workspace of 13 crates, following the same pattern we propose for DFT codes.

#### Gap analysis against OpenMX GPU

We analysed the GPU-enabled OpenMX 3.9.9 source code to evaluate whether tenferro-rs provides a sufficient computational foundation for DFT applications. The analysis compared OpenMX's computational primitives against tenferro-rs's existing API surface.

**Already covered by tenferro-rs:**

| OpenMX need | tenferro equivalent |
|-------------|---------------------|
| Dense GEMM (`dgemm`/`zgemm`, `cublasDgemm`) | `TensorPrims::BatchedGemm`, einsum |
| Standard eigenvalue (`dsyev`/`zheev`, `cusolverDnXsyevd`) | `tenferro-linalg::eigen()` |
| SVD, QR, LU | `tenferro-linalg` |
| Complex128 arithmetic | `num-complex::Complex64` |
| GPU memory model | `LogicalMemorySpace::GpuMemory` |
| Async GPU execution | `CompletionEvent` on `Tensor<T>` |
| C FFI + DLPack | `tenferro-capi` |

**Added to tenferro-rs as a result of this analysis:**

| Operation | Rationale | Backend mapping |
|-----------|-----------|-----------------|
| `cholesky()` | Overlap matrix S factorisation (`dpotrf` in OpenMX) | faer (CPU), cuSOLVER (GPU) |
| `solve()` | General linear solve A-x = b (Green's functions, NEGF) | faer (CPU), cuSOLVER (GPU) |
| `inv()` | Explicit matrix inversion (LU-based, used in OpenMX for overlap regularisation) | Composed from `lu()` + `solve()` |

**Deferred -- to be introduced when application development requires them:**

| Feature | Why deferred | Where it belongs |
|---------|-------------|------------------|
| Generalised eigenvalue (`geig`: A-x = lambda-B-x) | Core of SCF loop but requires application-level validation first | `tenferro-linalg` |
| FFT (3D forward/inverse) | Poisson solver; orthogonal to tensor contraction; better served by external `rustfft` + `cuFFT` via `Tensor::view()` | Application layer or thin wrapper crate |
| Sparse / block-sparse matrices | Hamiltonian construction; application manages sparsity, passes dense blocks to tenferro | Application layer + external crates (`sprs`) |
| MPI / distributed parallel | Node-level distribution; tenferro stays single-node (CPU/GPU) | Application layer |
| ScaLAPACK-equivalent distributed solvers | Distributed eigenvalue (`pdsyev`/`pzheev`) | External crate (e.g. ELPA Rust binding) |

**Architectural assessment:** tenferro-rs's layered structure (device -> algebra -> prims -> tensor -> einsum/linalg -> capi) is natural and sufficient for DFT applications. No structural changes are needed. The `TensorPrims<A>` describe-plan-execute pattern maps directly to the cuTENSOR API used by GPU DFT codes, and the `CompletionEvent` mechanism naturally corresponds to CUDA stream management. Application-specific concerns (sparsity, MPI, FFT) belong in the application layer, keeping tenferro-rs slim as a general-purpose tensor library.

### Package distribution

Rust's package manager `cargo` automatically handles all pure Rust dependencies. No CMake, no `pkg-config`, no `configure`.

```bash
cargo build --release    # Download all dependencies and build everything
```

`cargo build --release` produces a single executable with no runtime dependencies. For supercomputers, `cargo vendor` copies all dependency source code locally, enabling fully offline reproducible builds. The Makefile with seven commented-out supercomputer configurations becomes unnecessary.

---

## Part IV: Rewrite Strategy

What we propose is not a verbatim translation of legacy code to Rust. Incrementally decomposing 390,000 lines of C into crates is neither realistic nor necessary.

**The original C code is left untouched.** The only modifications are adding code to output numerical data for verification. On the Rust side, the right abstractions are designed from scratch. This lets us skip entirely the long process of gradually refactoring a massive legacy codebase.

### Phase 1: API skeleton design (the most important phase)

The most important phase, and the one that cannot be automated.

The question is not "how to convert `Force.c` to Rust." The real question is **what are the right abstractions?** Where to draw crate boundaries. What data should flow between Hamiltonian construction and the eigenvalue solver. Whether kinetic and potential energy contributions should be separate types. Whether the density matrix should be in real space or reciprocal space at interface boundaries.

These are physics questions. Three kinds of expertise must fuse:

1. **The intuition of first-principles physicists.** DFT practitioners know that the overlap matrix and the Hamiltonian matrix play different roles in the generalised eigenvalue problem. They should probably belong to separate construction pipelines, even if OpenMX constructs them similarly. Only someone who understands the physics can judge whether an API "makes physical sense."

2. **Cross-disciplinary architectural insight.** Researchers like Shinaoka et al., who work in both first-principles and tensor networks, can identify patterns invisible from a single field. The dense tensor operations, einsum contractions, batched linear algebra, and AD primitives provided by [tenferro-rs](https://github.com/tensor4all/tenferro-rs) are not tensor-network-specific -- they are the same operations DFT codes need. Drawing the boundary between "shared computational foundation" and "DFT-specific logic" requires experience in both worlds.

3. **Rapid materialisation by AI.** An AI agent can generate compilable Rust code from an instruction like "separate kinetic and potential Hamiltonian construction into different traits" in seconds. The physicist does not need to write Rust. They examine the resulting API and say "this captures the physics" or "the spin structure is wrong." Function bodies are `todo!()`, so compilation is instant. The loop of discuss -> materialise -> `cargo check` -> evaluate -> redesign runs at the speed of thought.

Running `cargo doc --open` generates HTML documentation for all crates' public APIs even at the skeleton stage. Physicists can survey the overall structure in a browser, checking inter-crate dependencies, type definitions, and function signatures. No need to read code.

Crate structure and public interfaces (function signatures, trait definitions, data types) are defined with `todo!()` bodies.

```
openmx-rs/
├── openmx-basis/           # PAO basis functions, pseudopotentials
├── openmx-grid/            # Real-space grid operations
├── openmx-hamiltonian/     # Hamiltonian construction
├── openmx-xc/              # Exchange-correlation functionals
├── openmx-poisson/         # FFT-based Poisson solver
├── openmx-eigen/           # Eigenvalue solver (uses tenferro-rs)
├── openmx-mixing/          # SCF mixing (DIIS, Kerker, Pulay)
├── openmx-scf/             # SCF loop control
├── openmx-force/           # Forces and stress (or AD-based)
├── openmx-transport/       # NEGF quantum transport
├── openmx-md/              # Molecular dynamics
└── openmx-io/              # Input/output (OpenMX format compatible)
```

The skeleton compiles in seconds. With `todo!()` bodies, the redesign loop runs at the speed of thought. If the physicist rejects five API designs before lunch, the AI has materialised all five.

**A perfect abstraction is not required from the start.** The skeleton phase converges quickly, but if a crate boundary turns out to be wrong later, it can be moved. Restructuring 390,000 lines of C is terrifying -- you cannot know what will break, and there are no tests to detect it. In Rust, the combination of the type system and the test suite makes restructuring safe. The compiler detects interface violations at compile time, and tests detect numerical regressions at runtime.

### Phase 2: Building a reference test database

Before writing Rust implementations, define what "correct" means. Prepare sets of reference inputs and expected outputs for each crate. This data is generated from the original OpenMX.

This work is mechanical. OpenMX already has test input files (stored in `work/`). Once the input files are chosen, an AI agent can do the rest autonomously.

1. Add code to OpenMX's C source to output the density, Hamiltonian matrix elements, eigenvalues, XC potential, Hartree potential, forces, etc. after each SCF step
2. Run with standard input sets (bulk Si, molecules, surface slabs, magnetic systems)
3. Collect outputs into a reference database (HDF5 or similar)

The result is a per-crate test contract:

- `openmx-xc`: electron density on grid -> exchange-correlation energy and potential
- `openmx-poisson`: charge density -> Hartree potential
- `openmx-eigen`: Hamiltonian and overlap matrices -> eigenvalues
- `openmx-scf`: input structure -> total energy and forces after N SCF steps

Once tests exist, implementation becomes "make the tests pass." This is the task AI agents excel at.

### Phase 3: New Rust implementation (AI agents, in parallel)

Once the API and test contracts are established, AI agents write implementations crate by crate. Since crates are independent, multiple agents can work in parallel.

- Agent A implements `openmx-xc`
- Agent B implements `openmx-poisson`
- Agent C implements `openmx-mixing`

The original C code serves as an algorithmic reference, but is not translated verbatim. Correct algorithms are implemented anew on top of the correct abstractions. Incremental compilation ensures each agent's edit-compile-test cycle runs in seconds.

### Phase 4: Cross-validation against the original (AI agents, automated)

Here a decisive strength of first-principles codes comes into play. **DFT calculations are deterministic.** The same input produces the same output to floating-point precision. No random numbers, no Monte Carlo noise, no statistical error.

This determinism makes automated verification straightforward:

1. Feed the same input to OpenMX (C) and openmx-rs (Rust)
2. Compare intermediate quantities (total energy, eigenvalues, density matrix, forces) at each SCF step
3. When a mismatch is found, **binary search** for the cause

```
SCF step 5:
  total_energy:  C = -1234.56789012   Rust = -1234.56789012   match
  force[0].x:    C = +0.00123456      Rust = +0.00123489      mismatch

AI binary search: "Force mismatch. Testing individual contributions..."
  kinetic:       match
  Hartree:       match
  nonlocal:      mismatch -> "Phase convention differs for L=2 Gaunt coefficients.
                  OpenMX uses Condon-Shortley convention; Rust side
                  omits the (-1)^m factor. Fixing..."
  [Fix applied, recompile in 3 seconds, re-verify]
  nonlocal:      match
  force[0].x:    match
```

AI agents can run this binary search autonomously. The determinism of DFT means there is no ambiguity in each comparison. The problem of "it might just be noise" simply does not exist.

### What you get in the end

| Before (OpenMX C) | After (openmx-rs) |
|--------------------|--------------------|
| 390,000 lines in 346 files | Rust workspace (~12 crates) |
| 330 global variables | Zero global state; data flows through function arguments |
| Hand-written Makefile (7 supercomputer configs) | `cargo build --release` |
| Header change recompiles all files | Crate change recompiles only that crate |
| No unit tests | Each crate independently testable |
| Bus factor = 1 | AI agents can maintain the code |
| Hand-written force calculation (12,000 lines) | Replaceable by automatic differentiation |
| Adding a new XC functional requires editing 7 files | Implement one trait in `openmx-xc` |

**Numerical results are identical to the original**, verified by automated comparison.

---

## Part V: Human-AI Integrated Development

The strategy described here is not speculation -- it is a pattern that already works.

### Three roles

| Role | Performed by |
|------|-------------|
| **Design decisions** | |
| Determine the right abstractions | First-principles physicists |
| Judge whether crate boundaries are physically sound | First-principles physicists |
| **Architectural insight** | |
| Draw the boundary between shared foundation and DFT-specific logic | Cross-disciplinary researchers (first-principles + tensor networks) |
| Identify reusable patterns across fields | Cross-disciplinary researchers |
| Connect DFT crates to the tenferro-rs foundation | Cross-disciplinary researchers + AI |
| **Implementation and verification** | |
| New implementation in Rust | AI agents |
| Writing tests | AI agents |
| Binary-search verification against original | AI agents |
| Root-cause analysis of numerical mismatches | Physicists + AI |
| **Automated guarantees** | |
| Memory safety | Rust compiler |
| Absence of data races | Rust compiler |
| Enforcement of crate boundaries and interface contracts | Rust compiler |

### Why the fusion matters

Neither humans, AI, nor the compiler can accomplish this alone.

First-principles physicists cannot rewrite 390,000 lines, and may not realise that the eigenvalue solver they need is the same operation that tensor network researchers use daily. Researchers like Shinaoka et al. can identify this connection. From experience building both DFT and tensor networks, they know that dense tensor contractions, batched SVD, and AD primitives are shared infrastructure. But they cannot redesign and reimplement an entire DFT code alone. AI agents implement fast but cannot judge whether an abstraction is physically correct. The Rust compiler detects memory bugs and enforces modularity but knows no physics.

When the three come together, a complete loop emerges:

1. Physicist: "The overlap matrix and Hamiltonian should be built in separate pipelines"
2. Cross-disciplinary researcher: "The eigenvalue computation should go through tenferro-rs's batched linear algebra. It can share the same backend as tensor networks"
3. AI agent materialises compilable Rust traits and types in seconds
4. Compiler verifies interface consistency across all 12 crates
5. Physicist evaluates: "this captures the physics" or "the spin degrees of freedom are in the wrong place"
6. This loop runs in minutes, not months

> **In the age of AI agents, rewriting is nearly free. The bottleneck is not writing code, but finding the right abstractions and designs.** Finding those abstractions requires a fusion of first-principles intuition, cross-disciplinary architectural insight, and rapid iteration by AI. This document itself is a product of that fusion.

---

## References

- [Why Rust for Julia tensor network developers](why_rusty_julia.md): sister document for the tensor network community
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs): Rust tensor computation workspace (POC)
- [OpenMX](https://www.openmx-square.org/): DFT code used as case study
