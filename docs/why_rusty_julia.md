# Why Rust for Julia tensor network developers

This document makes two arguments:

1. **Use Rust as a backend.** Julia tensor network libraries can gain significant benefits — faster precompilation, higher development feedback rate, and cross-language reuse — by moving their compute and storage layers to a shared Rust backend via C-FFI, while keeping Julia as the frontend.

2. **Build the foundations together.** A small set of well-maintained low-level Rust libraries — dense tensors, linear algebra, AD with custom type support — can serve as shared infrastructure for researchers worldwide. Once built, maintaining them is easy. Building them in the first place takes real effort and Rust+Julia know-how. Let's coordinate on that.

Repository: https://github.com/tensor4all/tenferro-rs
API docs: https://tensor4all.org/tenferro-rs/

---

## Part I: Use Rust as a backend

### Is this for you?

If any of these sound familiar, a Rust backend can help:

- Running all tests of your Julia package takes more than 10 minutes
- Your CI spends more time on precompilation than on actual tests
- You want to ship a library or a tool as a small, self-contained binary — without bundling the Julia runtime
- An AI agent edits your code, but then you wait minutes for Julia to recompile before you can see if it worked

For comparison: tensor4all-rs — a sizable Rust workspace containing quantics, TCI, and tree tensor network solvers — compiles the entire workspace from scratch in ~2 minutes. And that is the worst case. In practice, Rust's incremental compilation means only changed crates are recompiled, so a typical edit→test cycle takes tens of seconds. On CI, dependency build artifacts are cached, so only your own code is compiled on each run. For codebases at the scale of tensor network libraries, compilation is not a bottleneck.

> **In the era of AI agents, rewriting is nearly free. The bottleneck is no longer writing code — it is finding the right abstractions and designs.** And because rewriting is free, you do not need to get the design perfect upfront. You can iterate: review, refactor, redesign — cycles that used to take months now take hours.

### What tenferro-rs is trying to be

tenferro-rs is a layered tensor computation engine intended to be a **shared low level backend** across tensor network ecosystems.

It provides dense tensors, `einsum` with contraction tree optimization, linear algebra decompositions (SVD/QR/LU), automatic differentiation primitives (VJP/JVP), and a stable C API so other languages can reuse the same kernels.

The goal is not "another tensor library with another frontend API", but a backend that multiple communities can rely on and codevelop.

### Where Julia excels

Julia is hard to beat for experimentation and rapid prototyping. The code is concise and expressive — whether a human or an AI agent writes it, the same algorithm takes fewer lines than in most other languages. Multiple dispatch lets you compose abstractions naturally, and the REPL gives instant feedback. For small to medium codebases, this makes Julia one of the fastest languages to *think* in.

AI agents already demonstrate this pattern with other concise languages and tools. Watch an agent orchestrate a complex workflow: it chains Python one-liners for data manipulation, pipes `sed` and `awk` for text transformation, and drives `gh` CLI commands for GitHub operations — all fluently, because the tools are expressive and each invocation is short. Julia fits the same mold for scientific computing. An AI agent can write a DMRG sweep loop, define a Hamiltonian, or set up an index structure in a few lines of Julia, and the result is immediately readable and runnable. The combination of *concise language + AI generation* is powerful precisely when each piece of code stays small.

We want Julia to remain the primary language users write. The frontend — named indices, physics site types, algorithm orchestration — is where Julia's expressiveness matters most, and where code stays small.

### Why not implement everything in pure Julia?

The challenge appears as the codebase grows. Maintaining a large, performance critical Julia codebase requires constant attention to type stability, precompilation, and invalidation. Precompilation time scales with the complexity of the type hierarchy, and Julia's tensor network ecosystem has accumulated deep stacks of parametric types and metaprogramming that compile slowly.

For the narrow backend layer that needs sustained optimization and testing, a statically compiled language is more practical, while Julia remains the right choice for the frontend.

### Why Rust (and why now)

Until recently, asking computational physicists to maintain a large Rust codebase would have been unrealistic. Rust's ownership model and strict compiler are powerful, but the learning curve is steep for scientists whose primary job is physics, not systems programming. What changed is **agentic coding**: AI agents (Claude Code, Cursor, etc.) now handle the mechanical complexity of Rust (ownership annotations, lifetime bounds, trait implementations) while **humans focus on algorithms, abstractions, design, and correctness**. The compiler then validates everything the agent wrote, catching memory and concurrency bugs at compile time rather than at runtime.

#### Higher "feedback rate" for human + AI development

A compiled Rust binary has near-zero startup time with no JIT and no precompile wait. That means an AI agent can edit code, run the full test suite, and get feedback in seconds — a tight loop that is hard to achieve with large Julia codebases where precompilation latency is a bottleneck. We call this **feedback rate** (edit→run→observe cycles per unit time), and it is a first class design constraint.

This is not a criticism of Julia (Julia is great for exploration), but for a shared backend meant to be continuously optimized and regression tested, high feedback rate matters.

#### Eliminating precompilation overhead for end users

Beyond the developer's feedback rate, there is a direct user-facing benefit: **precompilation time reduction**. When the compute and storage layers move to Rust, the Julia packages that wrap them — with their complex type hierarchies, parametric generics, and metaprogramming — are eliminated from the precompilation pipeline entirely. The remaining Julia surface (user-facing API, algorithm loops) is thin and compiles fast.

#### Stable AD primitives across languages

Tensor network AD often needs careful control over primitives and performance. Correct AD rules for complex SVD, for example, are nontrivial, and some existing implementations get them wrong. Having one well tested implementation of these delicate rules in a small, self contained binary benefits every host language.

Rust is a good fit here because the resulting binary has no JIT and no GC, so any language can call AD primitives via C FFI without pulling in a heavy runtime. In Julia, we integrate via ChainRules style rules in the wrapper layer. In Python, via JAX custom_vjp/jvp or PyTorch autograd adapters. The **primitive implementation is shared**, even if each host exposes it differently.

Beyond primitives, tenferro-rs also provides a pure Rust tape based AD system that can compose these primitives into full computation graphs. Host languages can start by using individual primitives through their own AD frameworks (e.g., Zygote.jl, JAX), but as the Julia/Rust boundary shifts over time, high level AD orchestration can seamlessly move into the Rust backend as well.

#### Why not libtorch?

We see `libtorch` as an excellent tool for deep learning workloads, but libtorch's autograd is tightly coupled to its own Tensor type with fixed dtypes. It cannot natively differentiate through custom algebraic types (e.g., symmetric tensors with quantum number labels, block sparse structures). Tensor network workloads often need AD over such types, which is why tenferro-rs implements its own AD primitives from the ground up.

### The two-language problem is manageable

Combining two languages historically invites the "two-language problem": painful builds, fragile FFI wrappers, and version mismatches. The typical pain looks like this: a C++ library with its own CMake build system, a Python wrapper that must be compiled separately, system dependencies that may or may not be installed, and version conflicts that surface only at link time. The result is a build process that lives outside the host language's package manager and breaks in ways that are hard to diagnose.

Pure Rust avoids the worst of this, for two reasons.

**The Rust code integrates into the host's package system.** Pure Rust dependencies are resolved entirely by `cargo` — no system libraries, no CMake, no pkg-config. On the Julia side, [RustToolChain.jl](https://github.com/AtelierArith/RustToolChainExamples.jl) (or the approach used by [sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs)) builds the Rust code on the fly inside `Pkg.build()`, so users never touch `cargo` directly. For release distribution, the compiled binary can be packaged as a JLL artifact. Either way, the Rust code lives inside the host's package ecosystem, not beside it.

**C library dependencies are injected from the host, not linked at Rust build time.** When Rust code needs HDF5, BLAS, MPI, or CUDA, it does not link against system-installed C libraries at compile time. Instead, the libraries are obtained directly from the host language's own packages — HDF5.jl, MPI.jl, CUDA.jl, LinearAlgebra — and injected at runtime:

- [hdf5-rt](https://github.com/tensor4all/hdf5-rt): loads libhdf5 via `dlopen` at runtime. HDF5.jl already has libhdf5 loaded; hdf5-rt finds and reuses it.
- [cblas-inject](https://crates.io/crates/cblas-inject): BLAS function pointers are registered at runtime by the host. Julia's LinearAlgebra already has CBLAS loaded; the Rust code calls the same functions via injected pointers.
- [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt): MPI bindings with runtime loading. MPI.jl already has libmpi loaded; rsmpi-rt reuses it.
- CUDA: the same pattern applies. CUDA.jl manages the CUDA toolkit; the Rust code receives the runtime libraries from the host.

This pattern — pure Rust for all Rust dependencies, runtime injection for C/CUDA libraries from the host's packages — eliminates the build-time dependency hell that makes traditional two-language setups painful.

The two-language problem does not disappear entirely — bugs at the FFI boundary will still occur. But both Julia and Rust have strong versioning systems, so the classic headaches (version mismatches, ABI breakage) do not arise. What remains are ordinary bugs, and AI agents handle those routinely: detect the failure, read the error, apply the fix, run the tests.

### Shifting the Julia/Rust boundary

As AI agents and LLMs continue to improve, the way we interact with scientific computing code will change. Today, Julia serves as both the exploration layer and the orchestration layer. In the future, much of the orchestration (trial and error, hyperparameter search, workflow composition) will increasingly be handled by natural language interfaces driving AI agents directly. When that happens, the role of a dynamic programming language as a bridge between humans and machines is replaced by natural language, and the Julia surface layer shrinks.

Fortunately, moving code between Julia and Rust is smoother than one might expect, because the two type systems are surprisingly similar. Neither language has classical OOP inheritance; both use trait based polymorphism (multiple dispatch + abstract types in Julia, traits in Rust), parametric types with type constraints (`where T <: Number` vs `where T: Num`), and composition over inheritance. This means abstractions written in Julia often translate almost directly into Rust traits and generics, making it practical to shift the boundary incrementally.

---

## Part II: Let's build the fundamental Rust tensor libraries together

### Current status: proof of concept

tenferro-rs is currently a POC implementation. It demonstrates that the approach works — dense tensors, einsum, linear algebra, AD, and C-FFI — but it is not yet production-ready. The codebase needs hardening, broader scalar type support, thorough benchmarking, GPU backends beyond the abstraction layer, and a stable, well-documented C API surface.

### What we need to build

The Rust ecosystem lacks fundamental tensor computation libraries that the scientific computing community can depend on. These are not tensor-network-specific; they are general-purpose building blocks:

- **Dense tensor with strided views** — production-quality N-dimensional array with zero-copy slicing, robust memory management, and efficient layout
- **Einsum with contraction tree optimization** — general Einstein summation with automatic contraction ordering for arbitrary numbers of operands
- **Batched linear algebra** — SVD, QR, LU, eigen with batching, truncation, and support for real and complex scalars
- **AD primitives** — correct VJP/JVP rules for einsum, SVD, QR, and other decompositions (including the notoriously tricky complex SVD case)
- **Stable C-FFI** — a well-defined C API with opaque handles, DLPack interop, and clear lifecycle management, so any language can call these kernels
- **Device abstraction** — CPU and GPU execution behind a unified interface

These are the foundations. Higher-level constructs — block sparse tensors, diagonal tensors, symmetry sectors, graded tensors — are domain-specific and should be defined by each ecosystem (ITensor, TensorKit, etc.) on top of these fundamentals, either in Rust or in their host language.

### Why build them together

With AI agents, even a small group can maintain these libraries once they exist. The hard part is building them in the first place: getting the APIs right, accumulating Rust+Julia integration know-how (runtime dependency injection, C-FFI patterns, package system integration), and testing against real workloads. That initial effort benefits from coordination across groups who need the same foundations.

Once the low-level core is solid, many researchers worldwide can build on it — each ecosystem adding its own high-level constructs (index systems, block sparse storage, quantum number grading) without reimplementing dense contractions, decompositions, or AD from scratch.

### What becomes possible

As a concrete example, we have analyzed how the ITensor Julia ecosystem could use these foundations as a backend — see the proposal: [ITensor backend analysis](design/itensor_backend_analysis.md)

### Invitation

If you need the same foundations, let's coordinate. Whether that means improving the dense kernels, contributing AD rules, hardening the C API, or sharing Rust+Julia integration patterns — the goal is to build this initial layer together so everyone can move faster afterward.

## References

- https://zenn.dev/h_shinaoka/articles/f8395a2c4f54d0 (On Julia's future: Developer Experience in the Era of Generative AI)
- https://github.com/SpM-lab/sparse-ir-rs
- https://github.com/tensor4all/Tensor4all.jl
