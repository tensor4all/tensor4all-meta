# Why Rust for Julia tensor network developers

This document makes two arguments:

1. **Use Rust as a backend.** Julia tensor network libraries can gain significant benefits — faster precompilation, higher development feedback rate, and cross-language reuse — by moving their compute and storage layers to a shared Rust backend via C-FFI, while keeping Julia as the frontend.

2. **Build this together.** No single group can sustain a healthy tensor ecosystem alone. A shared Rust core, co-developed across communities, lets optimization effort be paid once and correctness improvements benefit everyone.

Repository: https://github.com/tensor4all/tenferro-rs
API docs: https://tensor4all.org/tenferro-rs/

---

## Part I: Use Rust as a backend

### Is this for you?

If any of these sound familiar, a Rust backend can help:

- `using ITensors` takes minutes every time you start a new Julia session
- Your CI spends more time on precompilation than on actual tests
- You want to ship a library or a tool as a small, self-contained binary — without bundling the Julia runtime
- An AI agent edits your code, but then you wait minutes for Julia to recompile before you can see if it worked

|  | Pure Julia (ITensors.jl) | Rust backend (tenferro-rs) |
|--|--------------------------|----------------------------|
| Cold start | ~10 min | ~2 min (full build from scratch) |
| Edit→test cycle | minutes (recompilation) | tens of seconds |
| Binary distribution | Julia runtime required | single `.so` / `.dylib` |
| Migration cost | — | **nearly free with AI agents** (given a good design plan or a good existing codebase to port from) |

In an agentic coding workflow where you need dozens of edit→test cycles per hour, this difference is decisive.

> **In the era of AI agents, rewriting is nearly free. The bottleneck is no longer writing code — it is deciding what to build.**

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

### Why the two-language problem does not come back

Combining two languages historically invites the "two-language problem": painful builds, fragile FFI wrappers, and version mismatches. Combining C++ with Python or Julia is something many of us would never want to do again.

Rust and Julia sidestep most of these issues:

1. **Interop through a plain C ABI.** Rust can export `extern "C"` functions with no runtime dependency (no GC, no JIT). Julia calls them via `ccall` with zero overhead. The boundary is thin and stable.
2. **Both have excellent package systems.** Rust's `cargo publish` is instant and atomic; Rust workspaces keep interdependent crates in sync. On the Julia side, `RustToolChain.jl` automates the Rust build step inside Julia's package manager, so end users never touch `cargo` directly.
3. **Memory safety makes debugging tractable.** A segfault inside a C++ backend is notoriously hard to diagnose, even for AI agents. In Rust, `unsafe` is confined to the thin C-FFI boundary, so crashes are rare and easy to localize.

In short, the combination does not recreate the pain of C++ interop because both ecosystems were designed for composable, versioned packaging from the start.

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

Tensor network tooling tends to fragment: multiple packages reimplement the same contractions, decompositions, and AD glue. Even with excellent contributors, the maintenance burden compounds.

A shared Rust implementation of the fundamentals means optimization effort (kernel tuning, memory layout, GPU offloading) is paid once and available to every host language via C-FFI. Correctness improvements — especially for subtle AD rules — benefit everyone.

Rust's workspace system keeps all crates in a single repository with a single CI pipeline, and its compiler enforces memory safety at compile time, so the codebase stays maintainable as it grows.

### What becomes possible

Once these fundamental libraries are production-ready, any tensor network ecosystem can use them as a backend. Each ecosystem keeps its own abstractions (named indices, block sparse storage, quantum number grading) while delegating the heavy computation to the shared Rust core.

As a concrete example, we have analyzed how the ITensor Julia ecosystem could do this — see the proposal: [ITensor backend analysis](docs/design/itensor_backend_analysis.md)

### Invitation

tenferro-rs is open and designed to be co-developed. We welcome contributions from any tensor network community — whether that means improving the dense kernels, contributing AD rules, hardening the C API, or simply filing issues about what your ecosystem needs from a shared backend.

## References

- https://zenn.dev/h_shinaoka/articles/f8395a2c4f54d0 (On Julia's future: Developer Experience in the Era of Generative AI)
- https://github.com/SpM-lab/sparse-ir-rs
- https://github.com/tensor4all/Tensor4all.jl
