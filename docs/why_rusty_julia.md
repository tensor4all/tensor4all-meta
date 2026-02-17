# Why Rust for Julia tensor network developers

This note explains why we are building **tenferro-rs** as a Rust core with a C API, even when the primary users may be Julia tensor network libraries.

A recurring theme is *feedback rate*: how many edit→run→observe cycles a developer (human or AI agent) can complete per unit time. As agentic coding becomes mainstream, feedback rate is a first class design constraint, and it shapes many of the choices below.

Repository: https://github.com/tensor4all/tenferro-rs
API docs: https://tensor4all.org/tenferro-rs/

## What tenferro-rs is trying to be

tenferro-rs is a layered tensor computation engine intended to be a **shared low level backend** across tensor network ecosystems.

It provides dense tensors, `einsum` with contraction tree optimization, linear algebra decompositions (SVD/QR/LU), automatic differentiation primitives (VJP/JVP), and a stable C API so other languages can reuse the same kernels.

The goal is not "another tensor library with another frontend API", but a backend that multiple communities can rely on and codevelop.

## Why not implement everything in pure Julia?

Pure Julia is great for research velocity and high level experimentation, and we want Julia to remain the primary language users write. However, maintaining a large, performance critical Julia codebase requires constant attention to type stability, precompilation, and invalidation. Rust's static typing and ahead of time compilation guarantee that correct code is performant without managing these complexities.

For the narrow backend layer that needs sustained optimization and testing, a statically compiled language is more practical, while Julia remains the right choice for the frontend.

## Why Rust

### 1) Higher “feedback rate” for human + AI development

As introduced above, *feedback rate* measures how many edit→run→observe cycles you can complete per unit time.
In agentic coding, this is often the limiting factor: the faster you get reliable feedback (tests, benchmarks, examples), the faster you iterate.

A compiled Rust core has almost no runtime compilation overhead, which keeps the edit and test loop tight for both humans and AI agents. In contrast, large Julia codebases can pay substantial latency for precompile and invalidation, which directly lowers feedback rate even when runtime performance is excellent.

This is not a criticism of Julia (Julia is great for exploration), but for a shared backend meant to be continuously optimized and regression tested, high feedback rate matters.

### 2) One core implementation across languages reduces ecosystem fragmentation

Tensor network tooling tends to fragment: multiple packages reimplement the same contractions, decompositions, and AD glue.
Even with excellent contributors, the maintenance burden compounds.

A shared core means optimization effort (einsum planning, kernel fusion, memory layout, GPU) is paid once, and correctness improvements benefit every host language. A stable C ABI enables long lived integration without tight coupling.

The reason to write this shared core in Rust rather than C or C++ is maintainability. Rust's compiler enforces memory safety and ownership at compile time, so the kind of subtle bugs that plague large C/C++ codebases (use after free, data races, double free) are caught before they reach users. The FFI surface will inevitably grow large, but the key advantage is that only the thin boundary layer requires manual attention; the rest of the codebase is protected by Rust's safety guarantees.

### 3) Packaging and versioning are simpler with a Rust workspace

Rust's workspace system lets you manage multiple related crates in a single repository and publish them to the registry together. In the Julia ecosystem, coordinating version bumps across multiple packages spread over separate repositories is not straightforward: registry registration is not instant, and synchronizing a batch of interdependent package upgrades can be tedious. AI agents can help with the mechanical parts, but the underlying workflow friction remains. A Rust workspace sidesteps this by keeping everything in one place with a single CI pipeline.

### 4) Stable AD primitives that can serve multiple hosts

Tensor network AD often needs careful control over primitives and performance.
If the same VJP/JVP primitives exist in one backend, multiple ecosystems can share them.

In Julia, we can integrate via ChainRules style rules in the wrapper layer. In Python, we can integrate via JAX custom_vjp/jvp or PyTorch autograd adapters. The key is that the **primitive implementation is shared**, even if each host exposes it differently.

Rust is a good fit here because the resulting binary has no JIT and no GC, so any language can call AD primitives via C FFI without pulling in a heavy runtime. For example, correct AD rules for complex SVD are nontrivial, and some existing implementations get them wrong. Having one well tested implementation of these delicate rules in a small, self contained binary benefits every host language.

## Why not just use libtorch (or Torch.jl)?

We see libtorch/Torch.jl as excellent tools, especially for deep learning workloads, but they are not a perfect fit for the role we want a tensor network backend to play.

The main reason is that libtorch's autograd is tightly coupled to its own Tensor type with fixed dtypes. It cannot natively differentiate through custom algebraic types (e.g., symmetric tensors with quantum number labels, block sparse structures). Tensor network workloads often need AD over such types, which is why tenferro-rs implements its own AD primitives from the ground up.

## Why the two-language problem does not come back

Combining two languages historically invites the "two-language problem": painful builds, fragile FFI wrappers, and version mismatches. Combining C++ with Python or Julia is something many of us would never want to do again — distributing libraries, managing dependencies, and coordinating releases becomes difficult in countless ways.

Rust and Julia sidestep most of these issues for two reasons:

1. **Interop through a plain C ABI.** Rust can export `extern "C"` functions with no runtime dependency (no GC, no JIT). Julia calls them via `ccall` with zero overhead. The boundary is thin and stable.
2. **Both have excellent package systems.** Rust's `cargo publish` is instant and atomic; Rust workspaces keep interdependent crates in sync. On the Julia side, `RustToolChain.jl` automates the Rust build step inside Julia's package manager, so end users never touch `cargo` directly.
3. **Memory safety makes debugging tractable.** A segfault inside a C++ backend is notoriously hard to diagnose, even for AI agents. In Rust, `unsafe` is confined to the thin C-FFI boundary, so crashes are rare and easy to localize.

In short, the combination does not recreate the pain of C++ interop because both ecosystems were designed for composable, versioned packaging from the start.

## Shifting the Julia/Rust boundary

As AI agents and LLMs continue to improve, the way we interact with scientific computing code will change. Today, Julia serves as both the exploration layer and the orchestration layer. In the future, much of the orchestration (trial and error, hyperparameter search, workflow composition) will increasingly be handled by natural language interfaces driving AI agents directly. When that happens, the role of a dynamic programming language as a bridge between humans and machines is replaced by natural language, and the Julia surface layer shrinks.

One might argue that, for now, AD can be delegated to Julia (e.g., Zygote.jl) while the Rust backend handles only tensor computation. But as the boundary shifts, that Julia runtime dependency becomes unnecessary baggage. Building AD and contraction planning into the compiled backend from the start avoids a costly migration later.

Fortunately, moving code between Julia and Rust is smoother than one might expect, because the two type systems are surprisingly similar. Neither language has classical OOP inheritance; both use trait based polymorphism (multiple dispatch + abstract types in Julia, traits in Rust), parametric types with type constraints (`where T <: Number` vs `where T: Num`), and composition over inheritance. This means abstractions written in Julia often translate almost directly into Rust traits and generics, making it practical to shift the boundary incrementally.

## References

- https://zenn.dev/h_shinaoka/articles/f8395a2c4f54d0 (On Julia's future: Developer Experience in the Era of Generative AI)
- https://github.com/SpM-lab/sparse-ir-rs
- https://github.com/tensor4all/Tensor4all.jl