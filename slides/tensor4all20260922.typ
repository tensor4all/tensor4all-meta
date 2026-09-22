// tensor4all meeting 2026-09-22: development update (May to September 2026)
// Author: Hiroshi Shinaoka
// Build: typst compile tensor4all20260922.typ

#set page(
  paper: "presentation-16-9",
  margin: (x: 2cm, y: 1.5cm),
  numbering: "1",
  footer: context [
    #h(1fr)
    #text(size: 14pt, fill: gray)[#counter(page).display()]
  ],
)

#set text(font: "Hiragino Sans", size: 18pt)

#let blue = rgb("#2563eb")
#let c-tenferro = rgb("#0f7bbf")
#let c-t4rs = rgb("#e94f37")
#let c-t4jl = rgb("#7b47a8")
#let c-green = rgb("#2aa16a")

#let title-slide(title, subtitle: none, author: none, date: none, acknowledgements: none) = {
  set align(center + horizon)
  block[
    #text(size: 44pt, weight: "bold")[#title]
    #if subtitle != none { v(0.5em); text(size: 26pt, fill: gray.darken(20%))[#subtitle] }
    #if author != none { v(1.5em); text(size: 24pt)[#author] }
    #if date != none { v(0.5em); text(size: 20pt, fill: gray)[#date] }
    #if acknowledgements != none { v(0.8em); text(size: 13pt, fill: gray)[#acknowledgements] }
  ]
}

#let slide(title, tag: none, color: blue, ..blocks) = {
  let body = blocks.pos().at(0)
  let refs = blocks.pos().at(1, default: none)
  pagebreak()
  block(width: 100%, height: 100%)[
    #grid(columns: (1fr, auto), align: (left, right + horizon),
      text(size: 26pt, weight: "bold", fill: color)[#title],
      if tag != none {
        box(fill: color, inset: (x: 10pt, y: 5pt), radius: 4pt, text(size: 13pt, fill: white, weight: "bold")[#tag])
      })
    #v(0.6em)
    #line(length: 100%, stroke: 0.5pt + color)
    #v(1.0em)
    #set text(size: 19pt)
    #set par(leading: 0.6em)
    #set list(spacing: 0.9em)
    #body
    #if refs != none {
      place(bottom + left, dy: 1.15cm, block(width: 88%, text(size: 11.5pt, fill: gray.darken(30%))[#refs]))
    }
  ]
}

#let code-block(code) = block(fill: rgb("#f1f5f9"), inset: 12pt, radius: 6pt, width: 100%, text(font: "Menlo", size: 13pt)[#code])
#let note(body) = text(size: 13pt, fill: gray.darken(30%))[#body]

// =====================================================================
#title-slide(
  "Tensor4all stack: development update",
  subtitle: "tenferro-rs · tensor4all-rs · hataori-rs · bindings",
  author: "Hiroshi Shinaoka",
  date: "tensor4all meeting, 2026-09-22",
  acknowledgements: "With S. Terasaki, R. Watanabe, L. Cheng, M. K. Ritter (ACI / TCI algorithms), S. Badr, K. Inayoshi, S. Dirnböck, M. Frankenbach, N. Ritz, J.-G. Liu, Y. Zhao and the HKUST(GZ) group",
)

// =====================================================================
#let stat(n, label) = align(center)[
  #text(size: 38pt, weight: "bold", fill: blue)[#n]
  #v(-0.7em)
  #text(size: 13pt, fill: gray.darken(30%))[#label]
]
#let bars(data, color: blue, maxv: 20) = {
  let h = 2.6cm
  grid(columns: data.len(), column-gutter: 5pt, align: bottom,
    ..data.map(((m, val)) => align(center)[
      #text(size: 11pt)[#val]
      #v(2pt)
      #rect(width: 100%, height: h * val / maxv, fill: color, radius: 2pt)
      #v(2pt)
      #text(size: 10pt, fill: gray.darken(20%))[#m]
    ]))
}
#slide("The community is growing")[
  #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 12pt,
    stat[34][contributors, whole org #linebreak() 29 in May],
    stat[12][contributors, Rust stack #linebreak() 7 in May],
    stat[78][stars on tenferro-rs #linebreak() 5 in May],
    stat[9][external repos using tenferro #linebreak() 0 in May],
  )
  #v(0.3em)
  #grid(columns: (1.1fr, 1fr), gutter: 28pt,
    [
      #text(size: 14pt, weight: "bold")[Cumulative contributors, Rust stack (2026)]
      #v(4pt)
      #bars((("Jan", 4), ("Feb", 4), ("Mar", 5), ("Apr", 7), ("May", 7), ("Jun", 9), ("Jul", 11), ("Aug", 12), ("Sep", 12)), color: c-tenferro, maxv: 13)
    ],
    [
      #set text(size: 14pt)
      #set list(spacing: 0.5em)
      - *tenferro-rs*: committers 3 → 6 since June; five new issue reporters from Jin-Guo Liu's group
      - *tensor4all-rs*: Lingrui Cheng landed 14 PRs; Samuel Badr, Ken Inayoshi joined
      - Dependents: yao-rs (Liu), TeNeT (Watanabe), latticeqcd-rs, hataori-rs
    ])
][
  Distinct commit authors since 2024 on GitHub, bots and forked repositories excluded; org total covers Julia, C++ and Rust repositories. Stars and dependents from the GitHub API on 2026-09-22.
]

// =====================================================================
#slide("Who contributed in 2026")[
  #set text(size: 11.5pt)
  #set par(leading: 0.4em)
  #let nw = text(fill: c-green, weight: "bold")[·]
  #grid(columns: (1.2fr, 1fr), gutter: 18pt,
    [
      #text(size: 14pt, weight: "bold", fill: c-tenferro)[Rust stack] #text(size: 11pt, fill: gray)[(#nw joined since May, ordered by first commit)]
      #v(2pt)
      #table(columns: (1.6fr, 1.1fr, 2.6fr), stroke: 0.4pt, inset: 3.8pt,
        [*Name*], [*Affiliation*], [*Repositories*],
        [Hiroshi Shinaoka], [Saitama], [all],
        [Satoshi Terasaki], [AtelierArith], [tenferro-rs, tenferro-benchmark, strided-rs, hataori-rs],
        [Jin-Guo Liu], [HKUST(GZ)], [omeinsum-rs, tenferro-rs],
        [Ken Inayoshi], [Saitama], [tensor4all-rs],
        [Nepomuk Ritz], [LMU], [tensor4all-rs],
        [Selina Dirnböck], [TU Wien], [tensor4all-rs],
        [#nw Xiwei Pan], [HKUST(GZ)], [omeinsum-rs],
        [#nw Ryo Watanabe], [Osaka], [tenferro-rs, strided-rs],
        [#nw Yusheng Zhao], [HKUST(GZ)], [omeinsum-rs, tenferro-rs],
        [#nw Samuel Badr], [TU Wien], [tensor4all-rs],
        [#nw Lingrui Cheng], [LMU / TUM], [tensor4all-rs, tensor4all-benchmark],
      )
    ],
    [
      #text(size: 14pt, weight: "bold", fill: c-t4jl)[Julia and C++ libraries]
      #v(2pt)
      #table(columns: (1.4fr, 2.4fr), stroke: 0.4pt, inset: 3.8pt,
        [*Name*], [*Repositories*],
        [Marc K. Ritter], [TensorCrossInterpolation.jl, AlternatingCrossInterpolation.jl],
        [Hirone Ishida], [tensorizingflows],
        [Martin Mikkelsen], [InterpolativeQTT.jl, Tensor4all.jl],
        [Nepomuk Ritz], [Tensor4all.jl],
        [Samuel Badr], [QuanticsGrids.jl, QuanticsTCI.jl],
        [Markus Frankenbach], [AlternatingCrossInterpolation.jl, TCIAlgorithms.jl],
        [T. Kloss, Y. Núñez Fernández, S. B., J. Fowkes], [xfac],
      )
      #v(4pt)
      #text(size: 11pt, fill: gray.darken(30%))[Issue reporters without commits: Guo P. Chen, LeoXia (HKUST(GZ)).]
    ])
][
  Distinct commit authors with at least one commit in 2026, bots excluded; one trivial single-commit author omitted.
]

// =====================================================================
#slide("The stack, and what this talk covers")[
  #set text(size: 16pt)
  #table(columns: (1.3fr, 4fr), stroke: 0.5pt, inset: 8pt,
    [*Layer*], [*Repository and role*],
    [Bindings], [#text(fill: c-t4jl, weight: "bold")[Tensor4all.jl] (Julia via C API) · #text(fill: c-t4jl, weight: "bold")[tensor4all-py] (PyO3 prototype)],
    [Tensor networks], [#text(fill: c-t4rs, weight: "bold")[tensor4all-rs]: TreeTN, TCI, ACI, quantics, partitioned networks],
    [Dense tensor engine], [#text(fill: c-tenferro, weight: "bold")[tenferro-rs]: tensors, sessions, einsum, linalg, FFT, autodiff; CPU / CUDA / WebGPU / Metal],
    [Parallel runtime], [#text(fill: c-green, weight: "bold")[hataori-rs]: dynamically scheduled `pmap` over threads and MPI ranks],
  )
  #v(0.8em)
  - Period: mid-May to mid-September 2026. Full features live in Rust; Julia and Python are thin entry points.
  - Plan: tenferro-rs (20 min) → tensor4all-rs (20 min) → hataori-rs, bindings, next steps (15 min).
]

// =====================================================================
#slide("Four months at a glance")[
  #set text(size: 16pt)
  #table(columns: (1.5fr, 1fr, 2.4fr), stroke: 0.5pt, inset: 8pt,
    [*Repository*], [*Merged PRs*], [*Releases*],
    [tenferro-rs], [about 350], [v0.2 (Jul) → v0.3 → v0.4 → *v0.5 (Sep 8)*, on crates.io],
    [tensor4all-rs], [about 115], [v0.2.0 (Jun)],
    [strided-rs], [about 65], [v0.3 (Jul) → v0.4 (Aug), on crates.io],
    [hataori-rs], [new], [first working engine (Aug)],
    [Tensor4all.jl], [10], [Windows CI],
  )
  #v(0.8em)
  #set text(size: 19pt)
  - Every release ships migration notes; GPU CI runs on RunPod.
  - Review, not typing, is the bottleneck: most code is written by agents under human gates (last slides).
]

// =====================================================================
#slide("tenferro-rs: sessions instead of per-call setup", tag: "tenferro-rs", color: c-tenferro)[
  - *Finding*: on small tensors the gap to PyTorch was per-operation overhead, not the kernels.
  - *Answer*: one session API shared by eager, autodiff and traced execution. Buffer pools, Rayon pool and BLAS handles live in the session.
  - tensor4all-rs now runs on an explicit CPU / CUDA execution context.
  #v(0.6em)
  #code-block[
    ```rust
    let cpu = CpuBackend::new()?;          // build once, reuse everywhere
    let s = cpu.session();                  // enter a shared execution scope
    let c = einsum_in(&s, "ij,jk->ik", [&a, &b])?;
    ```
  ]
]

// =====================================================================
#slide("tenferro-rs: linalg, einsum, GPU", tag: "tenferro-rs", color: c-tenferro)[
  - *Linear algebra*: full and compact SVD, incremental Householder QR, rank-revealing QR; derivative rules generated from linearization, with opt-in gauge conventions.
  - *Einsum*: strided-rs replay plans compiled once, ellipsis everywhere, TBLIS and grouped GEMM providers, a tropical (semiring) extension.
  - *CUDA*: cached cuTENSOR plans, cuFFT, on-device QR, asynchronous event contract. Unsupported ops return typed errors, never a silent CPU fallback.
  - *Portable*: WebGPU and Metal through CubeCL / CubeK, still experimental.
]

// =====================================================================
#slide("Benchmark: CPU einsum, Apple M5 Max", tag: "tenferro-benchmark", color: c-tenferro)[
  #set text(size: 15pt)
  #table(columns: (2.4fr, 1fr, 1fr, 1fr, 0.9fr, 1.1fr), stroke: 0.5pt, inset: 7pt, align: (left, right, right, right, right, right),
    [*Instance (ms)*], [*tenferro trace*], [*tenferro eager*], [*PyTorch*], [*JAX*], [*OMEinsum*],
    [matmul 1024], [8.4], [5.3], [*5.2*], [35.5], [40.7],
    [MERA closed network], [177], [190], [*160*], [848], [980],
    [batched likelihood (LM)], [*10.4*], [26.1], [18.4], [29.7], [85.8],
    [tensor-network permutation], [*123*], [156], [134], [212], [180],
  )
  #v(0.8em)
  #set text(size: 19pt)
  - Same BLAS (Accelerate), one thread, medians, refreshed mid-September.
  - Traced mode matches PyTorch on tensor-network-shaped instances; eager mode still pays dispatch on tiny ops.
][
  Full tables, GPU (A100) suites and run metadata: github.com/tensor4all/tenferro-benchmark
]

// =====================================================================
#slide("tensor4all-rs: what changed", tag: "tensor4all-rs", color: c-t4rs)[
  - *New algorithms*: randomized TreeTN contraction, cross interpolation native to trees, partitioned tree networks with reconstruction.
  - *Parallel*: adaptive interpolation runs on hataori-rs, Rayon by default and MPI opt-in.
  - *Quality*: an August audit removed library panics, introduced typed errors, layering rules and a performance evidence ledger.
  - *Boundaries*: C API for Julia unchanged; a PyO3 crate for Python started.
][
  Citation policy: each crate records provenance; users cite the original method papers plus upstream software papers (TCI: SciPost Phys. 18, 104 (2025); ITensor: SciPost Phys. Codebases 4 (2022)).
]

// =====================================================================
#slide("TreeTN: ground states, dynamics, storage", tag: "tensor4all-rs", color: c-t4rs)[
  - Two-site DMRG, TDVP and GSE-TDVP on general trees; data structures and sweep plans follow ITensorNetworks.jl.
  - *Root-edge-first sweep ordering*: fixed a ten-orders-of-magnitude accuracy loss on branching trees.
  - Variational fitting of sums of TreeTNs; dense tensor → TT-SVD constructor.
  - HDF5 schema for TreeTNs and an explicit CUDA contraction path.
][
  DMRG: White, PRL 69, 2863 (1992). TDVP: Haegeman et al., PRL 107, 070601 (2011); PRB 94, 165116 (2016). GSE: Yang and White, PRB 102, 094315 (2020). ITensorNetworks.jl (Apache-2.0, derived sweep plans).
]

// =====================================================================
#slide("Successive randomized compression (SRC)", tag: "tensor4all-rs", color: c-t4rs)[
  - Randomized algorithm for the compressed MPO × MPS product, by Camaño, Epperly and Tropp; extended here to general trees.
  - Fixed-rank and adaptive variants, Rust and C API. Adaptive SRC rests on tenferro's incremental Householder QR and rank-revealing QR.
  - CUDA-resident TreeTNs stay on device through the whole contraction.
  - Verified line by line against the authors' reference code; the provenance audit is in the repository docs.
][
  C. Camaño, E. N. Epperly, J. A. Tropp, "Successive randomized compression: a randomized algorithm for the compressed MPO-MPS product", arXiv:2504.06475 (2025). Reference code: github.com/chriscamano/RandomMPOMPS.
]

// =====================================================================
#slide("TreeACI: alternating cross interpolation on trees", tag: "tensor4all-rs", color: c-t4rs)[
  - ACI approximates elementwise functions of tensor trains by cross interpolation (Ritter et al.). Our chain crate is a port of AlternatingCrossInterpolation.jl.
  - *New*: ACI runs on a TreeTN directly, same options as the chain version, n-way products in one run.
  - Sweep is a deterministic minimum-retracing walk of length $2|E| - "diameter"$, reducing to the usual two sweeps on a chain.
  - Chain ACI gained a global-pivot guard and cache reuse across sweeps.
][
  M. K. Ritter et al., "Alternating cross interpolation", arXiv:2604.00037 (2026). Original Julia library: github.com/tensor4all/AlternatingCrossInterpolation.jl (Marc K. Ritter and contributors).
]

// =====================================================================
#slide("Partitioned tensor networks", tag: "tensor4all-rs", color: c-t4rs)[
  - *PartitionedTreeTN*: adaptive patching (Grosso et al.) generalized from tensor trains to arbitrary tree topologies.
  - Patch error budgets proportional to patch volume, $epsilon_p^2 prop "vol"_p$.
  - *Reconstruction*: repartition a patched target under a global L2 allowance; operators on index subsets such as the quantics Fourier transform.
  - *Parallel adaptive interpolation*: patch waves scheduled by hataori-rs, deterministic sampling across serial, Rayon and MPI runs.
][
  G. Grosso, M. K. Ritter, S. Rohshap, S. Badr, A. Kauch, M. Wallerberger, J. von Delft, H. Shinaoka, "Adaptive patching for tensor train computations", arXiv:2602.22372 (2026).
]

// =====================================================================
#slide("hataori-rs: tasks over threads and ranks", tag: "hataori-rs", color: c-green)[
  - A Rust engine for a dynamically scheduled `pmap`, inspired by Distributed.jl. Task granularity: seconds to minutes.
  - One API, four modes chosen by Cargo features: serial, Rayon, MPI, hybrid (ranks × threads).
  - MPI through rsmpi or a runtime-loaded MPI ABI, so it shares the runtime of MPI.jl and mpi4py.
  - Used by tensor4all-rs adaptive interpolation and a tenferro adapter. A long-lived distributed runtime is under design.
][
  Name: 機織り (weaving on a loom). github.com/shinaoka/hataori-rs
]

// =====================================================================
#slide("Bindings: Tensor4all.jl and Python", tag: "bindings", color: c-t4jl)[
  #grid(columns: (1fr, 1fr), gutter: 28pt,
    [
      *Tensor4all.jl* (best effort)
      - For light use and teaching; heavy users call tensor4all-rs directly
      - Windows CI, RustToolChain update, QR tolerance through the FFI
      - Usage audit skill: 12 downstream failure modes; batched readout beat pointwise readout by three orders of magnitude
    ],
    [
      *tensor4all-py* (PyO3 prototype)
      - Direct PyO3 boundary, not through the C API
      - Index, Tensor, TreeTN, batched TreeTCI and quantics cross interpolation from NumPy callbacks
      - Not yet: workspace membership, CI, wheels
    ])
]

// =====================================================================
#slide("How we develop, and what comes next")[
  #grid(columns: (1fr, 1fr), gutter: 28pt,
    [
      *Process*
      - Issues carry acceptance criteria; agents implement, read-only reviewer models gate design and diff
      - *Profile first*: no optimization without a paired measurement
      - Provenance policy and CITATION.cff per repository
    ],
    [
      *Next*
      - tenferro-rs JOSS paper, mid-October, with Jin-Guo Liu's Rust quantum circuit simulator
      - SRC and TreeACI benchmarks against zip-up and fitting on quantics workloads
      - Python: CI, wheels; Tensor4all.jl: catch up to TreeTN
    ])
  #v(1.2em)
  #align(center)[#text(size: 21pt, weight: "bold")[Discussion: which of these do you need first?]]
]

// =====================================================================
#slide("References")[
  #set text(size: 13.5pt)
  #set par(leading: 0.5em)
  #set enum(spacing: 0.55em)
  + Y. Núñez Fernández, M. K. Ritter, et al., Learning tensor networks with tensor cross interpolation: new algorithms and libraries, SciPost Phys. 18, 104 (2025).
  + Y. Núñez Fernández et al., Learning Feynman diagrams with tensor trains, Phys. Rev. X 12, 041018 (2022).
  + M. K. Ritter, Y. Núñez Fernández, M. Wallerberger, J. von Delft, H. Shinaoka, X. Waintal, Quantics tensor cross interpolation, Phys. Rev. Lett. 132, 056501 (2024).
  + M. K. Ritter et al., Alternating cross interpolation, arXiv:2604.00037 (2026); AlternatingCrossInterpolation.jl.
  + C. Camaño, E. N. Epperly, J. A. Tropp, Successive randomized compression, arXiv:2504.06475 (2025); RandomMPOMPS.
  + G. Grosso, M. K. Ritter, et al., Adaptive patching for tensor train computations, arXiv:2602.22372 (2026).
  + M. Fishman, S. R. White, E. M. Stoudenmire, The ITensor software library, SciPost Phys. Codebases 4 (2022); ITensorNetworks.jl.
  + S. R. White, Phys. Rev. Lett. 69, 2863 (1992); J. Haegeman et al., Phys. Rev. B 94, 165116 (2016); M. Yang and S. R. White, Phys. Rev. B 102, 094315 (2020).
  + J. Chen and M. Lindsey, Direct interpolative construction of the DFT as an MPO, arXiv:2404.03182.
  + Software: github.com/tensor4all/tenferro-rs, tensor4all-rs, tenferro-benchmark, Tensor4all.jl; github.com/shinaoka/hataori-rs.
]
