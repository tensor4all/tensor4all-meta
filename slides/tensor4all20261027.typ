// tensor4all meeting 2026-10-27: development update
// Build: typst compile tensor4all20261027.typ

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

#let blue = rgb("#111827")
#let c-tenferro = rgb("#2f6fb3")
#let c-t4rs = rgb("#2f6fb3")
#let c-t4jl = rgb("#7b47a8")
#let c-green = rgb("#6b7280")
#let accent = rgb("#c81e1e")
#let hi(t) = text(fill: accent, weight: "bold")[#t]

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

#let hl-blue = rgb("#2f6fb3")
#let stack-rows = (
  ("Frontends", "Julia, C, Python", (("Tensor4all.jl", "Julia frontend, best effort"), ("tensor4all-capi", "C API over tensor4all-rs"), ("tensor4all-py", "PyO3 prototype"))),
  ("Algorithms", "TT, TCI, tree networks", (("tensor4all-rs", "TT and TCI stack, network stack"), ("quanticsgrids-rs", "quantics grids, coordinates"))),
  ("Tensor runtime", "tensors, AD, backends", (("tenferro-rs", "tensor types, autodiff, CPU/GPU"), ("computegraph-rs", "graph IR for traced AD"))),
  ("Kernel providers", "contraction, linalg", (("cpueinsum-rs", "N-ary einsum, given order"), ("tcontract-rs", "binary contraction, C ABI"), ("tlinalg-rs", "batched SVD, QR, eigh, LU"))),
  ("Foundation", "views, dense kernels", (("strided-rs", "shared strided view types"), ("t4a-faer", "our fork of faer"), ("BLAS/LAPACK", "vendor library, optional"))),
)
// full = the stack slide; otherwise a small copy for the corner of a slide, with `hl` crates marked
#let stackmap(hl: (), full: false) = {
  let rh = if full { 1.45cm } else { 0.42cm }
  let gap = if full { 5pt } else { 1.5pt }
  let lw = if full { 4.4cm } else { 2.2cm }
  let fs = if full { 13pt } else { 6pt }
  let side-w = if full { 3.2cm } else { 1.4cm }
  let side-hit = hl.contains("hataori-rs")
  let box-of(name, desc) = {
    let hit = hl.contains(name)
    block(width: 100%, height: rh - (if full { 10pt } else { 2pt }), radius: if full { 5pt } else { 1.5pt }, inset: (x: if full { 7pt } else { 1.5pt }),
      fill: if hit { hl-blue } else { white }, stroke: 0.5pt + (if hit { hl-blue } else { gray.lighten(30%) }),
      align(left + horizon, if full [#text(size: fs, weight: "bold")[#name] \ #text(size: 9.5pt, fill: gray.darken(30%))[#desc]] else { text(size: fs, fill: if hit { white } else { gray.darken(25%) }, weight: if hit { "bold" } else { "regular" })[#name] }))
  }
  let rows = stack-rows.map(((label, sub, boxes)) => block(width: 100%, height: rh, radius: if full { 7pt } else { 2pt },
    fill: rgb("#eef2f7"), inset: (x: if full { 8pt } else { 2.5pt }),
    grid(columns: (lw, 1fr), align: horizon,
      if full [#text(size: 13pt, weight: "bold")[#label] \ #text(size: 9.5pt, fill: gray.darken(30%))[#sub]] else { text(size: fs, weight: "bold", fill: gray.darken(40%))[#label] },
      grid(columns: (1fr,) * boxes.len(), gutter: if full { 6pt } else { 1.5pt }, ..boxes.map(((n, d)) => box-of(n, d))))))
  grid(columns: (1fr, side-w), gutter: if full { 10pt } else { 3pt },
    stack(spacing: gap, ..rows),
    pad(top: rh + gap, bottom: rh + gap, block(width: 100%, height: 3 * rh + 2 * gap, radius: if full { 7pt } else { 2pt },
      fill: if side-hit { hl-blue } else { rgb("#eef2f7") }, inset: if full { 8pt } else { 2.5pt },
      if full [#text(size: 13pt, weight: "bold")[hataori-rs] \ #text(size: 9.5pt, fill: gray.darken(30%))[map and pmap over threads and MPI]] else { text(size: fs, weight: "bold", fill: if side-hit { white } else { gray.darken(40%) })[hataori-rs] })))
}
#let slide(title, tag: none, color: blue, crates: (), ..blocks) = {
  let body = blocks.pos().at(0)
  let refs = blocks.pos().at(1, default: none)
  pagebreak()
  block(width: 100%, height: 100%)[
    #grid(columns: (1fr, auto), align: (left, right + horizon),
      text(size: 26pt, weight: "bold")[#title],
      if crates.len() > 0 { place(top + right, dy: -1.35cm, box(width: 9.2cm, stackmap(hl: crates))) })
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



#let orange = rgb("#2f6fb3")
#let node(c: blue, r: 0.32cm) = circle(radius: r, fill: c.lighten(55%), stroke: 1pt + c)
#let hbar(label, v, maxv, color, txt) = grid(columns: (4.6cm, 1fr), gutter: 8pt, align: horizon,
  text(size: 15pt)[#label],
  stack(dir: ltr, spacing: 8pt, rect(width: 9cm * v / maxv, height: 0.55cm, fill: color, radius: 2pt), text(size: 14pt, weight: "bold")[#txt]))
#let card(title, sub, tagline, color, h: 6.6cm) = block(width: 100%, height: h, fill: color.lighten(90%), stroke: 1pt + color, radius: 6pt, inset: 10pt)[
  #text(size: 19pt, weight: "bold", fill: color)[#title]
  #v(2pt)
  #text(size: 14pt, fill: gray.darken(30%))[#sub]
  #v(6pt)
  #text(size: 16pt)[#tagline]
]
#let minitree(c, sc: 1.0) = box(width: 4.2cm * sc, height: 2.6cm * sc)[
  #let n(x, y) = place(top + left, dx: x * sc, dy: y * sc, node(c: c, r: 0.28cm * sc))
  #let e(x1, y1, x2, y2) = place(top + left, line(start: (x1 * sc, y1 * sc), end: (x2 * sc, y2 * sc), stroke: 1.4pt + c))
  #e(0.28cm, 1.08cm, 1.68cm, 1.38cm) #e(1.68cm, 1.38cm, 3.28cm, 0.48cm) #e(1.68cm, 1.38cm, 3.28cm, 2.28cm)
  #n(0cm, 0.8cm) #n(1.4cm, 1.1cm) #n(3cm, 0.2cm) #n(3cm, 2cm)
]
#let patches(x, y, w, depth) = {
  // split cells near the lower-left corner, where the function is hard
  let hard = x + y < 0.35 * 7cm and depth < 4
  if hard {
    let h = w / 2
    patches(x, y, h, depth + 1); patches(x + h, y, h, depth + 1)
    patches(x, y + h, h, depth + 1); patches(x + h, y + h, h, depth + 1)
  } else {
    place(top + left, dx: x, dy: 7cm - y - w, rect(width: w, height: w, fill: c-t4rs.lighten(70% + 7% * (3 - calc.min(depth, 3))), stroke: 0.6pt + c-t4rs))
  }
}
#let band(label, items, color) = block(width: 100%, fill: color.lighten(88%), stroke: 0.8pt + color, radius: 5pt, inset: 7pt)[
  #grid(columns: (3.8cm, 1fr), align: horizon, text(size: 14pt, weight: "bold", fill: color)[#label], text(size: 15pt)[#items])]

#let flow(..items) = grid(columns: items.pos().len() * 2 - 1, gutter: 8pt, align: horizon,
  ..items.pos().enumerate().map(((i, it)) => if i == 0 { (it,) } else { (text(size: 22pt, fill: gray)[→], it) }).flatten())
#let pill(t, c, w: auto) = block(width: w, fill: c.lighten(88%), stroke: 0.9pt + c, radius: 5pt, inset: 8pt, text(size: 15pt)[#t])

#title-slide(
  "tensor4all development update",
  subtitle: "Contributors, tree methods, tenferro, and the kernel libraries",
  author: "tensor4all developers",
  date: "tensor4all meeting, 2026-10-27",
)

// ---------------------------------------------------------------------
#slide("Who contributed since September 22")[
  #set text(size: 16pt)
  #table(columns: (1.4fr, 3.2fr), stroke: 0.4pt, inset: 5pt,
    [*Name*], [*Where*],
    [Samuel Badr], [tensor4all-rs],
    [Simon Becker], [tensor4all-rs: tree network storage (HDF5)],
    [Lingrui Cheng], [tensor4all-rs: tree patching, TreeRSI, tree reliability],
    [Lukas Devos], [tcontract-rs: the original contraction kernels],
    [Ken Inayoshi], [tensor4all-rs: performance, quantics transform precision],
    [Martin Mikkelsen], [tlinalg-rs, t4a-faer, tensor4all-rs, tenferro-rs],
    [Marc K. Ritter], [AlternatingCrossInterpolation.jl],
    [Rihito Sakurai], [tensor4all-rs: TCI pivot search fixes],
    [Hiroshi Shinaoka], [most repositories],
    [Satoshi Terasaki], [tenferro-rs, tenferro-benchmark, CI bot],
    [Ryo Watanabe], [t4a-faer],
  )
][
  Commit authors and co-authors on the default branches of tensor4all repositories, 2026-09-22 to 2026-10-27, alphabetical. Also upstream faer authors (via sync) and AI coding agents under human review.
]

// ---------------------------------------------------------------------
#slide("Activity since September 22")[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
    align(center)[#text(size: 44pt, weight: "bold", fill: blue)[11] \ #text(size: 15pt)[people committed]],
    align(center)[#text(size: 44pt, weight: "bold", fill: blue)[33] \ #text(size: 15pt)[repositories touched]],
    align(center)[#text(size: 44pt, weight: "bold", fill: blue)[15] \ #text(size: 15pt)[new public repositories]])
  #v(1em)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
    pill([*tensor4all-rs*: tree methods], c-t4rs),
    pill([*tenferro-rs*: execution engine, CUDA backend], c-tenferro),
    pill([*cpueinsum, tcontract, tlinalg*: split out of tenferro], orange))
][
  New repositories include tcontract-rs, cpueinsum-rs, tlinalg-rs, the faer fork, t4a-cuda-kernels, t4a-gpu-backend, threadteam-rs and their benchmark repositories.
]

// ---------------------------------------------------------------------
#slide("The stack")[
  #v(-0.4em)
  #stackmap(full: true)
][
  Interactive version: tensor4all.org/tensor4all-meta/stack/
]

// =====================================================================
#slide("Adaptive patching on trees", crates: ("tensor4all-rs",))[
  Published for tensor trains; #hi[Lingrui Cheng extended it to trees.]
  #v(0.4em)
  #grid(columns: (6.4cm, 1fr), gutter: 28pt, align: horizon,
    box(width: 6cm, height: 6cm, scale(85.7%, reflow: true, box(width: 7cm, height: 7cm, patches(0cm, 0cm, 7cm, 0)))),
    [
      - Split the domain *only where the function is hard*
      - Every patch is a small tree with the same shape
      - Works for chains and branched trees alike
      #v(0.6em)
      #note[On one test the patched run was slower and less accurate than a single network. When patching pays off is still open.]
    ])
][
  G. Grosso, M. K. Ritter, S. Rohshap, S. Badr, A. Kauch, M. Wallerberger, J. von Delft, H. Shinaoka, Adaptive patching for tensor train computations, arXiv:2602.22372. Trees: tensor4all-rs PR #900.
]

// ---------------------------------------------------------------------
#slide("Parallel patching with hataori", crates: ("tensor4all-rs", "hataori-rs"))[
  #align(center)[
    #flow(pill([one big problem], c-t4rs), pill([many independent patches], c-t4rs), pill([hataori: threads or MPI ranks], c-green), pill([one result], c-t4rs))
  ]
  #v(1em)
  - Patches are independent, so they can run on different cores or nodes
  - *hataori-rs* hands out the work; each worker uses its own tenferro engine
  - Chains and trees share one implementation; the chain-only crate was removed
  #v(0.5em)
  #note[Tested for correctness on 1 and 2 MPI ranks. Scaling not measured yet.]
][
  tensor4all-rs PRs #1005, #1009; hataori-rs PRs #25, #27.
]

// ---------------------------------------------------------------------
#slide("TreeRSI", crates: ("tensor4all-rs",))[
  Published for tensor trains; #hi[Lingrui Cheng extended it to trees.]
  #v(0.4em)
  #align(center)[
    #grid(columns: 5, gutter: 14pt, align: horizon,
      minitree(c-t4rs), text(size: 30pt)[×], minitree(c-tenferro), text(size: 30pt, fill: gray)[→], minitree(c-green))
    #grid(columns: (4.2cm, 1cm, 4.2cm, 1.3cm, 4.2cm), gutter: 14pt,
      text(size: 14pt)[tree A], [], text(size: 14pt)[tree B], [], text(size: 14pt)[A·B, already small])
  ]
  #v(1em)
  - Element-by-element product of two trees, compressed in one pass
  - Checked against the original authors' code, with the same random numbers
][
  Z. Meng, Y. Khoo, J. Li, E. M. Stoudenmire, Recursive sketched interpolation: efficient Hadamard products of tensor trains, arXiv:2602.17974.
]

// ---------------------------------------------------------------------
#slide("Other fixes in tensor4all-rs", crates: ("tensor4all-rs",))[
  #grid(columns: (1fr, 1fr), gutter: 12pt,
    card("Convergence reports", "Lingrui Cheng", [Says "rank limit reached" instead of a false "converged".], c-t4rs, h: 4.7cm),
    card("TCI pivots", "Rihito Sakurai", [Pivot search no longer stops at zero residuals or forgets good moves.], c-t4rs, h: 4.7cm),
    card("Speed", "Ken Inayoshi, Martin Mikkelsen", [Fewer copies in core loops; column-major grids in TCI.], c-t4rs, h: 4.7cm),
    card("Storage", "Simon Becker", [Tree networks saved to HDF5 keep their canonical form.], c-t4rs, h: 4.7cm))
][
  Full details of the tree work: slides/tensor4all20261027-lingrui-cheng.pdf
]

// ---------------------------------------------------------------------
#slide("Remaining work", crates: ("tensor4all-rs",))[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
    card("Patched products", hi[biggest gap], [Multiply two patched operators (MPO-MPO) end to end.], c-t4rs, h: 7.2cm),
    card("Error checks", "cheaper and safer", [Catch local errors that sampling misses, at lower cost; fix remaining TreeACI convergence issues.], c-t4rs, h: 7.2cm),
    card("Scaling", "not measured yet", [Run the MPI patching at scale and measure.], c-t4rs, h: 7.2cm))
][
  Source: Lingrui Cheng's slides, "Remaining work".
]

// =====================================================================
#slide("tenferro execution engine", crates: ("tenferro-rs",))[
  #align(center)[
    #flow(pill([your program], gray), pill([creates an *engine*: threads, memory, GPU stream], c-tenferro), pill([prepare once], c-tenferro), pill([run many times], c-tenferro))
  ]
  #v(1em)
  - The engine is an ordinary value: you create it and pass it to each call
  - Two engines can run side by side, for example one per MPI rank or per patch
  - Small operations stay on the calling thread
  #v(0.5em)
  #note[This is a breaking change; the old session API was removed.]
][
  tenferro-rs #2206, PR #2209.
]

// ---------------------------------------------------------------------
#slide("New CUDA backend", crates: ("tenferro-rs",))[
  #grid(columns: (1fr, 1fr), gutter: 24pt, align: horizon,
    [
      #pill([#align(center)[tenferro GPU engine]], c-tenferro, w: 100%)
      #v(4pt)
      #align(center)[#text(size: 20pt, fill: gray)[↓]]
      #grid(columns: (1fr, 1fr), gutter: 6pt,
        pill([*t4a-cuda-kernels* \ our own CUDA kernels, C ABI], orange, w: 100%),
        pill([*cuBLAS, cuTENSOR* \ vendor libraries], gray, w: 100%))
      #v(4pt)
      #pill([#align(center)[*t4a-gpu-backend*: streams, memory, launches]], orange, w: 100%)
    ],
    [
      - Replaces the earlier CubeCL-based CUDA path
      - The kernels use no tenferro types, so other CUDA codes can call them
      - Matrix products via cuBLAS, contractions via cuTENSOR
      - Migration is in progress, piece by piece; tested on an A100
    ])
][
  tenferro-rs #2208, PRs #2250 to #2263; t4a-cuda-kernels; t4a-gpu-backend.
]

// ---------------------------------------------------------------------
#slide("tenferro-benchmark", crates: ("tenferro-rs",))[
  #set text(size: 16pt)
  #align(center)[
    #flow(pill([tenferro main], gray), pill([CPU: every night, Linux and macOS \ GPU: A100, every weekend], c-tenferro), pill([PyTorch, JAX, Julia, HPTT, cuTENSOR], c-tenferro), pill([dashboard], gray))
  ]
  #v(0.6em)
  - Runs start automatically from GitHub Actions; the GPU runs use rented A100s
  - Raw data and settings for every run are kept in the repository
  - Example, A100, a tensor network contraction (f32): tenferro 49 ms, PyTorch 111 ms
  - Dashboard: tensor4all.org/tenferro-benchmark
][
  github.com/tensor4all/tenferro-benchmark; example from result/nvidia-gpu/gpu/tensornetwork.md, 2026-10-24, tenferro 06ac247. Most changes since September 22 by Satoshi Terasaki.
]

// =====================================================================
#slide("Why split out the kernels", crates: ("cpueinsum-rs", "tcontract-rs", "tlinalg-rs", "t4a-faer"))[
  #grid(columns: (1fr, auto, 1fr), gutter: 12pt, align: horizon,
    stack(spacing: 4pt, ..("Rust: tenferro", "Julia", "Python", "C, Fortran codes").map(t =>
      block(width: 100%, fill: c-tenferro.lighten(88%), stroke: 0.8pt + c-tenferro, radius: 4pt, inset: 5pt, text(size: 14pt)[#t]))),
    text(size: 26pt, fill: gray)[→ \ #text(size: 12pt)[C API]],
    block(width: 100%, fill: orange.lighten(88%), stroke: 1.2pt + orange, radius: 6pt, inset: 12pt)[
      #text(size: 18pt, weight: "bold", fill: orange)[Engine]
      #v(4pt)
      #text(size: 16pt)[tcontract · cpueinsum · tlinalg · t4a-faer]
    ])
  #v(0.2em)
  Started with *Lukas Devos* (CCQ, developer of TensorKit.jl) and his Rust prototype.
  #v(0.1em)
  #set list(spacing: 0.6em)
  - *Julia:* call precompiled kernels, less precompile time (future)
  - Fast kernels are too much work for one group to maintain
  - #hi[We want to maintain them together with other groups]
  - Planned: CPU kernel optimization with Guo Chen's group at HKUST(GZ)
]

// ---------------------------------------------------------------------
#slide("Four libraries", crates: ("cpueinsum-rs", "tcontract-rs", "tlinalg-rs", "t4a-faer"))[
  #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 10pt,
    card("tcontract-rs", "multiply two tensors", [From *Lukas Devos's prototype*. C API follows *TAPP*.], orange),
    card("cpueinsum-rs", "multiply many tensors", [*Cut out of tenferro.* Lower per-step overhead.], orange),
    card("tlinalg-rs", "SVD, QR, LU, eigen", [Thin layer over t4a-faer. *Batch* API like PyTorch.], orange),
    card("t4a-faer", "fork of faer", [New parallel model, bug fixes, *AI-driven* optimization.], orange))
  #v(0.8em)
  #text(size: 15pt, fill: gray.darken(30%))[All four now on crates.io and used by tenferro.]
]

// ---------------------------------------------------------------------
#slide("tcontract-rs", crates: ("tcontract-rs",))[
  #align(center)[
    #flow(pill([A, B with any strides], gray), pill([packed straight into small tiles], orange), pill([fast inner kernel (AVX, NEON)], orange), pill([D = A·B + C], gray))
  ]
  #v(1em)
  - No transpose or copy of the inputs first
  - f32, f64 and complex; batch indices, traces, conjugation
  - C API follows *TAPP*, a community standard for tensor products
  - One shared library, `libtcontract_capi`, for any language with a C FFI
][
  TAPP: arXiv:2601.07827. tcontract-rs README.
]

// ---------------------------------------------------------------------
#slide("cpueinsum-rs", crates: ("cpueinsum-rs",))[
  #align(center)[
    #flow(pill([a, b, c, d], gray), pill([step 1: a·b], orange), pill([step 2: (ab)·c], orange), pill([step 3: ...·d], orange), pill([result], gray))
  ]
  #v(1em)
  - *You* choose the order; cpueinsum plans every step once
  - One reusable scratch buffer for all the intermediates
  - Not included: order search, `"ij,jk->ik"` strings, GPUs
  - Usable without tenferro
]

// ---------------------------------------------------------------------
#slide("tlinalg-rs: batches", crates: ("tlinalg-rs",))[
  Like PyTorch, tlinalg takes a *batch*: many matrices of the same shape in one call.
  #v(0.6em)
  With 4 threads there are two ways to run it:
  + Give each thread its own share of the batch, a contiguous block of matrices.
  + Factorize the matrices one by one, each with all 4 threads.
  #v(0.6em)
  For small matrices the threads in the second way spend most of their time coordinating, so #hi[the first way is faster]. By default tlinalg picks the way itself; a caller can also fix it.
][
  tlinalg-rs 0.6. The default policy estimates each matrix's work and splits the batch across threads only when that repays the cost of starting them.
]

// ---------------------------------------------------------------------
#slide("t4a-faer", crates: ("t4a-faer",))[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
    card("Fork of faer", "dense linear algebra in Rust", [We use it for fast experiments, mostly run by AI agents.], orange),
    card("What we changed", "", [Parallel model, bug fixes, faster QR and LU paths.], orange),
    card("Experiment log", "PATCHSET.md", [Each change is recorded as kept or dropped, with the reason.], orange))
][
  faer by Sarah Quiñones. Our fork: github.com/tensor4all/faer, crates.io t4a-faer.
]

// ---------------------------------------------------------------------
#slide("faer-benchmark", crates: ("t4a-faer",))[
  #align(center)[
    #flow(pill([every commit of the fork], gray), pill([same machine, same protocol], orange), pill([fork vs upstream faer vs MKL, OpenBLAS], orange), pill([public pages], gray))
  ]
  #v(1em)
  - Nine families: GEMM, LU, Cholesky, QR, SVD, eigenvalues, ...
  - Each result is tied to a commit and a machine; results from older commits are marked
  - An unchanged control row checks the machine; if it moves, the result is not accepted
  - Raw data, scripts and every run's settings are in the repository
][
  github.com/tensor4all/faer-benchmark. Sibling repositories: tlinalg-benchmark, tcontract-benchmark, cpueinsum-benchmark, tenferro-benchmark.
]

// ---------------------------------------------------------------------
#slide("QR, 1 vs 8 threads", crates: ("t4a-faer",))[
  #align(center)[#image("figures/faer-qr-factor-1t-8t.svg", height: 7.6cm)]
  #v(0.2em)
  #set text(size: 16pt)
  #grid(columns: (1fr, 1fr), gutter: 20pt,
    [*1 thread:* our fork beats upstream faer and MKL.],
    [*8 threads:* MKL still leads. #hi[Parallel scaling is the next target.]])
][
  faer-benchmark, AMD Zen 5, f64, nominal GFLOP/s ($4n^3\/3$); fork 50c2b98 (n = 1024: a5e6b7d) vs upstream faer 0.24.4. QR with an explicit Q shows the same trend. Provisional: development host.
]

// ---------------------------------------------------------------------
#slide("Regular benchmarking")[
  #v(0.3em)
  #align(center)[#text(size: 20pt)[AI agents write most of this code. \ #hi[Public, repeatable benchmarks let others check the results.]]]
  #v(0.8em)
  - tenferro-benchmark already runs on a schedule: CPU every night, A100 every weekend
  - faer-benchmark runs on a development host for now; a dedicated machine will rerun it regularly
  - We will show the trends at these meetings
]

// ---------------------------------------------------------------------
#slide("Proposal: grow the Matrix space")[
  The tensor4all space on Matrix has almost 50 members. Please invite colleagues, students and users.
  #v(0.8em)
  #block(fill: rgb("#eef2f7"), inset: 12pt, radius: 6pt, width: 100%)[
    *Koushi*, a desktop Matrix client
    #v(2pt)
    #set text(size: 16pt)
    - Several Matrix accounts in one window, each with its own tab
    - Typing and search that work for Japanese, Chinese and Korean; interface in five languages
    - Local full-text search, including encrypted rooms
    - Open source (Apache-2.0), macOS build available
  ]
][
  github.com/shinaoka/koushi-matrix · shinaoka.github.io/koushi-matrix
]

// ---------------------------------------------------------------------
#slide("Proposal: alternate the meeting time")[
  Now: Tuesday 14:00 Vienna time, summer and winter. That is 21:00 or 22:00 in Tokyo.
  #v(0.4em)
  Proposal: alternate Tuesday 9:00 and 14:00 Vienna time, also fixed in Vienna time all year.
  #v(0.6em)
  #align(center)[
    #table(columns: 5, stroke: 0.5pt + gray, inset: 8pt, align: center,
      [*Vienna, Paris, Berlin*], [*Beijing*], [*Tokyo, Seoul*], [*New York*], [],
      [9:00], [16:00], [17:00], [3:00], [new],
      [14:00], [21:00], [22:00], [8:00], [current],
    )
  ]
][
  Winter time, from October 25 to March 28. In summer, Beijing, Tokyo and Seoul are one hour earlier.
]

// ---------------------------------------------------------------------
#slide("Discussion")[
  - *tensor4all-rs:* which tree workloads should we test with real users?
  - *tenferro:* who needs the CUDA backend first?
  - *Kernels:* who else wants to share them with us?
  - *t4a-faer:* send our fixes back to upstream faer?
  - *Matrix:* whom should we invite?
  - *Meeting time:* alternate 9:00 and 14:00?
]

// ---------------------------------------------------------------------
#slide("References")[
  #set text(size: 14pt)
  - Tree methods in detail (Lingrui Cheng): `slides/tensor4all20261027-lingrui-cheng.pdf` and its notes
  - Adaptive patching: G. Grosso et al., arXiv:2602.22372
  - RSI: Z. Meng, Y. Khoo, J. Li, E. M. Stoudenmire, arXiv:2602.17974
  - TAPP: arXiv:2601.07827
  - Stack map: tensor4all.org/tensor4all-meta/stack/
  - Code: github.com/tensor4all (tensor4all-rs, tenferro-rs, tcontract-rs, cpueinsum-rs, tlinalg-rs, faer)
]
