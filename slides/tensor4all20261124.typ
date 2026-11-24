// tensor4all meeting 2026-11-24: progress since 2026-10-27
// Build: typst compile --root .. tensor4all20261124.typ

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
#let c-main = rgb("#2f6fb3")
#let copper = rgb("#b7510b")
#let accent = rgb("#c81e1e")
#let hi(t) = text(fill: accent, weight: "bold")[#t]
#let note(body) = text(size: 13pt, fill: gray.darken(30%))[#body]

#let title-slide(title, subtitle: none, author: none, date: none) = {
  set align(center + horizon)
  block[
    #image("../branding/logo/svg/tensor4all-wordmark.svg", height: 1.6cm)
    #v(1em)
    #text(size: 40pt, weight: "bold")[#title]
    #if subtitle != none { v(0.5em); text(size: 24pt, fill: gray.darken(20%))[#subtitle] }
    #if author != none { v(1.5em); text(size: 22pt)[#author] }
    #if date != none { v(0.5em); text(size: 20pt, fill: gray)[#date] }
  ]
}

#let slide(title, ..blocks) = {
  let body = blocks.pos().at(0)
  let refs = blocks.pos().at(1, default: none)
  pagebreak()
  block(width: 100%, height: 100%)[
    #text(size: 26pt, weight: "bold")[#title]
    #v(0.6em)
    #line(length: 100%, stroke: 0.5pt + blue)
    #v(0.8em)
    #set text(size: 18pt)
    #set par(leading: 0.6em)
    #set list(spacing: 0.8em)
    #body
    #if refs != none {
      place(bottom + left, dy: 1.15cm, block(width: 88%, text(size: 11.5pt, fill: gray.darken(30%))[#refs]))
    }
  ]
}

#let stat(n, label) = align(center)[#text(size: 28pt, weight: "bold", fill: blue)[#n] \ #text(size: 12.5pt)[#label]]
#let pill(t, c, w: auto) = block(width: w, fill: c.lighten(88%), stroke: 0.9pt + c, radius: 5pt, inset: 8pt, text(size: 15pt)[#t])
#let card(title, sub, body, color, h: 5.6cm) = block(width: 100%, height: h, fill: color.lighten(90%), stroke: 1pt + color, radius: 6pt, inset: 10pt)[
  #text(size: 18pt, weight: "bold", fill: color)[#title]
  #if sub != "" { v(1pt); text(size: 13pt, fill: gray.darken(30%))[#sub] }
  #v(4pt)
  #text(size: 13pt)[#set list(spacing: 0.5em); #body]
]
// two bars per repository: PRs (blue) and issues (copper)
#let maxv = 425
#let bars(name, p, i) = (
  text(size: 11pt)[#name],
  stack(spacing: 1pt,
    stack(dir: ltr, spacing: 4pt, rect(width: 4.0cm * p / maxv, height: 0.22cm, fill: c-main, radius: 1.5pt), text(size: 9.5pt)[#p]),
    stack(dir: ltr, spacing: 4pt, rect(width: 4.0cm * i / maxv, height: 0.22cm, fill: copper, radius: 1.5pt), text(size: 9.5pt)[#i])),
)
// PRs and issues opened per month in org:tensor4all (GitHub search, by creation date)
#let monthly = (
  ("2025-01", 6, 3), ("2025-02", 7, 4), ("2025-03", 13, 9), ("2025-04", 15, 6),
  ("2025-05", 7, 3), ("2025-06", 12, 8), ("2025-07", 2, 3), ("2025-08", 7, 1),
  ("2025-09", 11, 3), ("2025-10", 6, 3), ("2025-11", 103, 1), ("2025-12", 20, 1),
  ("2026-01", 227, 70), ("2026-02", 297, 215), ("2026-03", 300, 311), ("2026-04", 220, 143),
  ("2026-05", 146, 159), ("2026-06", 126, 252), ("2026-07", 267, 227), ("2026-08", 220, 207),
  ("2026-09", 246, 192), ("2026-10", 1216, 615), ("2026-11", 1425, 1011),
)
#let monthly-chart(w: 13.4cm, h: 3.6cm) = {
  let ymax = 1500
  let n = monthly.len()
  let slot = w / n
  let bw = slot * 0.36
  let y(v) = h * v / ymax
  box(width: w + 1cm, height: h + 0.55cm, {
    for t in (0, 500, 1000, 1500) {
      place(top + left, dx: 0.9cm, dy: h - y(t), line(length: w, stroke: 0.3pt + gray.lighten(40%)))
      place(top + left, dx: 0cm, dy: h - y(t) - 0.2cm, box(width: 0.75cm, align(right, text(size: 9pt, fill: gray.darken(20%))[#t])))
    }
    for (k, (m, p, i)) in monthly.enumerate() {
      let x0 = 0.9cm + slot * k + slot * 0.14
      place(top + left, dx: x0, dy: h - y(p), rect(width: bw, height: y(p), fill: c-main))
      place(top + left, dx: x0 + bw, dy: h - y(i), rect(width: bw, height: y(i), fill: copper))
      let mm = m.slice(5)
      if mm == "01" or mm == "07" or k == n - 1 {
        place(top + left, dx: x0 - 0.3cm, dy: h + 0.12cm, text(size: 9pt, fill: gray.darken(30%))[#m])
      }
    }
    place(top + left, dx: 0.9cm + slot * (n - 1) - 1.0cm, dy: -0.05cm, text(size: 9pt, fill: accent)[to 11/24])
  })
}

#title-slide(
  "Progress since October 27",
  subtitle: "Activity, colmat, logos, a new name, the tenferro paper",
  author: "tensor4all developers",
  date: "tensor4all meeting, 2026-11-24",
)

// ---------------------------------------------------------------------
#slide("Activity")[
  #grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 10pt,
    stat("1739", [PRs (1674 merged)]),
    stat("1194", [issues (988 closed)]),
    stat("19", [people (bots excluded)]),
    stat("56", [active repositories]))
  #v(0.2em)
  #grid(columns: (1.45fr, 1fr), gutter: 14pt,
    [
      #text(size: 14pt, weight: "bold")[Per month] #h(0.6em)
      #box(rect(width: 0.45cm, height: 0.26cm, fill: c-main)) #text(size: 12pt)[PRs] #h(0.4em)
      #box(rect(width: 0.45cm, height: 0.26cm, fill: copper)) #text(size: 12pt)[issues]
      #v(0.1em)
      #monthly-chart()
    ],
    [
      #text(size: 14pt, weight: "bold")[Since October 27, by repository]
      #v(0.2em)
      #grid(columns: (4.4cm, 1fr), row-gutter: 2pt, align: horizon,
        ..bars("colmat (4 repos)", 340, 425),
        ..bars("tenferro-rs", 259, 182),
        ..bars("tensor4all-rs", 242, 187),
        ..bars("tenferro-benchmark", 191, 54),
        ..bars("strided-rs", 124, 59),
        ..bars("pencil-rs", 18, 102),
      )
    ])
][
  GitHub search, org:tensor4all (incl. private repos), by creation date; top numbers 2026-10-27 to 11-24. New repos: colmat-rs, colmat-next-rs, t4a-gemm, tenferro-examples, tenferro-sci-rs. colmat = t4a-faer, colmat-rs, -benchmark, -next-rs; colmat-rs issues include copies of old t4a-faer issues.
]

// ---------------------------------------------------------------------
#slide("colmat: the faer fork becomes its own library")[
  #grid(columns: (1fr, 1.25fr), gutter: 18pt,
    [
      #image("../branding/colmat/svg/colmat-wordmark.svg", height: 1.1cm)
      #v(0.3em)
      #set text(size: 13.5pt)
      #set list(spacing: 0.5em)
      - `t4a-faer` 0.33 → *`colmat` 0.34* (2026-11-17), now in `tensor4all/colmat-rs`; the old repository is archived
      - Goal: #hi[vendor-level performance with precise thread control], not "beat MKL"
      - Parallel work runs on a `threadteam` team the caller owns; no global thread pool
      - Credit stays with faer (Sarah Quiñones); users keep citing the faer paper
    ],
    [
      #text(size: 16pt, weight: "bold")[Optimization and validation since October]
      #v(0.2em)
      #set text(size: 13pt)
      #set list(spacing: 0.45em)
      - *Speed:* eigh, LU, QR, Cholesky, SVD measured against MKL, the gaps found and closed one by one (D&C merge across the team, small QR, row-major inputs, tile splits)
      - *Correctness:* checked against LAPACK's and OpenBLAS's own tests; fixes near underflow and overflow; Miri in CI
      - *New:* polar decomposition (QDWH), QL/RQ, generalized eigenvectors
      - *Upstream reports:* defects that also affect published faer kept as reproducers
      - *Next generation:* `colmat-next-rs`, a trial of the new design (scalar contract, double-double)
    ])
][
  github.com/tensor4all/colmat-rs · benchmarks: tensor4all.org/colmat-benchmark · design: tensor4all-meta docs/design/next-generation-dense-linear-algebra.md
]

// ---------------------------------------------------------------------
#slide("Logos adopted")[
  #grid(columns: (1fr, 1fr), gutter: 24pt, align: horizon,
    stack(spacing: 22pt,
      image("../branding/logo/svg/tensor4all-wordmark.svg", height: 1.5cm),
      image("../branding/tenferro/svg/tenferro-wordmark.svg", height: 1.2cm),
      image("../branding/colmat/svg/colmat-wordmark.svg", height: 1.2cm),
      image("../branding/threadteam/svg/threadteam-wordmark.svg", height: 1.2cm)),
    [
      #set text(size: 16pt)
      - *T∀* drawn as tensor diagrams: T, a three-leg tensor, for _tensor_; ∀, "for all", for _4all_
      - From Xavier's T4∀ idea and Jan's earlier circuit mark; the final form is Jan's
      - One family: IBM Plex Sans, ink and copper, the tensor4all.org palette
      - Drawn to survive printing on T-shirts and embroidery
      - All generated by one script; SVG and PNG, light and dark, favicons
    ])
][
  tensor4all.org/tensor4all-meta/logos/ · tensor4all-meta branding/ (adopted 2026-11)
]

// ---------------------------------------------------------------------
#slide("A new name for tensor4all-rs?")[
  #set text(size: 17pt)
  *Why:* the tensor4all ecosystem now goes beyond tensor networks (tenferro, colmat, threadteam, tenferro-sci, ...). "tensor4all-rs" sounds like all of it.
  #v(0.3em)
  #grid(columns: (1fr, 1fr), gutter: 20pt,
    [
      #text(weight: "bold")[Candidates from the Matrix room]
      #v(0.2em)
      #set text(size: 13pt)
      #table(columns: (4.4cm, 1fr), stroke: 0.4pt + gray, inset: 4pt,
        table.cell(fill: c-main.lighten(85%))[*tenkai*], table.cell(fill: c-main.lighten(85%))[#hi[front-runner]: unique, easy to say],
        [tennet], [*ten*\sor *net*\works; palindrome; name clash],
        [tenkyu], [天球, "celestial sphere"],
        [tenq / tenqai], [avoided: "q" is everywhere],
        [YETI, tentela, tenami, ...], [others],
      )
      #v(0.5em)
      #block(width: 100%, fill: accent.lighten(90%), stroke: 0.9pt + accent, radius: 5pt, inset: 7pt)[#set text(size: 15pt); *To discuss today:* tenkai-rs? other ideas?]
    ],
    [
      #text(weight: "bold")[Where we stand]
      #v(0.2em)
      #set text(size: 14pt)
      - Not decided yet; #hi[requires the consent of the original authors] (TCI.jl, xfac)
      - Yuriel: OK (by e-mail) · Marc Ritter and others: to be asked
      - Check that the name is free on crates.io, PyPI and the Julia registry
      - Then: rename the repository and crates, as with `t4a-faer` → `colmat`
    ])
][
  Discussion in the tensor4all-rs/jl Matrix room, November 2026.
]

// ---------------------------------------------------------------------
#slide("tenferro paper and tenferro-sci-rs")[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    card("Software paper on tenferro", "planned submission: end of January 2027", [
      - Draft in `tensor4all/tenferro-paper`; journal not fixed
      - Story: a Rust tensor foundation for the agentic-AI era; AD, CPU/GPU, on par with PyTorch and JAX
      - #hi[Open authorship:] substantial contributors to tenferro or its stack (strided, tcontract, tlinalg, colmat, tenferro-sci, ...) can join
      - Apply by a PR stating your contribution; join `#tenferro-tensor4all:matrix.org`
    ], c-main, h: 9.6cm),
    card("tenferro-sci-rs", "SciPy for tenferro; repository created 2026-11-20", [
      - The second layer: krylov, optimize, special, integrate, interpolate, plus fft and einsum moved out of tenferro
      - Pure Rust cores usable without tenferro; tenferro crates add AD, GPU, batching
      - Validated against SciPy and mpmath
      - First crate `specials` (xsf port, f64 and double-double): gamma family done; 12 families open as issues
      - Not a full SciPy port: add modules only when needed
    ], copper, h: 9.6cm))
][
  github.com/tensor4all/tenferro-paper (README, AUTHORS-NOTES.md) · github.com/tensor4all/tenferro-sci-rs · design: tensor4all-meta docs/design/tenferro-sci.md
]
