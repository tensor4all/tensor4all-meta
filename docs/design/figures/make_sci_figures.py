#!/usr/bin/env python3
"""Generate the SVG figures for tenferro-sci.md.

Usage (from the repository root):

    python3 docs/design/figures/make_sci_figures.py

Writes one SVG per figure next to this script. Standard library only; reuses
the drawing helpers of make_figures.py. Edit and rerun; commit the script and
the SVGs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import Figure  # noqa: E402


def sci_layers():
    f = Figure(900, 660)

    f.box(30, 20, 840, 80, "Users and agents",
          ["applications, tenferro-examples, tensor4all-rs, ..."],
          style="neutral")

    f.box(30, 140, 840, 140, "tenferro-sci-rs  (one workspace, one version)",
          style="green", title_at="top")
    names = [("tenferro-krylov", ["eigsolve, linsolve,", "exponentiate"]),
             ("tenferro-optimize", ["L-BFGS, LM,", "root finding"]),
             ("tenferro-special", ["gamma, Bessel,", "erf, ..."]),
             ("tenferro-integrate", ["quadrature,", "ODE + adjoint"]),
             ("tenferro-interpolate", ["splines,", "grid interpolation"])]
    x = 45
    for name, lines in names:
        f.box(x, 182, 158, 82, name, lines, style="plain")
        x += 165

    f.box(30, 330, 380, 190, "tenferro", style="green", title_at="top")
    f.box(45, 372, 350, 60, "tensors, runtime", ["CPU / CUDA / Metal"],
          style="plain")
    f.box(45, 444, 350, 60, "autodiff", ["eager + traced"], style="plain")

    f.box(440, 330, 430, 220,
          "Independent cores: pure Rust, no tenferro", style="amber",
          title_at="top")
    cores = [("krylov", ["Krylov methods", "(Ryo Watanabe)"]),
             ("special functions", ["pure Rust port of", "SciPy's xsf"]),
             ("quadrature rules", ["nodes and weights"]),
             ("interpolation", ["splines, Chebyshev", "weights"])]
    for i, (name, lines) in enumerate(cores):
        cx = 455 + (i % 2) * 205
        cy = 372 + (i // 2) * 84
        f.box(cx, cy, 195, 72, name, lines, style="plain")

    f.box(440, 580, 430, 64, "Any Rust code",
          ["plain Vec, ndarray, mdarray, symtn, ..."], style="neutral",
          dashed=True)

    f.arrow([(450, 102), (450, 136)])
    f.arrow([(220, 282), (220, 326)], label="tensors, AD rules",
            label_at=(220, 304))
    f.arrow([(655, 282), (655, 326)], label="algorithms",
            label_at=(655, 304))
    f.arrow([(655, 578), (655, 554)])
    f.save("sci_layers.svg")


def sci_adapter():
    f = Figure(860, 430)

    f.box(30, 30, 360, 170, "krylov  (the core)",
          ["iteration, orthogonalization,", "restarts, convergence,",
           "small projected problems (colmat)"], style="amber")
    f.box(30, 250, 360, 150, "Everywhere",
          ["any type that implements", "Vector / Basis / LinearMap:",
           "Vec<f64>, symtn tensors, ..."], style="neutral")

    f.box(470, 30, 360, 370, "tenferro-krylov  (only on tenferro)",
          style="green", title_at="top")
    f.box(490, 75, 320, 70, "trait impls for tenferro tensors",
          ["CPU and GPU, one engine per call"], style="plain")
    f.box(490, 160, 320, 70, "AD rules",
          ["implicit differentiation of", "eigen- and linear solves"],
          style="plain")
    f.box(490, 245, 320, 64, "batched and traced use",
          ["many small problems, compile once"], style="plain")
    f.box(490, 324, 320, 60, "one dependency for the user",
          ["re-exports krylov"], style="plain")

    f.arrow([(210, 202), (210, 246)])
    f.arrow([(392, 115), (466, 115)], label="depends on", label_at=(429, 92))
    f.save("sci_adapter.svg")


def sci_release():
    f = Figure(860, 330)

    f.box(30, 30, 250, 120, "tenferro-rs",
          ["released crates", "and main"], style="green")
    f.box(320, 30, 250, 120, "tenferro-sci-rs",
          ["all crates bumped", "together"], style="green")
    f.box(640, 30, 200, 120, "crates.io",
          ["tenferro-krylov, ...", "same version"], style="neutral")

    f.box(320, 200, 250, 110, "CI",
          ["against released tenferro", "and, daily, against main"],
          style="amber")
    f.box(30, 200, 250, 110, "SciPy oracle data",
          ["reference values and", "derivatives from SciPy"],
          style="purple")

    f.arrow([(282, 90), (316, 90)])
    f.arrow([(572, 90), (636, 90)], label="publish", label_at=(604, 66))
    f.arrow([(445, 152), (445, 196)])
    f.arrow([(282, 255), (316, 255)])
    f.save("sci_release.svg")


def sci_order():
    f = Figure(900, 590)

    f.box(20, 20, 860, 200, None, style="amber")
    f.text(36, 44, "Lower layers (other repositories)", size=15, weight=600,
           color="#6B4A12", anchor="start")
    lower = [
        (35, "CPU stack on colmat", ["colmat 0.34, threadteam 0.6", "tenferro-rs#2597"]),
        (250, "elem-simd-math", ["digamma, i0e, i1e, xlogy", "elem-simd-math-rs#17"]),
        (465, "tenferro core ops", ["erfc, lgamma, digamma, ...", "tenferro-rs#2692"]),
        (680, "tenferro-fft", ["DCT / DST", "(not filed yet)"]),
    ]
    for x, name, lines in lower:
        f.box(x, 62, 190, 140, name, lines, style="plain")
    f.arrow([(442, 132), (461, 132)])

    f.box(20, 340, 860, 230, None, style="green")
    f.text(36, 364, "tenferro-sci-rs, in this order", size=15, weight=600,
           color="#1F5A3F", anchor="start")
    crates = [
        (32, "0. groundwork", ["workspace,", "SciPy / mpmath", "reference data"]),
        (172, "1. krylov", ["tenferro tensors,", "implicit diff"]),
        (312, "2. optimize", ["L-BFGS, LM,", "implicit diff"]),
        (452, "3. special", ["xsf port +", "ExtensionOp"]),
        (592, "4. integrate", ["quadrature,", "ODE + adjoint"]),
        (732, "5. interpolate", ["splines,", "Chebyshev"]),
    ]
    for x, name, lines in crates:
        f.box(x, 382, 132, 118, name, lines, style="plain")

    # lower layers -> crates
    f.arrow([(130, 204), (130, 300), (238, 300), (238, 378)])
    f.arrow([(560, 204), (560, 300), (518, 300), (518, 378)])
    f.arrow([(775, 204), (775, 300), (798, 300), (798, 378)])
    # inside tenferro-sci-rs
    f.arrow([(166, 441), (170, 441)])
    f.arrow([(306, 441), (310, 441)])
    f.arrow([(378, 502), (378, 540), (658, 540), (658, 504)],
            label="Newton steps, implicit diff", label_at=(518, 540))
    f.save("sci_order.svg")


if __name__ == "__main__":
    sci_layers()
    sci_adapter()
    sci_release()
    sci_order()
