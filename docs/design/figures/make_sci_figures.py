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
    f = Figure(900, 600)

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

    f.box(30, 330, 470, 120, "tenferro", style="green", title_at="top")
    f.box(45, 370, 215, 64, "tensors, runtime",
          ["CPU / CUDA / Metal"], style="plain")
    f.box(270, 370, 215, 64, "autodiff",
          ["eager + traced"], style="plain")

    f.box(570, 330, 300, 120, "Independent algorithm cores",
          style="amber", title_at="top")
    f.box(585, 370, 270, 64, "krylov (Ryo Watanabe)",
          ["generic over Vector / LinearMap"], style="plain")

    f.box(570, 500, 300, 80, "Other users of the cores",
          ["symtn, mdarray, ndarray, ..."], style="neutral", dashed=True)

    f.arrow([(450, 102), (450, 136)])
    f.arrow([(265, 282), (265, 326)], label="tensors, AD rules",
            label_at=(265, 304))
    f.arrow([(720, 282), (720, 326)], label="algorithms",
            label_at=(720, 304))
    f.arrow([(720, 498), (720, 454)])
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


if __name__ == "__main__":
    sci_layers()
    sci_adapter()
    sci_release()
