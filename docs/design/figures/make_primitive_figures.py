#!/usr/bin/env python3
"""Generate the primitive-kind figures (P1 to P10) for
next-generation-dense-linear-algebra.md.

Usage (from the repository root):

    python3 docs/design/figures/make_primitive_figures.py

Writes one SVG per kind next to this script, named prim_p<N>_<name>.svg.
Standard library only; the palette and the Figure class come from
make_figures.py so that every design figure looks the same.

Colour code of every figure:
  purple  read only
  green   written (updated in place or overwritten)
  amber   computed in the same pass (a by-product)
  neutral a small or scalar operand
"""

from make_figures import Figure, STYLES

READ, WRITE, SIDE, SMALL = "purple", "green", "amber", "neutral"


def rect(f, x, y, w, h, style="plain", r=3, dashed=False, opacity=1.0):
    fill, stroke, _ = STYLES[style]
    dash = ' stroke-dasharray="5 3"' if dashed else ""
    op = f' fill-opacity="{opacity}"' if opacity != 1.0 else ""
    f.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}"{op} stroke="{stroke}" stroke-width="1.2"{dash}/>')


def poly(f, pts, style, dashed=False):
    fill, stroke, _ = STYLES[style]
    dash = ' stroke-dasharray="5 3"' if dashed else ""
    d = " ".join(f"{x},{y}" for x, y in pts)
    f.parts.append(
        f'<polygon points="{d}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.2"{dash}/>')


def line(f, pts, color="#77756F", width=1.5, dashed=False, arrow=False):
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    mk = ' marker-end="url(#head)"' if arrow else ""
    f.parts.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash}{mk}/>')


def lower(f, x, y, n, style, unit=False):
    """Lower triangle of an n x n square at (x, y)."""
    rect(f, x, y, n, n, "plain")
    poly(f, [(x, y), (x, y + n), (x + n, y + n)], style)
    if unit:
        line(f, [(x, y), (x + n, y + n)], color=STYLES[style][1], width=2.5)


def upper(f, x, y, n, style):
    rect(f, x, y, n, n, "plain")
    poly(f, [(x, y), (x + n, y), (x + n, y + n)], style)


def label(f, x, y, s, size=14, weight=400, color="#2C2C2A", anchor="middle",
          mono=False):
    f.text(x, y, s, size=size, weight=weight, color=color, anchor=anchor,
           mono=mono)


def op(f, x, y, s):
    label(f, x, y, s, size=20, color="#4A4844")


def title(f, s, sub=None):
    label(f, 24, 30, s, size=16, weight=600, anchor="start")
    if sub:
        label(f, 24, 50, sub, size=12.5, color="#4A4844", anchor="start")


def legend(f, y, items=("read", "written", "same pass", "small")):
    names = {"read": (READ, "read only"),
             "written": (WRITE, "written"),
             "same pass": (SIDE, "computed in the same pass"),
             "small": (SMALL, "small / scalar operand")}
    x = 24
    for key in items:
        style, text = names[key]
        rect(f, x, y - 11, 14, 14, style, r=2)
        label(f, x + 20, y, text, size=12, color="#4A4844", anchor="start")
        x += 34 + len(text) * 6.6


# P1 ----------------------------------------------------------------------

def p1():
    f = Figure(860, 300)
    title(f, "P1  Level-3 products",
          "compute-bound; a non-native scalar overrides the kernel "
          "(packing into high/low parts, Dot2, Ozaki split)")
    # gemm
    label(f, 140, 85, "gemm", weight=600)
    rect(f, 24, 100, 60, 70, WRITE)
    label(f, 54, 140, "C")
    op(f, 100, 142, "←")
    label(f, 124, 142, "α", size=15)
    rect(f, 136, 100, 40, 70, READ)
    label(f, 156, 140, "A")
    rect(f, 186, 115, 60, 40, READ)
    label(f, 216, 140, "B")
    op(f, 262, 142, "+")
    label(f, 280, 142, "β", size=15)
    rect(f, 292, 100, 60, 70, WRITE, dashed=True)
    label(f, 322, 140, "C")
    label(f, 188, 198, "C ← αAB + βC", size=12.5, mono=True)
    # gemmt
    label(f, 505, 85, "triangular output (herk/syrk/gemmt)", weight=600)
    rect(f, 400, 100, 70, 70, "plain")
    poly(f, [(400, 100), (400, 170), (470, 170)], WRITE)
    label(f, 418, 155, "C")
    op(f, 487, 142, "←")
    rect(f, 505, 100, 30, 70, READ)
    label(f, 520, 140, "A")
    rect(f, 545, 120, 70, 30, READ)
    label(f, 580, 140, "Aᴴ")
    op(f, 630, 142, "+")
    label(f, 655, 142, "C", size=15)
    label(f, 505, 198, "only one triangle of C is written", size=12.5)
    # trmm
    label(f, 765, 85, "trmm", weight=600)
    lower(f, 700, 105, 55, READ)
    label(f, 714, 152, "L")
    rect(f, 765, 105, 70, 55, WRITE)
    label(f, 800, 138, "B")
    label(f, 768, 198, "B ← L·B (in place)", size=12.5, mono=True)
    legend(f, 270, ("read", "written"))
    f.save("prim_p1_level3_products.svg")


# P2 ----------------------------------------------------------------------

def p2():
    f = Figure(860, 270)
    title(f, "P2  Level-3 triangular solve (trsm)",
          "many right-hand sides; the solution overwrites B")
    lower(f, 60, 80, 110, READ)
    label(f, 90, 165, "L")
    label(f, 115, 210, "triangular, read", size=12.5)
    op(f, 195, 140, "X")
    rect(f, 220, 80, 150, 110, WRITE)
    label(f, 295, 140, "B  →  X", size=15)
    label(f, 295, 210, "overwritten", size=12.5)
    label(f, 600, 110, "solve  L·X = B", size=15, mono=True)
    label(f, 600, 140, "blocked: one diagonal block by a small", size=12.5)
    label(f, 600, 160, "solve, the rest of B by P1 (gemm)", size=12.5)
    legend(f, 250, ("read", "written"))
    f.save("prim_p2_trsm.svg")


# P3 ----------------------------------------------------------------------

def p3():
    f = Figure(860, 330)
    title(f, "P3  Block-reflector apply (larfb, larft)",
          "C ← (I − V·T·Vᴴ)·C, from the left or the right; three products "
          "fused and split by colmat")
    # V
    rect(f, 30, 80, 40, 150, "plain")
    poly(f, [(30, 80), (70, 120), (70, 230), (30, 230)], READ)
    line(f, [(30, 80), (70, 120)], color=STYLES[READ][1], width=2.5)
    label(f, 50, 180, "V")
    label(f, 50, 250, "k reflectors", size=12)
    # T
    upper(f, 90, 80, 40, SMALL)
    label(f, 118, 108, "T", size=13)
    label(f, 110, 140, "k × k", size=12)
    # C
    rect(f, 150, 80, 110, 150, WRITE)
    label(f, 205, 160, "C")
    # steps
    x0 = 320
    label(f, x0, 90, "1.", weight=600, anchor="start")
    label(f, x0 + 22, 90, "W ← Vᴴ·C", mono=True, anchor="start")
    label(f, x0 + 170, 90, "(k × n, gemm/trmm)", size=12.5, anchor="start",
          color="#4A4844")
    label(f, x0, 125, "2.", weight=600, anchor="start")
    label(f, x0 + 22, 125, "W ← T·W", mono=True, anchor="start")
    label(f, x0 + 170, 125, "(small triangular product)", size=12.5,
          anchor="start", color="#4A4844")
    label(f, x0, 160, "3.", weight=600, anchor="start")
    label(f, x0 + 22, 160, "C ← C − V·W", mono=True, anchor="start")
    label(f, x0 + 170, 160, "(gemm/trmm)", size=12.5, anchor="start",
          color="#4A4844")
    rect(f, x0, 185, 500, 60, "amber", r=6, opacity=0.6)
    label(f, x0 + 250, 210, "larft forms T from V and the τ's;", size=12.5)
    label(f, x0 + 250, 230, "V is unit lower trapezoidal (diagonal implicit)",
          size=12.5)
    legend(f, 305, ("read", "written", "small"))
    f.save("prim_p3_block_reflector.svg")


# P4 ----------------------------------------------------------------------

def p4():
    f = Figure(860, 440)
    title(f, "P4  Fused sweep",
          "one pass over A: a rank-0/1/2 update, plus any of the by-products "
          "the next step needs")
    dy = 40
    # inputs of the update
    rect(f, 30, 90 + dy, 22, 170, READ)
    label(f, 41, 280 + dy, "u₁", size=13)
    rect(f, 58, 90 + dy, 22, 170, READ)
    label(f, 69, 280 + dy, "u₂", size=13)
    rect(f, 110, 62 + dy, 190, 18, READ)
    label(f, 316, 76 + dy, "v₁ᴴ", size=13, anchor="start")
    rect(f, 110, 40 + dy, 190, 18, READ, opacity=0.8)
    label(f, 316, 54 + dy, "v₂ᴴ", size=13, anchor="start")
    # A, swept column by column
    rect(f, 110, 90 + dy, 190, 170, WRITE)
    for i in range(1, 6):
        line(f, [(110 + i * 190 / 6, 90 + dy), (110 + i * 190 / 6, 260 + dy)],
             color="#9CC9B2", width=1)
    rect(f, 110 + 2 * 190 / 6, 90 + dy, 190 / 6, 170, SIDE, r=0,
         opacity=0.55)
    line(f, [(125, 275 + dy), (290, 275 + dy)], arrow=True)
    label(f, 205, 295 + dy, "column sweep (one read, one write of A)",
          size=12)
    label(f, 205, 180 + dy, "A", size=18)
    # by-products
    rect(f, 330, 90 + dy, 22, 170, SIDE)
    label(f, 341, 280 + dy, "A·x", size=12.5)
    rect(f, 110, 312 + dy, 190, 18, SIDE)
    label(f, 316, 326 + dy, "yᴴ·A", size=12.5, anchor="start")
    label(f, 230, 140 + dy, "★", size=18, color=STYLES["amber"][1])
    # right column: text
    x0 = 400
    label(f, x0, 100 + dy, "update", weight=600, anchor="start")
    label(f, x0, 122 + dy, "A ← A − u₁v₁ᴴ − u₂v₂ᴴ   (rank 0, 1 or 2;",
          mono=True, size=12.5, anchor="start")
    label(f, x0, 140 + dy, "      full matrix or one triangle)", mono=True,
          size=12.5, anchor="start")
    label(f, x0, 172 + dy, "by-products, from the updated columns",
          weight=600, anchor="start")
    rows = [
        ("A·x, yᴴ·A", "Hessenberg, bidiagonal (≈ GEMVER)"),
        ("symmetric A·x from a triangle", "tridiagonal"),
        ("column-norm downdate", "column-pivoted QR"),
        ("|a| max per column  ★", "LU leaf (partial pivoting)"),
        ("|a| max over A  ★", "full-pivoting LU, Bunch–Kaufman"),
    ]
    y = 196 + dy
    for a, b in rows:
        label(f, x0 + 10, y, "· " + a, size=12.5, anchor="start")
        label(f, x0 + 250, y, b, size=12.5, anchor="start", color="#4A4844")
        y += 22
    label(f, x0, y + 14, "★: index of the first maximum (tie rule is "
          "part of the contract)", size=12, anchor="start", color="#4A4844")
    legend(f, 420, ("read", "written", "same pass"))
    f.save("prim_p4_fused_sweep.svg")


# P5 ----------------------------------------------------------------------

def p5():
    f = Figure(860, 280)
    title(f, "P5  Level-2 triangular (trsv, trmv)",
          "one right-hand side: a recurrence, each entry waits for the "
          "previous ones")
    lower(f, 60, 80, 130, READ)
    label(f, 95, 175, "L")
    op(f, 215, 150, "x ←")
    label(f, 250, 150, "L⁻¹", size=16)
    xs = 290
    for i in range(6):
        rect(f, xs, 80 + i * 22, 26, 20, WRITE, r=2)
        label(f, xs + 13, 95 + i * 22, f"x{i}", size=11)
    for i in range(5):
        f.parts.append(
            f'<path d="M {xs + 28} {90 + i * 22} C {xs + 50} {95 + i * 22}, '
            f'{xs + 50} {105 + i * 22}, {xs + 28} {112 + i * 22}" fill="none" '
            f'stroke="#77756F" stroke-width="1.3" marker-end="url(#head)"/>')
    label(f, 560, 110, "xᵢ ← (xᵢ − Σⱼ<ᵢ lᵢⱼ xⱼ) / lᵢᵢ", mono=True, size=14)
    label(f, 560, 145, "memory-bound for f64; the dependency chain,", size=12.5)
    label(f, 560, 165, "not a sweep, is what sets the parallel structure",
          size=12.5)
    legend(f, 260, ("read", "written"))
    f.save("prim_p5_trsv.svg")


# P6 ----------------------------------------------------------------------

def p6():
    f = Figure(860, 330)
    title(f, "P6  Sequences of small transforms",
          "rotations on pairs of rows or columns; small reflectors chasing "
          "a bulge")
    # one rotation
    label(f, 140, 80, "one rotation", weight=600)
    rect(f, 30, 95, 200, 120, "plain")
    rect(f, 30, 125, 200, 18, WRITE, r=0)
    rect(f, 30, 175, 200, 18, WRITE, r=0)
    label(f, 18, 139, "i", size=12.5)
    label(f, 18, 189, "j", size=12.5)
    rect(f, 60, 235, 80, 50, SMALL)
    label(f, 100, 256, "c   s", mono=True, size=12.5)
    label(f, 100, 274, "−s̄  c", mono=True, size=12.5)
    label(f, 185, 262, "rows i, j ←", size=12, anchor="middle")
    label(f, 185, 278, "G · rows i, j", size=12, anchor="middle")
    # sequence (wavefront)
    label(f, 410, 80, "a sequence (lasr, QR sweeps)", weight=600)
    rect(f, 320, 95, 180, 180, "plain")
    for k in range(5):
        y = 100 + k * 34
        rect(f, 320, y, 180, 30, WRITE if k % 2 == 0 else "green", r=0,
             opacity=0.35 + 0.12 * k)
        label(f, 512, y + 20, f"G{k}", size=12, anchor="start")
    label(f, 410, 298, "fused application keeps rows in cache", size=12)
    # bulge chase
    label(f, 700, 80, "bulge chasing (Schur, QZ)", weight=600)
    n = 180
    x0, y0 = 620, 95
    rect(f, x0, y0, n, n, "plain")
    pts = [(x0, y0), (x0 + n, y0), (x0 + n, y0 + n), (x0 + n - n / 6, y0 + n),
           (x0, y0 + n / 6)]
    poly(f, pts, WRITE)
    for k in range(3):
        bx = x0 + 15 + k * 45
        by = y0 + 30 + k * 45
        rect(f, bx, by, 40, 40, SIDE if k == 1 else "plain", r=2,
             dashed=k != 1, opacity=0.9)
    line(f, [(x0 + 70, y0 + 95), (x0 + 110, y0 + 135)], arrow=True)
    label(f, 710, 298, "a 3-element reflector moves the bulge down", size=12)
    legend(f, 318, ("written", "small"))
    f.save("prim_p6_small_transforms.svg")


# P7 ----------------------------------------------------------------------

def p7():
    f = Figure(860, 250)
    title(f, "P7  Permutations (laswp, lapmt)",
          "swap row k with row piv[k], k = 0, 1, …; or columns")
    rect(f, 60, 75, 220, 140, "plain")
    rect(f, 60, 95, 220, 18, WRITE, r=0)
    rect(f, 60, 170, 220, 18, WRITE, r=0)
    label(f, 48, 109, "k", size=12.5)
    label(f, 36, 184, "piv[k]", size=12.5)
    f.parts.append(
        '<path d="M 290 104 C 330 110, 330 172, 290 179" fill="none" '
        'stroke="#77756F" stroke-width="1.5" marker-end="url(#head)" '
        'marker-start="url(#head)"/>')
    label(f, 560, 120, "no arithmetic: memory traffic only", size=13)
    label(f, 560, 145, "applied to whole rows, or to the part right of",
          size=12.5)
    label(f, 560, 165, "the panel (LU), in the order of the pivots", size=12.5)
    legend(f, 238, ("written",))
    f.save("prim_p7_permutations.svg")


# P8 ----------------------------------------------------------------------

def p8():
    f = Figure(860, 270)
    title(f, "P8  Level 1 and reductions",
          "vectors in, a vector or a scalar out")

    def vec(x, y, style, name):
        for i in range(5):
            rect(f, x, y + i * 20, 22, 18, style, r=2)
        label(f, x + 11, y + 118, name, size=12.5)

    # dot
    label(f, 85, 80, "dot, sum", weight=600)
    vec(40, 95, READ, "x")
    vec(72, 95, READ, "y")
    op(f, 112, 150, "→")
    rect(f, 128, 136, 24, 22, SIDE, r=2)
    label(f, 140, 213, "s = xᴴy", size=12, mono=True)
    # axpy
    label(f, 285, 80, "axpy, scal, copy", weight=600)
    vec(250, 95, READ, "x")
    vec(300, 95, WRITE, "y")
    label(f, 290, 233, "y ← αx + y", size=12, mono=True)
    # nrm2
    label(f, 485, 80, "norms with scaling", weight=600)
    vec(450, 95, READ, "x")
    op(f, 490, 150, "→")
    rect(f, 506, 136, 24, 22, SIDE, r=2)
    label(f, 545, 175, "‖x‖₂ without overflow:", size=12, anchor="start")
    label(f, 545, 193, "scale by safe_min/max", size=12, anchor="start")
    # iamax
    label(f, 730, 80, "|x| max and its index", weight=600)
    vec(700, 95, READ, "x")
    label(f, 711, 150, "★", size=14, color=STYLES["amber"][1])
    op(f, 740, 150, "→")
    label(f, 775, 155, "i = first max", size=12, anchor="start")
    legend(f, 258, ("read", "written", "same pass"))
    f.save("prim_p8_level1.svg")


# P9 ----------------------------------------------------------------------

def p9():
    f = Figure(860, 330)
    title(f, "P9  Scalar numerical kernels",
          "no SIMD and negligible time; the scalings and thresholds that "
          "decide correctness live here")
    items = [
        ("Householder generation (larfg)", "x  →  v, τ, β",
         "(I − τvvᴴ)x = βe₁"),
        ("rotation generation (lartg)", "(a, b)  →  c, s, r",
         "[c s; −s̄ c]·[a; b] = [r; 0]"),
        ("2×2 / 3×3 problems", "shifts, 2×2 eig and svd,",
         "shifted 2×2 solves"),
        ("secular equation (laed4)", "root λ of 1 + ρ Σ zᵢ²/(dᵢ − λ)",
         "between two poles"),
        ("dqds steps (lasq)", "one step of the", "qd recurrence"),
        ("safe reciprocal, hypot", "1/x and √(a²+b²)", "without overflow"),
    ]
    for k, (t, a, b) in enumerate(items):
        col, row = k % 3, k // 3
        x = 24 + col * 280
        y = 70 + row * 110
        rect(f, x, y, 260, 92, SMALL, r=8)
        label(f, x + 130, y + 22, t, size=13, weight=600)
        label(f, x + 130, y + 48, a, size=12.5, mono=True)
        label(f, x + 130, y + 68, b, size=12.5, mono=True)
    rect(f, 24, 292, 820, 28, "amber", r=6, opacity=0.6)
    label(f, 434, 311, "every one of them reads eps, safe_min or safe_max: "
          "the Precision contract", size=12.5)
    f.save("prim_p9_scalar_kernels.svg")


# P10 ---------------------------------------------------------------------

def p10():
    f = Figure(860, 330)
    title(f, "P10  Leaf factorisations (getf2, potf2, geqr2)",
          "an unblocked factorisation of a small block, composed from "
          "P4, P7, P8 and P9")
    rect(f, 40, 75, 200, 200, "plain")
    rect(f, 40, 75, 50, 200, WRITE, r=0)
    label(f, 65, 290, "panel", size=12.5)
    label(f, 150, 180, "rest of the matrix", size=12)
    label(f, 150, 198, "(P1/P2 later)", size=12)
    x0 = 300
    label(f, x0, 95, "for each column k of the panel:", weight=600,
          anchor="start")
    steps = [
        ("P8 / P4", "find the pivot (|a| max, first index)"),
        ("P7", "swap rows k and the pivot"),
        ("P9", "safe reciprocal of the pivot; scale the column"),
        ("P4", "rank-1 update of the rest of the panel"),
    ]
    y = 125
    for k, d in steps:
        rect(f, x0, y - 15, 64, 22, "amber", r=4, opacity=0.7)
        label(f, x0 + 32, y + 1, k, size=12)
        label(f, x0 + 80, y + 1, d, size=12.5, anchor="start")
        y += 32
    label(f, x0, y + 10, "small enough to stay in registers or L1: a natural "
          "override point", size=12.5, anchor="start", color="#4A4844")
    legend(f, 318, ("written",))
    f.save("prim_p10_leaf.svg")


if __name__ == "__main__":
    p1()
    p2()
    p3()
    p4()
    p5()
    p6()
    p7()
    p8()
    p9()
    p10()
