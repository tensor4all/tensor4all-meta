#!/usr/bin/env python3
"""Generate the SVG figures for core-boundary-and-execution-ownership.md.

Usage (from the repository root):

    python3 docs/design/figures/make_figures.py

Writes one SVG per figure next to this script. Standard library only.
Edit the figure functions below and rerun; commit the script and the SVGs.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# fill, stroke, text
STYLES = {
    "neutral": ("#F1EFEA", "#8A8780", "#2C2C2A"),
    "green": ("#E3F4EC", "#2F7A57", "#1F5A3F"),
    "purple": ("#EEEDFD", "#6A62D6", "#3A2F9A"),
    "amber": ("#FDF3E1", "#B7832F", "#6B4A12"),
    "red": ("#FBEAE6", "#C0563F", "#8A2E1C"),
    "plain": ("#FFFFFF", "#B8B5AE", "#2C2C2A"),
}
ARROW = "#77756F"


class Figure:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.parts = []

    def box(self, x, y, w, h, title=None, lines=(), style="neutral",
            title_at="center", mono_lines=False, dashed=False, r=10):
        fill, stroke, text = STYLES[style]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"{dash}/>')
        n = (1 if title else 0) + len(lines)
        if title_at == "top":
            ty = y + 24
        else:
            ty = y + h / 2 - (n - 1) * 10 + 5
        if title:
            self.text(x + w / 2, ty, title, size=15, weight=600, color=text)
            ty += 21
        for line in lines:
            self.text(x + w / 2, ty, line, size=12.5, color=text,
                      mono=mono_lines)
            ty += 18

    def text(self, x, y, s, size=12.5, weight=400, color="#2C2C2A",
             anchor="middle", mono=False):
        fam = MONO if mono else FONT
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
            f'{escape(s)}</text>')

    def arrow(self, points, label=None, label_at=None, dashed=False,
              label_lines=None):
        d = "M " + " L ".join(f"{x} {y}" for x, y in points)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="1.5"'
            f'{dash} marker-end="url(#head)"/>')
        lines = label_lines or ([label] if label else [])
        if lines:
            if label_at is None:
                (x1, y1), (x2, y2) = points[0], points[-1]
                label_at = ((x1 + x2) / 2, (y1 + y2) / 2)
            lx, ly = label_at
            wmax = max(len(s) for s in lines) * 6.9 + 14
            hh = 18 * len(lines) + 6
            self.parts.append(
                f'<rect x="{lx - wmax / 2}" y="{ly - hh / 2}" width="{wmax}" '
                f'height="{hh}" rx="4" fill="#FFFFFF"/>')
            for i, s in enumerate(lines):
                self.text(lx, ly - hh / 2 + 17 + 18 * i, s, size=12,
                          color="#4A4844")

    def svg(self):
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="{ARROW}" '
            'stroke-width="1.5"/></marker></defs>\n'
            f'<rect width="{self.w}" height="{self.h}" fill="#FCFCFA"/>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"

    def save(self, name):
        (OUT / name).write_text(self.svg())
        print("wrote", OUT / name)


def layers():
    f = Figure(800, 560)
    f.box(30, 20, 740, 150, "Downstream: symtn, tensor4all-rs, ...",
          style="neutral", title_at="top")
    f.box(55, 62, 330, 88, "Algorithms",
          ["fusion trees, blocks,", "network algorithms"], style="plain")
    f.box(415, 62, 330, 88, "Dedicated kernels",
          ["irregular batches,", "e.g. one GEMM per symmetry block"],
          style="plain")

    f.box(30, 225, 530, 175, "tenferro", style="green", title_at="top")
    f.box(55, 265, 250, 115, "Core",
          ["dense tensors, dtype, placement", "elementwise ops, dot_general",
           "transfers, CpuEngine"], style="plain")
    f.box(325, 265, 210, 115, "Extensions",
          ["linear algebra", "einsum, FFT"], style="plain")

    f.box(30, 455, 740, 90, "Provider crates", style="amber", title_at="top")
    for i, name in enumerate(["t4a-faer", "tlinalg", "tcontract", "cpueinsum"]):
        f.box(55 + i * 177, 490, 160, 40, lines=[name], style="plain",
              mono_lines=True, r=8)

    f.arrow([(220, 150), (220, 225)])
    f.arrow([(295, 400), (295, 455)])
    f.arrow([(665, 150), (665, 455)],
            label_lines=["direct call with", "engine.exec(budget)"],
            label_at=(665, 305))
    f.save("layers.svg")


def batches():
    f = Figure(800, 330)
    f.box(20, 20, 370, 290, "Regular batch", style="green", title_at="top")
    f.text(205, 66, "every item has the same shape", size=12.5,
           color="#1F5A3F")
    for k in range(4):
        x, y = 70 + k * 14, 95 + k * 14
        f.box(x, y, 90, 70, style="plain", r=4)
        f.box(x + 120, y, 70, 70, style="plain", r=4)
    f.text(157, 230, "A[b, i, j]", size=12, mono=True)
    f.text(267, 230, "B[b, j, k]", size=12, mono=True)
    f.text(205, 262, "one dot_general call", size=14, weight=600,
           color="#1F5A3F")
    f.text(205, 284, "in the tenferro core", size=12.5, color="#1F5A3F")

    f.box(410, 20, 370, 290, "Irregular batch", style="purple",
          title_at="top")
    f.text(595, 66, "each block has its own shape", size=12.5,
           color="#3A2F9A")
    blocks = [(440, 95, 40, 60, "3 x 5"), (500, 95, 90, 45, "7 x 4"),
              (610, 95, 30, 30, "2 x 2"), (660, 95, 70, 95, "5 x 8")]
    for x, y, w, h, lab in blocks:
        f.box(x, y, w, h, style="plain", r=4)
        f.text(x + w / 2, y + h + 16, lab, size=11.5, mono=True)
    f.text(595, 262, "the caller's own kernel", size=14, weight=600,
           color="#3A2F9A")
    f.text(595, 284, "loops over blocks, one provider GEMM each",
           size=12.5, color="#3A2F9A")
    f.save("batches.svg")


def ownership():
    f = Figure(820, 400)
    f.box(20, 20, 380, 360, "Today: tenferro owns the threads",
          style="red", title_at="top")
    f.box(55, 65, 240, 50, "caller / hataori", style="plain")
    f.box(55, 175, 240, 60, "ResourceArbiter",
          ["engine registry, owner locals"], style="plain")
    f.box(55, 295, 240, 60, "pool built by tenferro",
          ["held session, phase lease"], style="plain")
    f.arrow([(125, 115), (125, 175)], label="ask for admission",
            label_at=(125, 145))
    f.arrow([(175, 235), (175, 295)])
    f.arrow([(295, 325), (345, 325), (345, 90), (295, 90)],
            label_lines=["lends", "one worker", "per lane"],
            label_at=(345, 207))

    f.box(420, 20, 380, 360, "This design: the caller owns the threads",
          style="green", title_at="top")
    f.box(480, 65, 260, 50, "caller / hataori", style="plain")
    f.box(450, 180, 320, 175, "CpuEngine", ["used through &mut"],
          style="plain", title_at="top")
    f.box(470, 240, 280, 45, "Pool", ["built by the caller"], style="purple",
          r=8)
    f.box(470, 295, 280, 45, "Workspace",
          ["buffers, run scratch"], style="purple", r=8)
    f.arrow([(610, 115), (610, 180)], label="builds and holds",
            label_at=(610, 148))
    f.save("ownership.svg")


def operation():
    f = Figure(920, 300)
    xs = [20, 250, 480, 710]
    boxes = [
        ("caller thread", ["not a worker", "of the engine's pool"], "neutral"),
        ("CpuEngine", ["&mut, budget = whole pool", "error if the caller is",
                       "one of its own workers"], "green"),
        ("provider", ["tcontract, tlinalg, ...", "work model picks",
                      "how many workers"], "amber"),
        ("pool workers", ["only the chosen", "workers wake up"], "purple"),
    ]
    for x, (t, ls, s) in zip(xs, boxes):
        f.box(x, 30, 190, 120, t, ls, style=s)
    f.arrow([(210, 90), (250, 90)])
    f.arrow([(440, 90), (480, 90)])
    f.arrow([(670, 90), (710, 90)])
    f.text(230, 76, "op", size=11.5, color="#4A4844")
    f.text(460, 76, "Exec", size=11.5, color="#4A4844", mono=True)

    f.box(250, 200, 190, 70, "downstream kernel", ["e.g. symtn blocks"],
          style="neutral")
    f.arrow([(440, 235), (575, 235), (575, 150)],
            label="engine.exec(budget)", label_at=(560, 210))
    f.save("operation.svg")


def nesting():
    f = Figure(860, 470)
    f.box(40, 20, 780, 75, "hataori: k lane threads (its own pool)",
          ["builds k compute pools of m threads; CPU overlap is the "
           "caller's responsibility"], style="neutral")
    xs = [40, 330, 620]
    names = ["CpuEngine 1", "CpuEngine 2", "CpuEngine k"]
    cpus = ["CPUs 0 .. m-1", "CPUs m .. 2m-1", "CPUs (k-1)m .. km-1"]
    for x, n, c in zip(xs, names, cpus):
        f.arrow([(x + 100, 95), (x + 100, 140)], label="lane thread calls ops",
                label_at=(x + 100, 117))
        f.box(x, 140, 200, 230, None, style="green")
        f.text(x + 100, 172, n, size=16, weight=600, color="#1F5A3F")
        f.text(x + 100, 194, "&mut, one lane at a time", size=12.5,
               color="#1F5A3F")
        f.box(x + 20, 212, 160, 66, "Compute pool", ["m threads,", "built by hataori"],
              style="purple", r=8)
        f.box(x + 20, 290, 160, 60, "Workspace", ["per engine"],
              style="purple", r=8)
        f.text(x + 100, 400, c, size=12.5, mono=True, color="#4A4844")
    f.text(575, 260, "...", size=22, color="#77756F")
    f.text(430, 445, "Operations inside a lane use their engine's whole pool. "
           "Nesting = more engines, never a split pool.",
           size=12.5, color="#4A4844")
    f.save("nesting.svg")


def runtime_split():
    f = Figure(860, 430)
    f.box(230, 20, 400, 110, "Runtime (shared, prepared once)",
          ["graphs, prepared programs and plans,", "AD rules, values and gradients"],
          style="neutral", title_at="top")
    xs = [40, 330, 620]
    names = ["run 1", "run 2", "run k"]
    for x, n in zip(xs, names):
        f.arrow([(430, 130), (x + 100, 200)], dashed=True)
        lx = x + 128 if x == 330 else x + 100   # keep the middle label off its arrow
        f.text(lx, 190, n, size=12.5, color="#4A4844")
        f.box(x, 205, 200, 175, None, style="green")
        f.text(x + 100, 232, "CpuEngine", size=15, weight=600, color="#1F5A3F")
        f.text(x + 100, 252, "passed in per call", size=12, color="#1F5A3F")
        f.box(x + 20, 265, 160, 45, "Pool", style="purple", r=8)
        f.box(x + 20, 318, 160, 50, "Workspace",
              ["buffers, run scratch"], style="purple", r=8)
    f.text(575, 300, "...", size=22, color="#77756F")
    f.text(430, 412, "Dashed: concurrent runs share one compiled program. "
           "Workspaces and caches live in each run's engine.",
           size=12.5, color="#4A4844")
    f.save("runtime_split.svg")


if __name__ == "__main__":
    layers()
    batches()
    ownership()
    operation()
    nesting()
    runtime_split()
