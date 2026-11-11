"""Export every tensor4all logo (tensor4all, tenferro, colmat) from one place.

usage: python3 branding/tools/export_logos.py <IBMPlexSans-SemiBold.ttf> [branding_dir]

All wordmarks share one typeface (IBM Plex Sans SemiBold, SIL OFL 1.1), one
tracking and one palette, so the family stays consistent when one of them
changes. Text is shaped with HarfBuzz (kerning applied) and converted to
outlines. PNGs are rendered with `rsvg-convert` when it is on PATH.

Needs: fontTools, uharfbuzz.
"""
import math
import pathlib
import shutil
import subprocess
import sys

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------- shared style

TRACKING_EM = -0.03

# The family palette is the tensor4all.org site palette.
FAMILY = {
    "light": {"ink": "#1d2126", "accent": "#a3521a", "pale": "#d6d2c8", "bg": "#fbfaf7"},
    "dark": {"ink": "#e6e8eb", "accent": "#f0a46b", "pale": "#3a414a", "bg": "#14171b"},
}
# colmat keeps its original fresh palette as an alternative.
COLMAT_FRESH = {
    "light": {"ink": "#25323e", "accent": "#dca240", "pale": "#d3d6d8", "bg": "#f7f6f3"},
    "dark": {"ink": "#e3e8ec", "accent": "#e8b24f", "pale": "#3a4450", "bg": "#14171b"},
}

# Text size relative to the mark: the x-height is 0.47 of the mark height, and
# the gap between mark and text is 0.2 of the mark height (tensor4all values).
XHEIGHT_PER_MARK = 0.468
GAP_PER_MARK = 0.204


class Font:
    def __init__(self, path):
        self.tt = TTFont(path)
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upm = self.tt["head"].unitsPerEm
        self.xheight = self.tt["OS/2"].sxHeight
        blob = hb.Blob.from_file_path(path)
        self.hb_font = hb.Font(hb.Face(blob))

    def size_for_xheight(self, xheight):
        return xheight * self.upm / self.xheight

    def run(self, text, size, x, baseline):
        """Outline `text` with kerning and tracking.

        Returns (list of (char, path d), (xmin, ymin, xmax, ymax), advance end x).
        """
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {"kern": True, "liga": False})
        scale = size / self.upm
        bounds = BoundsPen(self.glyphs)
        out, pen_x = [], x
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            name = self.order[info.codepoint]
            gx = pen_x + pos.x_offset * scale
            transform = (scale, 0, 0, -scale, gx, baseline - pos.y_offset * scale)
            pen = SVGPathPen(self.glyphs)
            self.glyphs[name].draw(TransformPen(pen, transform))
            self.glyphs[name].draw(TransformPen(bounds, transform))
            out.append((text[info.cluster], pen.getCommands()))
            pen_x += pos.x_advance * scale + TRACKING_EM * size
        return out, bounds.bounds, pen_x - TRACKING_EM * size


def svg(box, body, title, style=""):
    x0, y0, x1, y1 = box
    style_block = f"<style>{style}</style>" if style else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>{style_block}{body}</svg>\n')


def square_around(box, pad=0.0, side=None):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = side if side is not None else max(x1 - x0, y1 - y0) + 2 * pad
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def auto_style(palette, roles):
    light = "".join(f".{r}{{{p}:{palette['light'][c]}}}" for r, (p, c) in roles.items())
    dark = "".join(f".{r}{{{p}:{palette['dark'][c]}}}" for r, (p, c) in roles.items())
    return light + f"@media (prefers-color-scheme:dark){{{dark}}}"


def text_paths(glyphs, color_of):
    return "".join(f'<path fill="{color_of(i, ch)}" d="{d}"/>' for i, (ch, d) in enumerate(glyphs))


# ------------------------------------------------------------------ tensor4all
# A tensor network drawn as a circuit. Its bonds on the left, middle and right
# mimic the t, the 4 (in copper) and the a of the name.

# Drawn for print: about 2.4 lines per mm at a left-chest size, no enclosed
# holes narrower than 1 mm, and solid nodes instead of hollow rings, which
# fill in on fabric. See branding/logo/README.md, "Print".
T4A_STROKE = 10.0
T4A_NODE_R = 1.6           # node radius in stroke widths
T4A_INK_LINES = [
    "M68 24V193A17 17 0 0 0 85 210H110",     # the t: left leg, bottom-left corner
    "M36 56H296A15 15 0 0 1 311 71V210",     # top line, right side
    "M269 118H342M269 210H342",               # right legs
    "M269 118A46 46 0 0 0 269 210",          # the a: a semicircular bond
]
T4A_ACCENT_LINES = ["M110 210L189 56V252M110 210H269"]  # the 4
T4A_INK_NODES = [(68, 56), (311, 118), (311, 210)]
T4A_ACCENT_NODES = [(189, 56), (110, 210), (189, 210)]
T4A_TEXT_H = 225.0         # mark height that sets the wordmark text size


def t4a_box(weight=1.0):
    """Outer edge of strokes and nodes."""
    h = T4A_STROKE * weight / 2
    return (36 - h, 24 - h, 342 + h, 252 + h)


T4A_BOX = t4a_box()


def t4a_mark(colors=None, cls=False, weight=1.0, uid="t4a"):
    """Bonds as strokes, nodes as solid dots; `weight` scales both for small sizes."""
    sw = T4A_STROKE * weight
    r = T4A_NODE_R * sw

    def stroke(role):
        return f'class="{role}"' if cls else f'stroke="{colors[role]}"'

    def fill(role):
        return f'class="{role}f"' if cls else f'fill="{colors[role]}"'

    ink = "".join(f'<path d="{d}"/>' for d in T4A_INK_LINES)
    acc = "".join(f'<path d="{d}"/>' for d in T4A_ACCENT_LINES)
    dots_ink = "".join(f'<circle cx="{x}" cy="{y}" r="{r:.2f}"/>' for x, y in T4A_INK_NODES)
    dots_acc = "".join(f'<circle cx="{x}" cy="{y}" r="{r:.2f}"/>' for x, y in T4A_ACCENT_NODES)
    return (f'<g fill="none" stroke-width="{sw:.2f}" stroke-linejoin="round">'
            f'<g {stroke("ink")}>{ink}</g><g {stroke("accent")}>{acc}</g></g>'
            f'<g {fill("ink")}>{dots_ink}</g><g {fill("accent")}>{dots_acc}</g>')


def t4a_four(colors=None, cls=False):
    """16 px icon: only the copper 4."""
    paint = 'class="accent"' if cls else f'stroke="{colors["accent"]}"'
    return (f'<path {paint} fill="none" stroke-width="30" stroke-linecap="round" stroke-linejoin="round" '
            f'd="M110 210L189 56V239M110 210H259"/>')


# -------------------------------------------------------------------- tenferro
# A hex nut (ferro: iron, which rusts: Rust) with a tensor product as its hole.

TF_NUT = "100,60 80,94.64 40,94.64 20,60 40,25.36 80,25.36"  # circumradius 40, snug around the hole
TF_STROKE = 7.0       # heavy enough for embroidery at a left-chest size
TF_STROKE_SMALL = 9.0
def tf_box(sw=TF_STROKE):
    return (20 - sw / 2, 25.36 - sw / 2, 100 + sw / 2, 94.64 + sw / 2)


TF_BOX = tf_box()


def tf_mark(colors=None, cls=False, sw=TF_STROKE, cross_only=False):
    def paint(role):
        return f'class="{role}"' if cls else f'stroke="{colors[role]}"'

    a, b = 44.5, 75.5
    parts = [f'<polygon {paint("ink")} stroke-linejoin="round" points="{TF_NUT}"/>']
    if not cross_only:
        parts.append(f'<circle {paint("accent")} cx="60" cy="60" r="22"/>')
    parts.append(f'<path {paint("accent")} stroke-linecap="round" d="M{a} {a}L{b} {b}M{b} {a}L{a} {b}"/>')
    return f'<g fill="none" stroke-width="{sw}">' + "".join(parts) + "</g>"


# ---------------------------------------------------------------------- colmat
# The m of colmat drawn as matrix entries: two peaks and the col (mountain
# pass, saddle point) between them, marked in the accent color.

COLMAT_M = [  # 4 x 9; '#' entry, 'o' the col, '.' zero. Few, large entries print cleanly.
    "..#...#..",
    ".###.###.",
    "####o####",
    "#########",
]
COLMAT_SMALL = [  # 4 x 7, for favicons
    ".#...#.",
    "###.###",
    "###o###",
    "#######",
]
COLMAT_TINY = [  # 3 x 5, for 16 px
    ".#.#.",
    "##o##",
    "#####",
]
DOT_RATIO = 0.32  # dot radius / pitch: gaps of 0.36 pitch, over 1 mm at a left-chest size
DOT_RATIO_SMALL = 0.42  # fuller dots for favicons


def dot_grid(pattern, x0, top, pitch, colors=None, cls=False, zeros=False, ratio=DOT_RATIO):
    r = ratio * pitch
    out = []
    for i, row in enumerate(pattern):
        for j, c in enumerate(row):
            cx, cy = x0 + r + j * pitch, top + r + i * pitch
            role = {"#": "ink", "o": "accent", ".": "pale"}[c]
            if role == "pale" and not zeros:
                continue
            paint = f'class="{role}f"' if cls else f'fill="{colors[role]}"'
            out.append(f'<circle {paint} cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
    width = 2 * r + (len(pattern[0]) - 1) * pitch
    height = 2 * r + (len(pattern) - 1) * pitch
    return "".join(out), (x0, top, x0 + width, top + height)


def pitch_for_height(pattern, height):
    return height / (2 * DOT_RATIO + len(pattern) - 1)


# ---------------------------------------------------------------------- output

def main():
    font = Font(sys.argv[1])
    root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).resolve().parents[1])
    written = []

    def write(project, name, text):
        d = root / project / "svg"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
        written.append(f"{project}/svg/{name}")

    # ---- tensor4all
    p = "logo"
    mark_h = T4A_TEXT_H
    for theme in ("light", "dark"):
        c = FAMILY[theme]
        sfx = "" if theme == "light" else "-dark"
        write(p, f"tensor4all-mark{sfx}.svg", svg(square_around(T4A_BOX, 8), t4a_mark(c, uid=f"m{sfx}"), "tensor4all"))
        size = font.size_for_xheight(XHEIGHT_PER_MARK * mark_h)
        baseline = 201.0
        glyphs, gb, _ = font.run("tensor4all", size, 0, baseline)
        dx = T4A_BOX[2] + GAP_PER_MARK * mark_h - gb[0]
        glyphs, gb, _ = font.run("tensor4all", size, dx, baseline)
        text = text_paths(glyphs, lambda i, ch: c["accent"] if ch == "4" else c["ink"])
        box = (T4A_BOX[0] - 8, T4A_BOX[1] - 8, gb[2] + 8, T4A_BOX[3] + 8)
        write(p, f"tensor4all-wordmark{sfx}.svg", svg(box, t4a_mark(c, uid=f"w{sfx}") + text, "tensor4all"))
        bg = f'<rect x="-1000" y="-1000" width="3000" height="3000" fill="{c["bg"]}"/>'
        write(p, f"tensor4all-avatar{sfx}.svg",
              svg(square_around(T4A_BOX, side=470), bg + t4a_mark(c, uid=f"a{sfx}"), "tensor4all"))
    t4a_auto = auto_style(FAMILY, {"ink": ("stroke", "ink"), "accent": ("stroke", "accent"),
                                   "inkf": ("fill", "ink"), "accentf": ("fill", "accent")})
    write(p, "tensor4all-mark-auto.svg",
          svg(square_around(T4A_BOX, 8), t4a_mark(cls=True, uid="auto"), "tensor4all", t4a_auto))
    write(p, "favicon.svg",
          svg(square_around(t4a_box(1.3), 10), t4a_mark(cls=True, weight=1.3, uid="fav"), "tensor4all", t4a_auto))
    c = FAMILY["light"]
    bg = f'<rect x="-1000" y="-1000" width="3000" height="3000" fill="{c["bg"]}"/>'
    write(p, "tensor4all-icon-small.svg",
          svg(square_around(t4a_box(1.3), 14), bg + t4a_mark(c, weight=1.3, uid="s"), "tensor4all"))
    write(p, "tensor4all-icon-16.svg",
          svg(square_around((95, 41, 274, 254), 6), bg + t4a_four(c), "tensor4all"))

    # ---- tenferro
    p = "tenferro"
    mark_h = TF_BOX[3] - TF_BOX[1]
    for theme in ("light", "dark"):
        c = FAMILY[theme]
        sfx = "" if theme == "light" else "-dark"
        write(p, f"tenferro-mark{sfx}.svg", svg(square_around(TF_BOX, 4), tf_mark(c), "tenferro"))
        size = font.size_for_xheight(XHEIGHT_PER_MARK * mark_h)
        glyphs, gb, _ = font.run("tenferro", size, 0, 0)
        dy = 60 - (gb[1] + gb[3]) / 2
        dx = TF_BOX[2] + GAP_PER_MARK * mark_h - gb[0]
        glyphs, gb, _ = font.run("tenferro", size, dx, dy)
        text = text_paths(glyphs, lambda i, ch: c["accent"] if i >= 3 else c["ink"])
        box = (TF_BOX[0] - 4, TF_BOX[1] - 4, gb[2] + 4, TF_BOX[3] + 4)
        write(p, f"tenferro-wordmark{sfx}.svg", svg(box, tf_mark(c) + text, "tenferro"))
        bg = f'<rect x="-100" y="-100" width="400" height="400" fill="{c["bg"]}"/>'
        write(p, f"tenferro-avatar{sfx}.svg", svg(square_around(TF_BOX, side=146), bg + tf_mark(c), "tenferro"))
    tf_auto = auto_style(FAMILY, {"ink": ("stroke", "ink"), "accent": ("stroke", "accent")})
    write(p, "tenferro-mark-auto.svg", svg(square_around(TF_BOX, 4), tf_mark(cls=True), "tenferro", tf_auto))
    write(p, "favicon.svg", svg(square_around(tf_box(TF_STROKE_SMALL), 4), tf_mark(cls=True, sw=TF_STROKE_SMALL), "tenferro", tf_auto))
    c = FAMILY["light"]
    bg = f'<rect x="-100" y="-100" width="400" height="400" fill="{c["bg"]}"/>'
    write(p, "tenferro-icon-small.svg", svg(square_around(tf_box(TF_STROKE_SMALL), 4), bg + tf_mark(c, sw=TF_STROKE_SMALL), "tenferro"))
    write(p, "tenferro-icon-16.svg", svg(square_around(tf_box(12), 4), bg + tf_mark(c, sw=12, cross_only=True), "tenferro"))

    # ---- colmat
    p = "colmat"
    size = 193.5           # same text size as the tensor4all wordmark
    baseline = 0.0
    xh = font.xheight * size / font.upm
    pitch = pitch_for_height(COLMAT_M, xh)
    gap = 0.30 * xh
    for pal_name, pal in (("", FAMILY), ("-fresh", COLMAT_FRESH)):
        for zeros in (False, True):
            for theme in ("light", "dark"):
                c = pal[theme]
                sfx = ("-grid" if zeros else "") + pal_name + ("" if theme == "light" else "-dark")
                # Equal visible gaps on both sides of the dots, measured from glyph outlines.
                col, cb, _ = font.run("col", size, 0, baseline)
                dots, db = dot_grid(COLMAT_M, cb[2] + gap, baseline - xh, pitch, c, zeros=zeros)
                _, ab0, _ = font.run("at", size, 0, baseline)
                at, ab, _ = font.run("at", size, db[2] + gap - ab0[0], baseline)
                ink = lambda i, ch: c["ink"]
                body = text_paths(col, ink) + dots + text_paths(at, ink)
                box = (cb[0] - 12, cb[1] - 12, ab[2] + 12, max(cb[3], ab[3]) + 12)
                write(p, f"colmat-wordmark{sfx}.svg", svg(box, body, "colmat"))
        for theme in ("light", "dark"):
            c = pal[theme]
            sfx = pal_name + ("" if theme == "light" else "-dark")
            dots, db = dot_grid(COLMAT_M, 0, 0, 20, c)
            write(p, f"colmat-mark{sfx}.svg", svg(square_around(db, 8), dots, "colmat"))
            bg = f'<rect x="-500" y="-500" width="1200" height="1200" fill="{c["bg"]}"/>'
            write(p, f"colmat-avatar{sfx}.svg", svg(square_around(db, side=2.2 * math.hypot(db[2] / 2, db[3] / 2)), bg + dots, "colmat"))
    cm_auto = auto_style(FAMILY, {"inkf": ("fill", "ink"), "accentf": ("fill", "accent"), "palef": ("fill", "pale")})
    dots, db = dot_grid(COLMAT_M, 0, 0, 20, cls=True)
    write(p, "colmat-mark-auto.svg", svg(square_around(db, 8), dots, "colmat", cm_auto))
    dots, db = dot_grid(COLMAT_SMALL, 0, 0, 20, cls=True, ratio=DOT_RATIO_SMALL)
    write(p, "favicon.svg", svg(square_around(db, 4), dots, "colmat", cm_auto))
    c = FAMILY["light"]
    bg = f'<rect x="-500" y="-500" width="1200" height="1200" fill="{c["bg"]}"/>'
    dots, db = dot_grid(COLMAT_SMALL, 0, 0, 20, c, ratio=DOT_RATIO_SMALL)
    write(p, "colmat-icon-small.svg", svg(square_around(db, 8), bg + dots, "colmat"))
    dots, db = dot_grid(COLMAT_TINY, 0, 0, 20, c, ratio=DOT_RATIO_SMALL)
    write(p, "colmat-icon-16.svg", svg(square_around(db, 6), bg + dots, "colmat"))

    print("\n".join(written))
    render_pngs(root)


PNG_JOBS = {
    "logo": "tensor4all",
    "tenferro": "tenferro",
    "colmat": "colmat",
}


def render_pngs(root):
    if not shutil.which("rsvg-convert"):
        print("rsvg-convert not found: PNGs not rendered", file=sys.stderr)
        return

    def run(src, out, *args):
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsvg-convert", *args, str(src), "-o", str(out)], check=True)

    for folder, name in PNG_JOBS.items():
        svgd, pngd = root / folder / "svg", root / folder / "png"
        run(svgd / f"{name}-wordmark.svg", pngd / f"{name}-wordmark-240h.png", "-h", "240")
        run(svgd / f"{name}-wordmark-dark.svg", pngd / f"{name}-wordmark-dark-240h.png", "-h", "240")
        for s in (512, 1024):
            run(svgd / f"{name}-avatar.svg", pngd / f"{name}-avatar-{s}.png", "-w", str(s))
        run(svgd / f"{name}-avatar-dark.svg", pngd / f"{name}-avatar-dark-512.png", "-w", "512")
        small = svgd / f"{name}-icon-small.svg"
        tiny = svgd / f"{name}-icon-16.svg"
        run(tiny if tiny.exists() else small, pngd / "favicon-16.png", "-w", "16")
        for s in (32, 48):
            run(small, pngd / f"favicon-{s}.png", "-w", str(s))
        run(svgd / f"{name}-avatar.svg", pngd / "apple-touch-icon.png", "-w", "180")
    print("PNGs rendered")


if __name__ == "__main__":
    main()
