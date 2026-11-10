"""Export production SVGs for the tenferro hex-nut logo (proposal L1).

usage: python3 export_logo.py <IBMPlexSans-Bold.ttf|woff> <outdir>

The mark is a hex nut drawn as line art with a tensor-product sign (a circled
cross) as its hole. All geometry is in a 120-unit box.
"""
import pathlib
import sys

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

PALETTE = {
    "light": {"nut": "#1e2126", "otimes": "#995629", "ink": "#1e2126",
              "ferro": "#995629", "bg": "#fbfaf7"},
    "dark": {"nut": "#f2f2f0", "otimes": "#c47a45", "ink": "#f2f2f0",
             "ferro": "#c47a45", "bg": "#14171b"},
}

# Stroke weights. "small" is tuned for 16-48 px.
WEIGHTS = {"regular": 4.5, "small": 9.0}

# Hex nut, flat top and bottom, circumradius 50 around (60, 60).
NUT = "110,60 85,103.3 35,103.3 10,60 35,16.7 85,16.7"
OTIMES_R = 22
CROSS = 15.5  # half-length of each diagonal of the cross, inside the circle

FONT_TRACKING_EM = -0.01

AUTO_STYLE = (
    ".nut{{stroke:{l[nut]}}}.otimes{{stroke:{l[otimes]}}}"
    "@media (prefers-color-scheme:dark){{.nut{{stroke:{d[nut]}}}.otimes{{stroke:{d[otimes]}}}}}"
).format(l=PALETTE["light"], d=PALETTE["dark"])


def mark_group(weight, colors, use_classes=False, cross_only=False):
    """SVG <g> drawing the mark in the 120-unit box."""
    sw = WEIGHTS[weight]

    def paint(role):
        return f'class="{role}"' if use_classes else f'stroke="{colors[role]}"'

    a, b = 60 - CROSS, 60 + CROSS
    parts = [f'<polygon {paint("nut")} stroke-linejoin="round" points="{NUT}"/>']
    if not cross_only:
        parts.append(f'<circle {paint("otimes")} cx="60" cy="60" r="{OTIMES_R}"/>')
    parts.append(f'<path {paint("otimes")} stroke-linecap="round" d="M{a} {a}L{b} {b}M{b} {a}L{a} {b}"/>')
    return f'<g fill="none" stroke-width="{sw}">' + "".join(parts) + "</g>"


def svg(x0, y0, view_w, view_h, body, title, style=""):
    style_block = f"<style>{style}</style>" if style else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {view_w:.2f} {view_h:.2f}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>{style_block}{body}</svg>\n')


def wordmark_paths(font_path, text, size):
    """Outline `text` at the origin; return ([(char, d)], glyph bounds)."""
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    bounds = BoundsPen(glyphs)
    out, pen_x = [], 0.0
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        transform = (scale, 0, 0, -scale, pen_x, 0)
        glyphs[name].draw(TransformPen(pen, transform))
        glyphs[name].draw(TransformPen(bounds, transform))
        out.append((ch, pen.getCommands()))
        pen_x += glyphs[name].width * scale + FONT_TRACKING_EM * size
    return out, bounds.bounds


def main():
    font_path, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name, text):
        (outdir / name).write_text(text)
        written.append(name)

    # Mark: transparent background.
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "-dark"
        write(f"tenferro-mark{suffix}.svg",
              svg(4, 4, 112, 112, mark_group("regular", PALETTE[theme]), "tenferro"))
    write("tenferro-mark-auto.svg",
          svg(4, 4, 112, 112, mark_group("regular", None, use_classes=True), "tenferro", AUTO_STYLE))
    # Favicon: heavier strokes, follows the OS theme.
    write("favicon.svg",
          svg(2, 2, 116, 116, mark_group("small", None, use_classes=True), "tenferro", AUTO_STYLE))

    # Horizontal wordmark: mark + outlined "tenferro", "ferro" in copper.
    # Glyph height is about 70 % of the mark, as in the tensor4all wordmark.
    font_size, gap = 84.3, 24
    glyphs, (gx0, gy0, gx1, gy1) = wordmark_paths(font_path, "tenferro", font_size)
    tx = 112.25 + gap - gx0
    ty = 60 - (gy0 + gy1) / 2  # center the glyph box on the mark
    for theme in ("light", "dark"):
        c = PALETTE[theme]
        text = "".join(
            f'<path fill="{c["ferro"] if i >= 3 else c["ink"]}" d="{d}"/>'
            for i, (_, d) in enumerate(glyphs))
        body = mark_group("regular", c) + f'<g transform="translate({tx:.2f} {ty:.2f})">{text}</g>'
        width = tx + gx1 + 6 - 4
        suffix = "" if theme == "light" else "-dark"
        write(f"tenferro-wordmark{suffix}.svg", svg(4, 0, width, 120, body, "tenferro"))

    # Square avatar with brand background (GitHub, social cards). Circle-crop safe.
    for theme in ("light", "dark"):
        c = PALETTE[theme]
        suffix = "" if theme == "light" else "-dark"
        body = (f'<rect x="-30" y="-30" width="180" height="180" fill="{c["bg"]}"/>'
                + mark_group("regular", c))
        write(f"tenferro-avatar{suffix}.svg", svg(-30, -30, 180, 180, body, "tenferro"))

    # Opaque small icons for PNG favicons; 16 px drops the circle.
    c = PALETTE["light"]
    bg = f'<rect x="0" y="0" width="120" height="120" fill="{c["bg"]}"/>'
    write("tenferro-icon-small.svg", svg(0, 0, 120, 120, bg + mark_group("small", c), "tenferro"))
    write("tenferro-icon-16.svg",
          svg(0, 0, 120, 120, bg + mark_group("small", c, cross_only=True), "tenferro"))
    print("\n".join(written))


if __name__ == "__main__":
    main()
