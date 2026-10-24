"""Export production SVGs for the tensor4all tesseract logo (Concept A).

usage: python3 export_logo.py <font.woff|ttf> <outdir>
"""
import itertools
import math
import pathlib
import sys

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

PALETTE = {
    "light": {"edge": "#1f5fa8", "edge2": "#8fb0d8", "link": "#b9c9dc",
              "ink": "#1d2126", "four": "#a3521a", "bg": "#fbfaf7"},
    "dark": {"edge": "#79b0ee", "edge2": "#4a6f99", "link": "#34506f",
             "ink": "#e6e8eb", "four": "#f0a46b", "bg": "#14171b"},
}

# Stroke weights in units of a 120-unit box. "small" is tuned for 16-48 px.
WEIGHTS = {
    "regular": {"edge": 4.0, "edge2": 3.4, "link": 3.0, "inner": 0.48},
    "small": {"edge": 10.0, "edge2": 6.0, "link": 5.5, "inner": 0.50},
}


def tesseract_lines(inner_scale):
    """Edges of a tesseract as (group, p, q) in raw projected coordinates."""
    yaw, pitch = math.radians(32), math.radians(22)

    def project(v):
        x, y, z, w = (2 * c - 1 for c in v)
        s = 1.0 if w > 0 else inner_scale
        x, y, z = x * s, y * s, z * s
        x1 = x * math.cos(yaw) - y * math.sin(yaw)
        y1 = x * math.sin(yaw) + y * math.cos(yaw)
        z1 = z * math.cos(pitch) - y1 * math.sin(pitch)
        return (x1, -z1)

    verts = list(itertools.product((0, 1), repeat=4))
    lines = []
    for a, b in itertools.combinations(verts, 2):
        diff = [i for i in range(4) if a[i] != b[i]]
        if len(diff) != 1:
            continue
        group = "link" if diff[0] == 3 else ("edge" if a[3] == 1 else "edge2")
        lines.append((group, project(a), project(b)))
    return lines


def mark_group(x0, y0, size, weight, colors, use_classes=False):
    """SVG <g> drawing the mark so its outer stroke edges fill the square box."""
    w = WEIGHTS[weight]
    lines = tesseract_lines(w["inner"])
    pts = [p for _, p, q in lines for p in (p, q)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    unit = size / 120.0
    pad = w["edge"] * unit / 2  # keep round caps inside the box
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    k = (size - 2 * pad) / span
    ox = x0 + (size - (max(xs) - min(xs)) * k) / 2 - min(xs) * k
    oy = y0 + (size - (max(ys) - min(ys)) * k) / 2 - min(ys) * k
    out = []
    for group in ("link", "edge2", "edge"):
        paint = f'class="{group}"' if use_classes else f'stroke="{colors[group]}"'
        segs = " ".join(
            f"M{p[0] * k + ox:.2f} {p[1] * k + oy:.2f}L{q[0] * k + ox:.2f} {q[1] * k + oy:.2f}"
            for g, p, q in lines if g == group)
        out.append(f'<path {paint} stroke-width="{w[group] * unit:.2f}" d="{segs}"/>')
    return ('<g fill="none" stroke-linecap="round" stroke-linejoin="round">'
            + "".join(out) + "</g>")


def svg(view_w, view_h, body, title, style=""):
    style_block = f"<style>{style}</style>" if style else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.2f} {view_h:.2f}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>{style_block}{body}</svg>\n')


AUTO_STYLE = (
    ".edge{{stroke:{l[edge]}}}.edge2{{stroke:{l[edge2]}}}.link{{stroke:{l[link]}}}"
    "@media (prefers-color-scheme:dark){{.edge{{stroke:{d[edge]}}}"
    ".edge2{{stroke:{d[edge2]}}}.link{{stroke:{d[link]}}}}}"
).format(l=PALETTE["light"], d=PALETTE["dark"])


def wordmark_paths(font_path, text, size, x, baseline, tracking_em=-0.03):
    """Outline `text` and return (path elements per glyph, end x)."""
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    scale = size / upm
    out = []
    pen_x = x
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, pen_x, baseline)))
        out.append((ch, pen.getCommands()))
        pen_x += glyphs[name].width * scale + tracking_em * size
    return out, pen_x - tracking_em * size


def main():
    font_path, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name, text):
        (outdir / name).write_text(text)
        written.append(name)

    # Mark: transparent background, 120x120 with breathing room.
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "-dark"
        write(f"tensor4all-mark{suffix}.svg",
              svg(120, 120, mark_group(8, 8, 104, "regular", PALETTE[theme]), "tensor4all"))
    write("tensor4all-mark-auto.svg",
          svg(120, 120, mark_group(8, 8, 104, "regular", None, use_classes=True), "tensor4all", AUTO_STYLE))
    # Favicon: heavier strokes, nearly edge to edge, follows the OS theme.
    write("favicon.svg",
          svg(120, 120, mark_group(4, 4, 112, "small", None, use_classes=True), "tensor4all", AUTO_STYLE))

    # Horizontal wordmark: mark + outlined "tensor4all", the 4 in the warm accent.
    mark_size, gap, font_size = 104, 22, 70
    baseline = 60 + 0.516 * font_size / 2  # center the x-height on the mark
    for theme in ("light", "dark"):
        c = PALETTE[theme]
        glyphs, end_x = wordmark_paths(font_path, "tensor4all", font_size, 8 + mark_size + gap, baseline)
        text = "".join(
            f'<path fill="{c["four"] if ch == "4" else c["ink"]}" d="{d}"/>' for ch, d in glyphs)
        width = end_x + 8
        suffix = "" if theme == "light" else "-dark"
        write(f"tensor4all-wordmark{suffix}.svg",
              svg(width, 120, mark_group(8, 8, mark_size, "regular", c) + text, "tensor4all"))

    # Square avatar with brand background (GitHub org, social cards). Circle-crop safe.
    for theme in ("light", "dark"):
        c = PALETTE[theme]
        suffix = "" if theme == "light" else "-dark"
        body = f'<rect width="120" height="120" fill="{c["bg"]}"/>' + mark_group(26, 26, 68, "regular", c)
        write(f"tensor4all-avatar{suffix}.svg", svg(120, 120, body, "tensor4all"))

    # Opaque small icon for PNG favicons / apple-touch-icon.
    c = PALETTE["light"]
    write("tensor4all-icon-small.svg",
          svg(120, 120, f'<rect width="120" height="120" rx="0" fill="{c["bg"]}"/>'
              + mark_group(10, 10, 100, "small", c), "tensor4all"))
    print("\n".join(written))


if __name__ == "__main__":
    main()
