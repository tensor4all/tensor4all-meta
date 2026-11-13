# Logos

Logos of tensor4all and its libraries. All are proposals. They are shown, with download links, at <https://tensor4all.org/tensor4all-meta/logos/>.

| Folder | Project |
| --- | --- |
| [`logo/`](logo/README.md) | tensor4all, the organization |
| [`tenferro/`](tenferro/README.md) | tenferro-rs, the tensor runtime |
| [`colmat/`](colmat/README.md) | colmat, the faer fork formerly published as `t4a-faer` |
| [`threadteam/`](threadteam/README.md) | threadteam-rs, the one owner of CPU threads |

## One family

Every logo uses the same type and palette, so they read as one family:

- **Type:** IBM Plex Sans SemiBold, SIL Open Font License 1.1, tracking -0.03 em, kerning on, converted to outlines.
- **Palette:** the tensor4all.org site palette. Ink `#1d2126` and copper `#b7510b` on light backgrounds; `#e6e8eb` and `#f0a46b` on dark ones. colmat also keeps its original navy and amber as the `-fresh` variant.
- **Size:** in a horizontal wordmark the x-height is 0.47 of the mark height, and the gap between mark and text is 0.2 of it. The tensor4all mark is itself lettering, so its text sits on the mark's baseline with an x-height of 0.58 of the T∀ cap height.

## Regenerating

[`tools/export_logos.py`](tools/export_logos.py) writes every SVG and, when `rsvg-convert` is on `PATH`, every PNG. It needs `fontTools`, `uharfbuzz` and a copy of IBM Plex Sans SemiBold.

```bash
python3 -m pip install fonttools uharfbuzz
python3 branding/tools/export_logos.py path/to/IBMPlexSans-SemiBold.ttf branding
```

Geometry, patterns and colors are constants at the top of each section of the script. Change them there rather than editing the SVGs by hand.
