"""Validate and preview the hand-authored figures in docs/figures/ (CPU only, no downloads, no installs).

    /usr/bin/python3 -I scripts/figure_check.py [--scale 1 2] [--out .cache/figcheck]

* SVG: parses the XML, checks that every url(#id) / href="#id" reference resolves, then renders PNG previews with the
  system librsvg + cairo (python3-gi). The project venv has no gi, so run this with the system python3 and
  `source scripts/env.sh` first (it points fontconfig's cache at ./.cache). Previews are for checking only and are
  written under .cache/, never into docs/.
* TikZ: pdflatex into the output folder (TeX caches under .cache/), fails on font warnings or Type 3 fonts, PNG preview.
* DOT: a structural check (balanced braces/quotes, every edge endpoint declared). Rendering needs Graphviz `dot`,
  which is not installed on this machine; the check says so instead of failing.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
REF = re.compile(r"url\(#([^)]+)\)")


def check_svg(path: Path) -> list[str]:
    errs: list[str] = []
    tree = ET.parse(path)  # raises on malformed XML
    ids = {el.get("id") for el in tree.iter() if el.get("id")}
    refs = set()
    for el in tree.iter():
        for k, v in el.attrib.items():
            refs.update(REF.findall(v))
            if k.endswith("href") and v.startswith("#"):
                refs.add(v[1:])
        if el.tag.endswith("style") and el.text:
            refs.update(REF.findall(el.text))
    missing = sorted(refs - ids)
    if missing:
        errs.append(f"unresolved references: {missing}")
    root = tree.getroot()
    w, h = root.get("width"), root.get("height")
    n_text = sum(1 for el in tree.iter() if el.tag.endswith("}text"))
    print(f"[svg] {path.relative_to(ROOT)}: well-formed, {w}x{h}, {len(ids)} ids, {len(refs)} refs, {n_text} text elements")
    return errs


def render_svg(path: Path, out: Path, scales: list[float]) -> None:
    try:
        import gi

        gi.require_version("Rsvg", "2.0")
        import cairo
        from gi.repository import Rsvg
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[svg] render skipped: librsvg/cairo bindings unavailable ({e})")
        return
    out.mkdir(parents=True, exist_ok=True)
    handle = Rsvg.Handle.new_from_file(str(path))
    dim = handle.get_dimensions()
    for s in scales:
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(dim.width * s), int(dim.height * s))
        ctx = cairo.Context(surf)
        ctx.scale(s, s)
        handle.render_cairo(ctx)
        dst = out / f"{path.stem}@{s:g}x.png"
        surf.write_to_png(str(dst))
        print(f"[svg] rendered {dst.relative_to(ROOT)} ({int(dim.width * s)}x{int(dim.height * s)}, librsvg {Rsvg.MAJOR_VERSION}.{Rsvg.MINOR_VERSION})")


def check_dot(path: Path, out: Path) -> list[str]:
    errs: list[str] = []
    src = path.read_text(encoding="utf-8")
    body = re.sub(r'"(?:[^"\\]|\\.)*"', '""', src)          # blank out quoted strings
    body = re.sub(r"<(?:[^<>]|<[^<>]*>)*>", "<>", body)      # and HTML-like labels (one nesting level)
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    if src.count('"') % 2:
        errs.append("odd number of double quotes")
    if body.count("{") != body.count("}"):
        errs.append(f"unbalanced braces: {body.count('{')} open vs {body.count('}')} close")
    nodes = set(re.findall(r"^\s*([A-Za-z_][\w]*)\s*\[", body, flags=re.M))
    edges = re.findall(r"([A-Za-z_][\w]*)(?::\w+)?\s*->\s*([A-Za-z_][\w]*)", body)
    undeclared = sorted({n for e in edges for n in e} - nodes - {"node", "edge", "graph"})
    if undeclared:
        errs.append(f"edge endpoints without a node statement: {undeclared}")
    print(f"[dot] {path.relative_to(ROOT)}: {len(nodes - {'node', 'edge', 'graph'})} nodes, {len(edges)} edges, braces balanced={body.count('{') == body.count('}')}")
    dot = shutil.which("dot")
    if dot:
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{path.stem}.dot.svg"
        subprocess.run([dot, "-Tsvg", str(path), "-o", str(dst)], check=True)
        print(f"[dot] rendered {dst.relative_to(ROOT)}")
    else:
        print("[dot] render skipped: Graphviz `dot` is not installed (structural check only)")
    return errs


def build_tikz(path: Path, out: Path) -> list[str]:
    """pdflatex into `out` (TeX caches under .cache/), require Type 1 fonts only, then a 150-dpi PNG preview."""
    import os

    if not shutil.which("pdflatex"):
        print("[tex] build skipped: pdflatex not installed")
        return []
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEXMFVAR=str(ROOT / ".cache" / "texmf-var"), TEXMFCONFIG=str(ROOT / ".cache" / "texmf-config"))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", path.name],
                       cwd=path.parent, env=env, capture_output=True, text=True)
    pdf = out / f"{path.stem}.pdf"
    if r.returncode != 0 or not pdf.exists():
        return [f"pdflatex failed on {path.name}; see {out / (path.stem + '.log')}"]
    log = (out / f"{path.stem}.log").read_text(errors="replace")
    warns = len(re.findall(r"Font Warning|Missing character", log))
    errs = [f"{warns} font warnings in {path.stem}.log"] if warns else []
    if shutil.which("pdffonts"):
        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
        if "Type 3" in fonts:
            errs.append(f"{pdf.name} embeds a Type 3 (bitmap) font")
    print(f"[tex] {path.relative_to(ROOT)} -> {pdf.relative_to(ROOT)} (font warnings: {warns})")
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf), str(out / f"{path.stem}.tikz")], check=True)
        print(f"[tex] rendered {(out / (path.stem + '.tikz.png')).relative_to(ROOT)}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--out", type=Path, default=ROOT / ".cache" / "figcheck")
    ap.add_argument("--no-tex", action="store_true", help="skip the TikZ build")
    a = ap.parse_args()
    errs: list[str] = []
    for svg in sorted(FIG.glob("*.svg")):
        e = check_svg(svg)
        errs += e
        if not e:
            render_svg(svg, a.out, a.scale)
    for dot in sorted(FIG.glob("*.dot")):
        errs += check_dot(dot, a.out)
    if not a.no_tex:
        for tex in sorted(FIG.glob("*.tex")):
            errs += build_tikz(tex, a.out)
    for e in errs:
        print("ERROR:", e)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
