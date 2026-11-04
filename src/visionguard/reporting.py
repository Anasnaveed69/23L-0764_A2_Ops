"""Write small LaTeX tables / macros that the report \\input{}s, so numbers in the PDF always
match what the scripts actually computed (no hand-copying)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT

GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures" / "auto"


def tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def write_table(name: str, header: list[str], rows: list[list], colspec: str | None = None,
                caption: str | None = None, label: str | None = None) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    colspec = colspec or ("l" + "r" * (len(header) - 1))
    lines = []
    if caption:
        lines += [r"\begin{table}[H]", r"\centering", r"\small"]
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
              " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines += [rf"\caption{{{caption}}}"]
        if label:
            lines += [rf"\label{{{label}}}"]
        lines += [r"\end{table}"]
    path = GEN / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_macros(name: str, values: dict[str, object]) -> Path:
    """\\newcommand{\\vgKey}{value} for every key, e.g. \\vgBestEpoch."""
    GEN.mkdir(parents=True, exist_ok=True)
    digits = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                      "Seven", "Eight", "Nine"]))
    lines = []
    for k, v in values.items():
        k = "".join(digits.get(ch, ch) for ch in k if ch.isalnum())  # LaTeX names: letters only
        if isinstance(v, float):
            v = f"{v:.3f}"
        lines.append(rf"\providecommand{{\{k}}}{{}}\renewcommand{{\{k}}}{{{tex_escape(v)}}}")
    path = GEN / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_json(name: str, obj) -> Path:
    GEN.mkdir(parents=True, exist_ok=True)
    path = GEN / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def fmt(x, nd=3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
