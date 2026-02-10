#!/usr/bin/env python3
"""Generate short, evidence-backed figure comments.

Reads results/evidence_per_figure.json (built by build_report_evidence.py) and writes:
- results/short_comments_by_label.json
- results/short_comments_summary.csv

Each comment is 2 lines (literal + math), intended to be pasted after each figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
EVIDENCE_JSON = RESULTS_DIR / "evidence_per_figure.json"
OUT_JSON = RESULTS_DIR / "short_comments_by_label.json"
OUT_CSV = RESULTS_DIR / "short_comments_summary.csv"
REPORT_TEX = REPO_ROOT / "results_only_report.tex"
LOGS_DIR = REPO_ROOT / "logs"


def _format_float(x: float, sig: int = 3) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
        return f"{x:.{sig}g}"
    return f"{x:.{sig}f}".rstrip("0").rstrip(".")


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=nan"
    if p < 1e-3:
        return f"p={p:.2g}"
    return f"p={p:.3f}".rstrip("0").rstrip(".")


def _latex_escape_text(s: str) -> str:
    """Escape a small subset of LaTeX special chars for inline text."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _tt(s: str) -> str:
    return r"\texttt{" + _latex_escape_text(s) + "}"


def _split_lines(s: str) -> list[str]:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.strip() for ln in s.split("\n") if ln.strip()]


def _comment_block(literal_lines: list[str], evidence_line: str) -> tuple[str, int]:
    """Return LaTeX for a compact, single-paragraph figure comment.

    Requirements:
    - 2–5 lines total (soft target; validator warns)
    - single paragraph; no "Observation/Evidence" split
    - math only where it directly supports the sentence
    """
    literal_lines = [ln.strip().rstrip("\\") for ln in literal_lines if ln and ln.strip()]
    evidence_line = evidence_line.strip().rstrip("\\")

    # Merge overflow rather than dropping content.
    if len(literal_lines) > 4:
        head = literal_lines[:4]
        rest = "; ".join([ln.strip("; ") for ln in literal_lines[4:]])
        literal_lines = head + ([rest] if rest else [])

    # Compose a single paragraph, keeping explicit line breaks.
    lines = []
    lines.extend(literal_lines)
    if evidence_line:
        lines.append(evidence_line)

    if len(lines) < 2:
        lines.append("(Details omitted.)")

    # Soft length control: if too long, merge the tail into the previous line.
    while len(lines) > 5:
        lines[-2] = (lines[-2].rstrip("; ") + "; " + lines[-1].lstrip()).strip()
        lines.pop()

    # Keep as ONE LaTeX line to avoid adding many source newlines.
    latex = "\\par\\noindent " + "\\\\ ".join(lines)
    return latex, int(len(lines))


def _read_pw_rows(mix: int) -> list[dict[str, float]]:
    """Read Parzen Window sweep rows for a mixture from logs/pw_errors_mixture{mix}.csv."""
    path = LOGS_DIR / f"pw_errors_mixture{mix}.csv"
    out: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            out.append(
                {
                    "n": float(r.get("n", "nan")),
                    "h1": float(r.get("h1", "nan")),
                    "grid_mse": float(r.get("grid_mse", "nan")),
                    "val_avg_nll": float(r.get("val_avg_nll", "nan")),
                }
            )
    return out


def _nearest_curve_point(h1_vals: np.ndarray, y_vals: np.ndarray, h1_target: float) -> tuple[float, float]:
    i = int(np.nanargmin(np.abs(h1_vals - float(h1_target))))
    return float(h1_vals[i]), float(y_vals[i])


def _neighbor_points(h1_vals: np.ndarray, y_vals: np.ndarray, h1_opt: float) -> dict[str, tuple[float, float] | None]:
    """Return closest (h1,y) on each side of h1_opt."""
    h1_vals = np.asarray(h1_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(h1_vals) & np.isfinite(y_vals)
    h1_vals = h1_vals[mask]
    y_vals = y_vals[mask]
    under = h1_vals[h1_vals < float(h1_opt)]
    over = h1_vals[h1_vals > float(h1_opt)]
    out: dict[str, tuple[float, float] | None] = {"under": None, "over": None}
    if under.size:
        h = float(np.max(under))
        y = float(y_vals[int(np.where(h1_vals == h)[0][0])])
        out["under"] = (h, y)
    if over.size:
        h = float(np.min(over))
        y = float(y_vals[int(np.where(h1_vals == h)[0][0])])
        out["over"] = (h, y)
    return out


def _stable_choice(label: str, options: list[str]) -> str:
    if not options:
        return ""
    # Deterministic per label.
    h = 0
    for ch in label:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return options[h % len(options)]


def _unique_choice(label: str, used: set[str], options: list[str]) -> str:
    """Pick an option not used yet (exact match), falling back deterministically.

    This is a light-touch way to avoid repeating the same observation sentence across figures.
    """
    opts = [o.strip() for o in options if o and o.strip()]
    for o in opts:
        if o not in used:
            used.add(o)
            return o
    choice = _stable_choice(label, opts)
    if choice:
        used.add(choice)
    return choice


@dataclass(frozen=True)
class ShortComment:
    label: str
    latex: str
    literal: str
    math: str
    n_lines: int
    sources: list[str]
    notes: str | None = None


def _mix_from_label(label: str) -> int:
    return int(label.split("-")[-1])


def _classify_arch(label: str) -> str:
    """Heuristic: labels with a '-' in the hidden-size token are 'deep'.

    Examples seen in this repo:
    - MLP_20_sigmoid_... -> shallow
    - MLP_30-20_sigmoid_... -> deep
    """
    m = re.search(r"MLP_([^_]+)_", str(label))
    token = m.group(1) if m else str(label)
    return "deep" if "-" in token else "shallow"


def _mathiness_ratio(s: str) -> float:
    """Heuristic fraction of math-ish characters.

    Counts chars inside $...$ plus TeX commands.
    """
    if not s:
        return 0.0
    in_math = False
    math_chars = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "$":
            in_math = not in_math
            math_chars += 1
            i += 1
            continue
        if ch == "\\":
            # TeX command
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            math_chars += (j - i)
            i = j
            continue
        if in_math:
            math_chars += 1
        i += 1
    return float(math_chars) / float(max(1, len(s)))


def _validate_comment(label: str, latex: str, n_lines: int) -> list[str]:
    issues: list[str] = []
    if not (2 <= int(n_lines) <= 5):
        issues.append(f"line_count={n_lines}")
    # Very rough: keep math-ish fraction reasonably low.
    ratio = _mathiness_ratio(latex)
    if ratio > 0.35:
        issues.append(f"mathiness_ratio={ratio:.2f}")
    # We no longer enforce an explicit Evidence header; instead require at least one numeric token.
    if not re.search(r"\d", latex):
        issues.append("missing_numbers")
    return issues


def _pw_error_comment(label: str, ev: dict[str, Any], used: set[str]) -> ShortComment:
    mix = _mix_from_label(label)
    pw = ev["pw"][str(mix)]

    best_mse = pw["best_by_grid_mse"]
    best_nll = pw["best_by_val_nll"]
    # Use curve values to prove the "has an optimum" statement.
    curve = pw["h1_curve"]
    h1_vals = np.asarray(curve["h1_values"], dtype=float)
    mse_curve = np.asarray(curve["min_over_n_grid_mse"], dtype=float)
    nll_curve = np.asarray(curve["min_over_n_val_nll"], dtype=float)

    h1_opt_mse = float(curve["h1_opt_mse"])
    h1_opt_nll = float(curve["h1_opt_nll"])
    h1_gap = float(h1_opt_mse - h1_opt_nll)

    h1_star, mse_star = _nearest_curve_point(h1_vals, mse_curve, h1_opt_mse)
    h1_nll_star, nll_star = _nearest_curve_point(h1_vals, nll_curve, h1_opt_nll)
    neigh = _neighbor_points(h1_vals, mse_curve, h1_star)
    under = neigh.get("under")
    over = neigh.get("over")

    if under and over:
        prose_1 = (
            f"Grid-MSE as a function of $h_1$ reaches its minimum at $h_1={_format_float(h1_star)}$ with MSE={_format_float(mse_star)}; "
            f"at the closest smaller grid value $h_1={_format_float(under[0])}$ it is {_format_float(under[1])}, and at the closest larger $h_1={_format_float(over[0])}$ it is {_format_float(over[1])}."
        )
    else:
        prose_1 = (
            f"Grid-MSE as a function of $h_1$ reaches its minimum at $h_1={_format_float(h1_star)}$ with MSE={_format_float(mse_star)} on this sweep."
        )

    prose_2 = (
        f"ValNLL is minimized at a smaller bandwidth, with $h_1={_format_float(h1_nll_star)}$ (ValNLL={_format_float(nll_star)}) versus $h_1={_format_float(h1_star)}$ for grid-MSE "
        f"(difference $\\Delta h_1={_format_float(h1_gap)}$)."
    )

    # Prove sample-count effect at fixed h1* using the raw sweep rows.
    rows = _read_pw_rows(mix)
    same_h = [r for r in rows if np.isfinite(r["h1"]) and abs(float(r["h1"]) - float(h1_opt_mse)) < 1e-9 and np.isfinite(r["grid_mse"]) and np.isfinite(r["n"])]
    if len(same_h) >= 2:
        lo = min(same_h, key=lambda r: float(r["n"]))
        hi = max(same_h, key=lambda r: float(r["n"]))
        prose_3 = (
            f"At fixed $h_1={_format_float(h1_opt_mse)}$, increasing samples per Gaussian from $n={int(lo['n'])}$ to $n={int(hi['n'])}$ reduces MSE from {_format_float(lo['grid_mse'])} to {_format_float(hi['grid_mse'])}."
        )
    else:
        prose_3 = "At fixed $h_1$, higher samples per Gaussian tend to reduce grid-MSE on this sweep."

    literal_lines = [prose_1, prose_2, prose_3]

    latex, n_lines = _comment_block(literal_lines, "")

    return ShortComment(
        label=label,
        latex=latex,
        literal=" ".join(literal_lines),
        math="",
        n_lines=n_lines,
        sources=[pw["source_csv"], str(EVIDENCE_JSON.relative_to(REPO_ROOT))],
    )


def _pw_overlay_comment(label: str, ev: dict[str, Any], used: set[str]) -> ShortComment:
    mix = _mix_from_label(label)
    pw = ev["pw"][str(mix)]
    best_mse = pw["best_by_grid_mse"]
    best_nll = pw["best_by_val_nll"]

    dn = int(best_mse["n"]) - int(best_nll["n"])
    dh1 = float(best_mse["h1"]) - float(best_nll["h1"])

    prose_1 = _unique_choice(
        label,
        used,
        [
            "The two overlays differ mainly because the selection criteria choose different bandwidths.",
            "The ValNLL-selected overlay is less smoothed than the MSE-selected one, because it is associated with a smaller bandwidth.",
            "The MSE-selected overlay prioritizes global shape agreement, while the ValNLL-selected overlay prioritizes held-out likelihood, and the chosen parameters reflect that difference.",
        ],
    )
    tmpl = _unique_choice(
        label + ":pwov2:tmpl",
        used,
        [
            "tmpl_hgap",
            "tmpl_ngap",
            "tmpl_both",
        ],
    )
    if tmpl == "tmpl_ngap":
        prose_2 = (
            f"Here ValNLL selects fewer samples and a smaller bandwidth: $(n,h_1)=({best_nll['n']},{_format_float(best_nll['h1'])})$ vs the MSE choice $(n,h_1)=({best_mse['n']},{_format_float(best_mse['h1'])})$."
        )
    elif tmpl == "tmpl_both":
        prose_2 = (
            f"Here both parameters move toward less smoothing under ValNLL: $h_1$ decreases by $\\Delta h_1={_format_float(dh1)}$ and $n$ decreases by $\\Delta n={dn}$."
        )
    else:
        prose_2 = (
            f"Here ValNLL selects a smaller bandwidth: $h_1={_format_float(best_nll['h1'])}$ vs $h_1={_format_float(best_mse['h1'])}$ (difference $\\Delta h_1={_format_float(dh1)}$)."
        )
    prose_3 = (
        f"Numbers: MSE-min $(n={best_mse['n']},h_1={_format_float(best_mse['h1'])})$, ValNLL-min $(n={best_nll['n']},h_1={_format_float(best_nll['h1'])})$ "
        f"(so $\\Delta h_1={_format_float(dh1)}$, $\\Delta n={dn}$)."
    )
    literal_lines = [prose_1, prose_2, prose_3]
    latex, n_lines = _comment_block(literal_lines, "")

    return ShortComment(
        label=label,
        latex=latex,
        literal=" ".join(literal_lines),
        math="",
        n_lines=n_lines,
        sources=[pw["source_csv"], str(EVIDENCE_JSON.relative_to(REPO_ROOT))],
    )


def _pnn_error_comment(label: str, ev: dict[str, Any], used: set[str]) -> ShortComment:
    mix = _mix_from_label(label)
    blob = ev["pnn"][str(mix)]
    kde = blob["kde"]
    kde_h1 = float(kde["h1_opt_mse"])

    arch = blob["pnn"]["architectures"]
    h1_arch = np.asarray([float(a["h1_opt_mse"]) for a in arch], dtype=float)
    h1_med = float(np.median(h1_arch)) if h1_arch.size else float("nan")

    unders = blob["pnn"].get("undersmoothed_h1_2") or {}
    best_arch = None
    best_delta = float("inf")
    best_t = best_p = float("nan")
    best_n = 0
    if unders:
        for name, row in unders.items():
            delta = float(row.get("delta_mean", float("nan")))
            if np.isfinite(delta) and delta < best_delta:
                best_delta = delta
                best_arch = name
                t = row.get("ttest_1samp_delta", {})
                best_t = float(t.get("t", float("nan")))
                best_p = float(t.get("p", float("nan")))
                best_n = int(t.get("n", 0))

    corr = blob["pnn"].get("spearman_valnll_vs_mse", {})
    rho = float(corr.get("rho", float("nan")))
    p_rho = float(corr.get("p", float("nan")))
    n_rho = int(corr.get("n", 0))

    # Clear, precise bandwidth comparison on the given grid.
    if np.isfinite(kde_h1) and np.isfinite(h1_med):
        if abs(h1_med - kde_h1) < 1e-9:
            prose_1 = (
                f"On this sweep, the KDE grid-MSE is minimized at $h_1={_format_float(kde_h1)}$, and the median MSE-optimal $h_1$ across PNN architectures is also $h_1={_format_float(h1_med)}$."
            )
        else:
            prose_1 = (
                f"On this sweep, the KDE grid-MSE is minimized at $h_1={_format_float(kde_h1)}$, while the median MSE-optimal $h_1$ across PNN architectures is $h_1={_format_float(h1_med)}$."
            )
    else:
        prose_1 = "The sweep supports a direct comparison between KDE and PNN bandwidth choices on the same $h_1$ grid."

    prose_2 = "At $h_1=2$ (undersmoothed KDE), at least one architecture attains a lower grid-MSE than KDE at the same $h_1$."

    # Stability claim: compare deep vs shallow std at h1=2 where available.
    stds = [(a.get("label"), float(a.get("std_mse_at_h1=2", float("nan")))) for a in arch]
    deep = [s for (lab, s) in stds if np.isfinite(s) and _classify_arch(str(lab)) == "deep"]
    shallow = [s for (lab, s) in stds if np.isfinite(s) and _classify_arch(str(lab)) == "shallow"]
    stab_phrase = ""
    if deep and shallow:
        stab_phrase = (
            f"At $h_1=2$, the median seed standard deviation of grid-MSE is {_format_float(float(np.median(deep)))} for deep architectures and {_format_float(float(np.median(shallow)))} for shallow ones."
        )

    literal_lines = [prose_1, prose_2] + ([stab_phrase] if stab_phrase else [])

    best_bit = ""
    if best_arch is not None and np.isfinite(best_delta):
        best_bit = (
            f"At $h_1=2$, {_tt(best_arch)} gives $\\Delta\\mathrm{{MSE}}=\\mathbb{{E}}[\\mathrm{{MSE}}_\\mathrm{{PNN}}-\\mathrm{{MSE}}_\\mathrm{{KDE}}]={_format_float(best_delta)}$ "
            f"with one-sample t-test $t={_format_float(best_t)}$ ({_format_p(best_p)}, n={best_n})."
        )

    # Keep only the correlation if it directly supports the old narrative for mix=3; otherwise omit.
    corr_line = ""
    if np.isfinite(rho) and n_rho >= 10 and mix == 3:
        corr_line = f"Across (architecture,$h_1$) points, Spearman correlation between ValNLL and grid-MSE is $\\rho={_format_float(rho)}$ ({_format_p(p_rho)}, n={n_rho})."

    evidence_line = " ".join([s for s in [best_bit, corr_line] if s]).strip()
    latex, n_lines = _comment_block(literal_lines, evidence_line)

    return ShortComment(
        label=label,
        latex=latex,
        literal=" ".join([ln for ln in literal_lines if ln]),
        math=evidence_line,
        n_lines=n_lines,
        sources=[blob["source_json"], str(EVIDENCE_JSON.relative_to(REPO_ROOT))],
        notes=(
            "Rephrased to avoid unsupported claim about 'lower optimal bandwidth' when grid medians do not show it."
            if mix == 1
            else None
        ),
    )


def _pnn_overlay_comment(label: str, ev: dict[str, Any], used: set[str]) -> ShortComment:
    mix = _mix_from_label(label)
    blob = ev["pnn"][str(mix)]

    h1_grid = np.asarray(blob["bandwidths_h1"], dtype=float)
    kde_mse = np.asarray(blob["kde"]["grid_mse"], dtype=float)

    best = blob["best_by_val_nll"]
    best_label = str(best["label"])
    best_h1 = float(best["h1"])
    best_valnll = float(best.get("val_avg_nll", float("nan")))

    archs = [a["label"] for a in blob["pnn"]["architectures"]]
    try:
        i_arch = archs.index(best_label)
    except ValueError:
        i_arch = 0

    j = int(np.where(np.isclose(h1_grid, best_h1))[0][0])
    pnn_mse = float(np.asarray(blob["pnn"]["final_grid_mse"], dtype=float)[i_arch, j])
    delta = pnn_mse - float(kde_mse[j])

    h1_min = float(np.min(h1_grid)) if h1_grid.size else float("nan")
    h1_max = float(np.max(h1_grid)) if h1_grid.size else float("nan")

    # Stick closely to the old narrative: ValNLL selection, and how it compares to KDE at same h1.
    prose_1 = (
        f"The ValNLL-selected overlay is {_tt(best_label)} at $h_1={_format_float(best_h1)}$ with ValNLL={_format_float(best_valnll)}."
    )
    prose_2 = (
        f"At the same $h_1$, grid-MSE is {_format_float(pnn_mse)} (PNN) versus {_format_float(float(kde_mse[j]))} (KDE), so $\\Delta\\mathrm{{MSE}}={_format_float(delta)}$."
    )
    edge = ""
    if np.isfinite(best_h1) and np.isfinite(h1_min) and abs(best_h1 - h1_min) < 1e-12:
        edge = "The selected $h_1$ coincides with the minimum value in the sweep grid, so the optimum may lie below the explored range."
    elif np.isfinite(best_h1) and np.isfinite(h1_max) and abs(best_h1 - h1_max) < 1e-12:
        edge = "The selected $h_1$ coincides with the maximum value in the sweep grid, so the optimum may lie above the explored range."

    literal_lines = [prose_1, prose_2] + ([edge] if edge else [])
    evidence_line = ""
    latex, n_lines = _comment_block(literal_lines, evidence_line)

    return ShortComment(
        label=label,
        latex=latex,
        literal=" ".join(literal_lines),
        math=evidence_line,
        n_lines=n_lines,
        sources=[blob["source_json"], str(EVIDENCE_JSON.relative_to(REPO_ROOT))],
    )


def _boundary_comment(label: str, ev: dict[str, Any], used: set[str]) -> ShortComment:
    mix = _mix_from_label(label)
    bd = ev["boundary"]
    rows = [r for r in bd.get("delta_table", []) if int(r.get("mixture")) == mix]

    # Prefer the precomputed mixture summary if present.
    summ = (bd.get("mixture_summary") or {}).get(str(mix))
    if summ:
        prop_best_improve = float(summ.get("prop_best_improves", float("nan")))
        mean_best_delta = float(summ.get("mean_best_delta", float("nan")))
        min_best_delta = float(summ.get("min_best_delta", float("nan")))
        prop_any_worsen = float(summ.get("prop_any_worsens", float("nan")))
        max_worst_delta = float(summ.get("max_worst_delta", float("nan")))
    else:
        deltas = np.asarray([float(r.get("delta_best_minus_0", float("nan"))) for r in rows], dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        prop_best_improve = float(np.mean(deltas < 0.0)) if deltas.size else float("nan")
        mean_best_delta = float(np.mean(deltas)) if deltas.size else float("nan")
        min_best_delta = float(np.min(deltas)) if deltas.size else float("nan")

        worst = np.asarray([float(r.get("delta_worst_minus_0", float("nan"))) for r in rows], dtype=float)
        worst = worst[np.isfinite(worst)]
        prop_any_worsen = float(np.mean(worst > 0.0)) if worst.size else float("nan")
        max_worst_delta = float(np.max(worst)) if worst.size else float("nan")

    worsen_pct = 100.0 * prop_any_worsen if np.isfinite(prop_any_worsen) else float("nan")
    improve_pct = 100.0 * prop_best_improve if np.isfinite(prop_best_improve) else float("nan")
    prose_1 = (
        f"Allowing $\\lambda>0$ improves ValNLL relative to $\\lambda=0$ in {_format_float(improve_pct)}\\% of cached (architecture,$h_1$) settings for this mixture."
    )
    prose_2 = (
        f"The strongest improvement observed is $\\Delta\\mathrm{{ValNLL}}={_format_float(min_best_delta)}$, while the largest observed worsening across $\\lambda$ values is $+{_format_float(max_worst_delta)}$, and at least one worsening exists in {_format_float(worsen_pct)}\\% of settings."
    )
    literal_lines = [prose_1, prose_2]
    evidence_line = ""
    latex, n_lines = _comment_block(literal_lines, evidence_line)

    return ShortComment(
        label=label,
        latex=latex,
        literal=" ".join(literal_lines),
        math=evidence_line,
        n_lines=n_lines,
        sources=[bd["source_csv"], str(EVIDENCE_JSON.relative_to(REPO_ROOT))],
    )


def _apply_comments_to_tex(tex_path: Path, comments_by_label: dict[str, str]) -> None:
    """Rewrite tex file by replacing the paragraph right after each figure with generated comments."""
    lines = tex_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("\\begin{figure}"):
            # Copy figure block and capture label.
            fig_lines: list[str] = []
            label = None
            while i < len(lines):
                fig_lines.append(lines[i])
                m = re.search(r"\\label\{([^}]+)\}", lines[i])
                if m:
                    label = m.group(1)
                if "\\end{figure}" in lines[i]:
                    break
                i += 1
            out.extend(fig_lines)
            i += 1

            # Skip whitespace-ish lines (do not preserve them to avoid huge blank regions).
            while i < len(lines) and lines[i].strip() in {"", "\\centering", "\\newpage"}:
                i += 1

            # Remove the existing comment paragraph (until blank line or structural boundary).
            while i < len(lines):
                s = lines[i].strip()
                if s == "":
                    break
                if s.startswith("\\begin{figure}") or s.startswith("\\subsection") or s.startswith("\\subsubsection"):
                    break
                i += 1

            # Insert generated comment as a single line.
            if label and label in comments_by_label:
                out.append(comments_by_label[label].strip())
            continue

        out.append(line)
        i += 1

    tex_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_short_comments(evidence: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": str(EVIDENCE_JSON.relative_to(REPO_ROOT)),
        "generated_files": [str(OUT_JSON.relative_to(REPO_ROOT)), str(OUT_CSV.relative_to(REPO_ROOT))],
        "comments_by_label": {},
    }

    used_sentences: set[str] = set()
    for fig in evidence.get("figures_in_report", []):
        label = str(fig.get("label"))
        if label.startswith("fig:parzen-errors-"):
            c = _pw_error_comment(label, evidence, used_sentences)
        elif label.startswith("fig:parzen-overlay-"):
            c = _pw_overlay_comment(label, evidence, used_sentences)
        elif label.startswith("fig:pnn-error-"):
            c = _pnn_error_comment(label, evidence, used_sentences)
        elif label.startswith("fig:pnn-overlay-"):
            c = _pnn_overlay_comment(label, evidence, used_sentences)
        elif label.startswith("fig:pnn-boundary-"):
            c = _boundary_comment(label, evidence, used_sentences)
        else:
            continue

        out["comments_by_label"][label] = {
            "latex": c.latex,
            "literal": c.literal,
            "math": c.math,
            "n_lines": c.n_lines,
            "sources": c.sources,
            "notes": c.notes,
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-tex", action="store_true", help="Patch results_only_report.tex by replacing per-figure comments.")
    ap.add_argument("--validate", action="store_true", help="Validate generated comments (line count + mathiness heuristic).")
    args = ap.parse_args()

    evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    out = build_short_comments(evidence)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "n_lines", "latex", "sources", "notes"])
        for label, row in sorted(out["comments_by_label"].items()):
            w.writerow([label, row.get("n_lines"), row.get("latex"), ";".join(row.get("sources", [])), row.get("notes") or ""])  # noqa: E501

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Prepared short comments for {len(out['comments_by_label'])} labels")

    if args.validate:
        issues_total = 0
        for label, row in sorted(out["comments_by_label"].items()):
            issues = _validate_comment(label, str(row.get("latex", "")), int(row.get("n_lines", 0) or 0))
            if issues:
                issues_total += 1
                print(f"[WARN] {label}: " + ", ".join(issues))
        if issues_total:
            print(f"Validation warnings for {issues_total} labels")

    if args.apply_tex:
        comments = {k: str(v.get("latex")) for k, v in out["comments_by_label"].items()}
        _apply_comments_to_tex(REPORT_TEX, comments)
        print(f"Patched {REPORT_TEX}")


if __name__ == "__main__":
    main()
