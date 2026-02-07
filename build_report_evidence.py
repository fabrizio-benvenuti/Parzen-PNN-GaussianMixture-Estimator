#!/usr/bin/env python3
"""Build evidence-backed, LaTeX-ready addenda for results_only_report.tex.

This script reads existing experiment artifacts (CSV/JSON) already in the repo and
produces:
- results/evidence_per_figure.json
- results/evidence_summary.csv

It is intentionally self-contained (stdlib + numpy + scipy).
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
REPORT_TEX = REPO_ROOT / "results_only_report.tex"


@dataclass(frozen=True)
class RegressionSummary:
    slope: float
    intercept: float
    r2: float
    pvalue: float


def _linregress(x: np.ndarray, y: np.ndarray) -> RegressionSummary:
    res = stats.linregress(x, y)
    r2 = float(res.rvalue) ** 2
    return RegressionSummary(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r2=r2,
        pvalue=float(res.pvalue),
    )


def _safe_log(arr: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(arr, eps))


def _format_float(x: float, sig: int = 3) -> str:
    if not np.isfinite(x):
        return "nan"
    # Use scientific for tiny/huge values.
    ax = abs(float(x))
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
        return f"{x:.{sig}g}"
    return f"{x:.{sig}f}".rstrip("0").rstrip(".")


def _format_mean_std(values: Iterable[float], sig: int = 3) -> tuple[str, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ("nan", float("nan"), float("nan"))
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return (f"{_format_float(mu, sig)}\\pm{_format_float(sd, sig)}", mu, sd)


def _cohens_d_one_sample(sample: np.ndarray, mu0: float) -> float:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return float("nan")
    sd = float(np.std(sample, ddof=1))
    if sd == 0.0:
        return float("inf") if float(np.mean(sample)) != mu0 else 0.0
    return (float(np.mean(sample)) - float(mu0)) / sd


def _ttest_vs_constant(sample: np.ndarray, constant: float) -> dict[str, float]:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return {"t": float("nan"), "p": float("nan"), "n": int(sample.size)}
    res = stats.ttest_1samp(sample, popmean=float(constant))
    return {"t": float(res.statistic), "p": float(res.pvalue), "n": int(sample.size)}


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(np.sum(mask))}
    res = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(np.sum(mask))}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_pw_errors(mixture_idx: int) -> dict[str, Any]:
    path = LOGS_DIR / f"pw_errors_mixture{mixture_idx}.csv"
    rows = read_csv_rows(path)

    # Parse numeric columns.
    n = np.array([float(r["n"]) for r in rows], dtype=float)
    h1 = np.array([float(r["h1"]) for r in rows], dtype=float)
    mse = np.array([float(r["grid_mse"]) for r in rows], dtype=float)
    nll = np.array([float(r["val_avg_nll"]) for r in rows], dtype=float)

    # Best configurations.
    i_best_mse = int(np.nanargmin(mse))
    i_best_nll = int(np.nanargmin(nll))

    best_mse = {
        "n": int(round(float(n[i_best_mse]))),
        "h1": float(h1[i_best_mse]),
        "grid_mse": float(mse[i_best_mse]),
        "val_avg_nll": float(nll[i_best_mse]),
    }
    best_nll = {
        "n": int(round(float(n[i_best_nll]))),
        "h1": float(h1[i_best_nll]),
        "grid_mse": float(mse[i_best_nll]),
        "val_avg_nll": float(nll[i_best_nll]),
    }

    # Reduce surface to a 1D curve in h1 by taking the best (min) over n for each h1.
    unique_h1 = np.unique(h1)
    curve_mse = []
    curve_nll = []
    for hv in unique_h1:
        mask = h1 == hv
        curve_mse.append(float(np.min(mse[mask])))
        curve_nll.append(float(np.min(nll[mask])))

    unique_h1 = np.asarray(unique_h1, dtype=float)
    curve_mse = np.asarray(curve_mse, dtype=float)
    curve_nll = np.asarray(curve_nll, dtype=float)

    h1_opt_mse = float(unique_h1[int(np.nanargmin(curve_mse))])
    h1_opt_nll = float(unique_h1[int(np.nanargmin(curve_nll))])

    # “Exponential” vs h1 claim: fit log(MSE) ~ a + b*h1 separately on under/over regions.
    under = unique_h1 < h1_opt_mse
    over = unique_h1 > h1_opt_mse

    reg_under = None
    reg_over = None
    if int(np.sum(under)) >= 3:
        reg_under = _linregress(unique_h1[under], _safe_log(curve_mse[under]))
    if int(np.sum(over)) >= 3:
        reg_over = _linregress(unique_h1[over], _safe_log(curve_mse[over]))

    # ValNLL trend vs bandwidth on the min-over-n curve.
    # (Not log-transformed; used to quantify how quickly ValNLL changes with h1.)
    reg_nll_under = None
    reg_nll_over = None
    if int(np.sum(under)) >= 3:
        reg_nll_under = _linregress(unique_h1[under], curve_nll[under])
    if int(np.sum(over)) >= 3:
        reg_nll_over = _linregress(unique_h1[over], curve_nll[over])

    # “Effect of n” at the MSE-optimal h1: fit log(MSE) ~ a + b*log(n).
    mask_hopt = np.isclose(h1, h1_opt_mse)
    reg_n = None
    if int(np.sum(mask_hopt)) >= 5:
        reg_n = _linregress(_safe_log(n[mask_hopt]), _safe_log(mse[mask_hopt]))

    # Correlation between n and MSE at h1_opt.
    corr_n = _spearman(n[mask_hopt], mse[mask_hopt]) if int(np.sum(mask_hopt)) >= 5 else {"rho": float("nan"), "p": float("nan"), "n": 0}

    return {
        "source_csv": str(path.relative_to(REPO_ROOT)),
        "best_by_grid_mse": best_mse,
        "best_by_val_nll": best_nll,
        "h1_curve": {
            "h1_values": unique_h1.tolist(),
            "min_over_n_grid_mse": curve_mse.tolist(),
            "min_over_n_val_nll": curve_nll.tolist(),
            "h1_opt_mse": h1_opt_mse,
            "h1_opt_nll": h1_opt_nll,
        },
        "regressions": {
            "log_mse_vs_h1_under": None if reg_under is None else reg_under.__dict__,
            "log_mse_vs_h1_over": None if reg_over is None else reg_over.__dict__,
            "log_mse_vs_log_n_at_h1opt": None if reg_n is None else reg_n.__dict__,
            "valnll_vs_h1_under": None if reg_nll_under is None else reg_nll_under.__dict__,
            "valnll_vs_h1_over": None if reg_nll_over is None else reg_nll_over.__dict__,
        },
        "correlations": {
            "spearman_n_vs_mse_at_h1opt": corr_n,
        },
    }


def summarize_pnn_sweep(mixture_idx: int) -> dict[str, Any]:
    path = RESULTS_DIR / f"sweep_results_mixture{mixture_idx}.json"
    blob = read_json(path)
    h1s = np.asarray(blob["bandwidths_h1"], dtype=float)

    kde = blob["kde"]
    kde_mse = np.asarray(kde["grid_mse"], dtype=float)
    kde_nll = np.asarray(kde["val_avg_nll"], dtype=float)

    pnn = blob["pnn"]
    arch_labels = [a["label"] for a in blob["architectures"]]

    pnn_mse_mean = np.asarray(pnn["final_grid_mse"], dtype=float)  # [arch, h1]
    pnn_mse_std = np.asarray(pnn["final_grid_mse_std"], dtype=float)
    pnn_mse_runs = np.asarray(pnn["final_grid_mse_runs"], dtype=float)  # [arch, h1, run]

    pnn_nll = np.asarray(pnn.get("val_avg_nll"), dtype=float)  # [arch, h1]

    # Per-architecture optimal bandwidths.
    arch_summaries: list[dict[str, Any]] = []
    for i, label in enumerate(arch_labels):
        i_mse = int(np.nanargmin(pnn_mse_mean[i]))
        i_nll = int(np.nanargmin(pnn_nll[i])) if pnn_nll.ndim == 2 else i_mse
        arch_summaries.append(
            {
                "label": label,
                "h1_opt_mse": float(h1s[i_mse]),
                "mse_at_h1opt": float(pnn_mse_mean[i, i_mse]),
                "h1_opt_val_nll": float(h1s[i_nll]),
                "val_nll_at_h1opt": float(pnn_nll[i, i_nll]) if pnn_nll.ndim == 2 else float("nan"),
                "std_mse_at_h1=2": float(pnn_mse_std[i, int(np.where(h1s == 2.0)[0][0])]) if 2.0 in set(h1s.tolist()) else float("nan"),
            }
        )

    kde_h1_opt_mse = float(h1s[int(np.nanargmin(kde_mse))])
    kde_h1_opt_nll = float(h1s[int(np.nanargmin(kde_nll))])

    # Undersmoothing comparison at h1=2.0 (if present): PNN runs vs deterministic KDE.
    undersmooth = {}
    if 2.0 in set(h1s.tolist()):
        j = int(np.where(h1s == 2.0)[0][0])
        kde_mse_at = float(kde_mse[j])
        for i, label in enumerate(arch_labels):
            runs = pnn_mse_runs[i, j, :]
            diff = runs - kde_mse_at
            t = _ttest_vs_constant(diff, 0.0)
            d = _cohens_d_one_sample(diff, 0.0)
            undersmooth[label] = {
                "kde_grid_mse": kde_mse_at,
                "pnn_grid_mse_mean": float(np.mean(runs)),
                "pnn_grid_mse_std": float(np.std(runs, ddof=1)),
                "delta_mean": float(np.mean(diff)),
                "ttest_1samp_delta": t,
                "cohens_d_delta": float(d),
            }

    # ValNLL–MSE alignment: correlation across all (arch, h1) points.
    corr = {"rho": float("nan"), "p": float("nan"), "n": 0}
    if pnn_nll.ndim == 2:
        corr = _spearman(pnn_nll.reshape(-1), pnn_mse_mean.reshape(-1))

    return {
        "source_json": str(path.relative_to(REPO_ROOT)),
        "bandwidths_h1": h1s.tolist(),
        "kde": {
            "grid_mse": kde_mse.tolist(),
            "val_avg_nll": kde_nll.tolist(),
            "h1_opt_mse": kde_h1_opt_mse,
            "h1_opt_val_nll": kde_h1_opt_nll,
        },
        "pnn": {
            "architectures": arch_summaries,
            "final_grid_mse": pnn_mse_mean.tolist(),
            "final_grid_mse_std": pnn_mse_std.tolist(),
            "val_avg_nll": pnn_nll.tolist() if isinstance(pnn_nll, np.ndarray) else None,
            "undersmoothed_h1_2": undersmooth,
            "spearman_valnll_vs_mse": corr,
        },
        "best_by_val_nll": blob.get("best_by_val_nll"),
    }


def summarize_boundary_lambda_cache() -> dict[str, Any]:
    path = RESULTS_DIR / "boundary_penalty_lambda_sweep_cache.csv"
    rows = read_csv_rows(path)

    # Convert to numeric arrays grouped by mixture and (arch,h1).
    out: dict[str, Any] = {"source_csv": str(path.relative_to(REPO_ROOT)), "by_mixture": {}}

    # Build indexing.
    for r in rows:
        mix = int(float(r["mixture"]))
        arch = str(r["architecture"])
        h1 = float(r["h1"])
        lam = float(r["lambda"])
        mu = float(r["mean_val_nll"])
        sd = float(r["std_val_nll"])
        n_runs = int(float(r["n_runs"]))

        mix_key = str(mix)
        out["by_mixture"].setdefault(mix_key, {}).setdefault(arch, {}).setdefault(str(h1), []).append(
            {"lambda": lam, "mean_val_nll": mu, "std_val_nll": sd, "n_runs": n_runs}
        )

    # For each mixture/arch/h1 compute best lambda vs lambda=0, plus whether *any* lambda worsens vs 0.
    deltas: list[dict[str, Any]] = []
    for mix_key, by_arch in out["by_mixture"].items():
        for arch, by_h1 in by_arch.items():
            for h1_str, entries in by_h1.items():
                entries_sorted = sorted(entries, key=lambda e: float(e["lambda"]))
                # find lambda=0 and best
                lam0 = next((e for e in entries_sorted if abs(float(e["lambda"])) < 1e-12), None)
                best = min(entries_sorted, key=lambda e: float(e["mean_val_nll"]))
                if lam0 is None:
                    continue

                # Worst (largest) increase relative to lambda=0.
                deltas_vs0 = [float(e["mean_val_nll"]) - float(lam0["mean_val_nll"]) for e in entries_sorted]
                i_worst = int(np.argmax(np.asarray(deltas_vs0, dtype=float)))
                worst = entries_sorted[i_worst]
                deltas.append(
                    {
                        "mixture": int(mix_key),
                        "architecture": arch,
                        "h1": float(h1_str),
                        "val_nll_lambda0": float(lam0["mean_val_nll"]),
                        "best_lambda": float(best["lambda"]),
                        "best_val_nll": float(best["mean_val_nll"]),
                        "delta_best_minus_0": float(best["mean_val_nll"] - lam0["mean_val_nll"]),
                        "worst_lambda": float(worst["lambda"]),
                        "worst_val_nll": float(worst["mean_val_nll"]),
                        "delta_worst_minus_0": float(worst["mean_val_nll"] - lam0["mean_val_nll"]),
                    }
                )

    out["delta_table"] = deltas

    # Mixture-level summaries for convenient reporting.
    mix_summary: dict[str, dict[str, float]] = {}
    for mix in sorted({int(d["mixture"]) for d in deltas}):
        here = [d for d in deltas if int(d["mixture"]) == mix]
        best_arr = np.asarray([float(d["delta_best_minus_0"]) for d in here], dtype=float)
        worst_arr = np.asarray([float(d.get("delta_worst_minus_0", float("nan"))) for d in here], dtype=float)
        best_arr = best_arr[np.isfinite(best_arr)]
        worst_arr = worst_arr[np.isfinite(worst_arr)]
        if best_arr.size == 0 or worst_arr.size == 0:
            continue
        mix_summary[str(mix)] = {
            "n_configs": float(best_arr.size),
            "prop_best_improves": float(np.mean(best_arr < 0.0)),
            "mean_best_delta": float(np.mean(best_arr)),
            "median_best_delta": float(np.median(best_arr)),
            "min_best_delta": float(np.min(best_arr)),
            "max_best_delta": float(np.max(best_arr)),
            "q10_best_delta": float(np.quantile(best_arr, 0.10)),
            "q90_best_delta": float(np.quantile(best_arr, 0.90)),
            "prop_any_worsens": float(np.mean(worst_arr > 0.0)),
            "mean_worst_delta": float(np.mean(worst_arr)),
            "max_worst_delta": float(np.max(worst_arr)),
        }
    out["mixture_summary"] = mix_summary
    return out


def extract_figures_and_comments(tex_path: Path) -> list[dict[str, Any]]:
    """Very lightweight parsing: finds figure envs and grabs the next non-empty paragraph as comment."""

    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    figures: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("\\begin{figure}"):
            fig_block = []
            start = i
            while i < len(lines) and "\\end{figure}" not in lines[i]:
                fig_block.append(lines[i])
                i += 1
            if i < len(lines):
                fig_block.append(lines[i])
            block = "\n".join(fig_block)
            # extract includegraphics files
            inc = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", block)
            if not inc:
                inc = re.findall(r"\\includegraphics\{([^}]+)\}", block)
            label_m = re.search(r"\\label\{([^}]+)\}", block)
            label = label_m.group(1) if label_m else None
            caption_m = re.search(r"\\caption\{([\s\S]*?)\}\s*\\label", block)
            caption = caption_m.group(1).strip() if caption_m else None

            # Next comment paragraph: collect consecutive non-empty lines until blank or next figure/section.
            comment_lines: list[str] = []
            j = i + 1
            # skip whitespace and centering directives
            while j < len(lines) and lines[j].strip() in {"", "\\centering", "\\newpage"}:
                j += 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith("\\begin{figure}") or s.startswith("\\subsection") or s.startswith("\\subsubsection"):
                    break
                if s == "":
                    break
                comment_lines.append(lines[j])
                j += 1

            figures.append(
                {
                    "label": label,
                    "includes": inc,
                    "caption": caption,
                    "comment": "\n".join(comment_lines).strip(),
                    "line_start": start + 1,
                }
            )
        i += 1

    return figures


def build_evidence() -> dict[str, Any]:
    figures = extract_figures_and_comments(REPORT_TEX)

    pw = {str(m): summarize_pw_errors(m) for m in (1, 2, 3)}
    pnn = {str(m): summarize_pnn_sweep(m) for m in (1, 2, 3)}
    boundary = summarize_boundary_lambda_cache()

    # Build per-figure addenda (LaTeX-ready), keyed by label.
    addenda: dict[str, str] = {}

    # Parzen Window error figures.
    for mix in (1, 2, 3):
        pw_m = pw[str(mix)]
        best_mse = pw_m["best_by_grid_mse"]
        best_nll = pw_m["best_by_val_nll"]
        reg_u = pw_m["regressions"]["log_mse_vs_h1_under"]
        reg_o = pw_m["regressions"]["log_mse_vs_h1_over"]
        reg_n = pw_m["regressions"]["log_mse_vs_log_n_at_h1opt"]

        slope_u = reg_u["slope"] if reg_u else float("nan")
        r2_u = reg_u["r2"] if reg_u else float("nan")
        slope_o = reg_o["slope"] if reg_o else float("nan")
        r2_o = reg_o["r2"] if reg_o else float("nan")

        alpha_n = reg_n["slope"] if reg_n else float("nan")
        r2_n = reg_n["r2"] if reg_n else float("nan")

        # Note: reg on log(MSE) vs log(n) slope is an elasticity.
        add = (
            "\\par\\noindent\\textbf{Evidence.} "
            f"From the sweep in \\texttt{{{pw_m['source_csv']}}}, "
            f"the global minimizers are "
            f"$\\arg\\min_{{(n,h_1)}}\\mathrm{{MSE}}=(n={best_mse['n']},\\ h_1={_format_float(best_mse['h1'])})$ "
            f"with $\\mathrm{{MSE}}={_format_float(best_mse['grid_mse'])}$, and "
            f"$\\arg\\min_{{(n,h_1)}}\\mathrm{{ValNLL}}=(n={best_nll['n']},\\ h_1={_format_float(best_nll['h1'])})$ "
            f"with $\\mathrm{{ValNLL}}={_format_float(best_nll['val_avg_nll'])}$. "
        )

        if np.isfinite(slope_u) and np.isfinite(slope_o):
            add += (
                f"Fitting $\\log\\mathrm{{MSE}}(h_1)=a+bh_1$ on the lower-bandwidth side ($h_1<h_1^*$) gives "
                f"$b={_format_float(slope_u)}$ ($R^2={_format_float(r2_u)}$), "
                f"whereas on the oversmoothed side ($h_1>h_1^*$) it gives "
                f"$b={_format_float(slope_o)}$ ($R^2={_format_float(r2_o)}$), "
                f"quantifying the steeper rise under undersmoothing. "
            )

        if np.isfinite(alpha_n):
            add += (
                f"At $h_1=h_1^*$, a log--log fit $\\log\\mathrm{{MSE}}=c+\\alpha\\log n$ yields "
                f"$\\alpha={_format_float(alpha_n)}$ ($R^2={_format_float(r2_n)}$), i.e. "
                f"$\\mathrm{{MSE}}\\propto n^{{{_format_float(alpha_n)}}}$ on this slice. "
            )

        addenda[f"fig:parzen-errors-{mix}"] = add

    # Parzen Window overlay figures: compare selected n for ValNLL vs MSE.
    for mix in (1, 2, 3):
        pw_m = pw[str(mix)]
        best_mse = pw_m["best_by_grid_mse"]
        best_nll = pw_m["best_by_val_nll"]
        addenda[f"fig:parzen-overlay-{mix}"] = (
            "\\par\\noindent\\textbf{Evidence.} "
            f"The configuration selected by minimizing grid MSE uses $n={best_mse['n']}$ and $h_1={_format_float(best_mse['h1'])}$ "
            f"(grid $\\mathrm{{MSE}}={_format_float(best_mse['grid_mse'])}$), while the ValNLL-minimizer uses "
            f"$n={best_nll['n']}$ and $h_1={_format_float(best_nll['h1'])}$ "
            f"(ValNLL $={_format_float(best_nll['val_avg_nll'])}$). "
            f"In particular, $n_{{\\mathrm{{MSE}}}}-n_{{\\mathrm{{ValNLL}}}}={best_mse['n']-best_nll['n']}$ and "
            f"$h_{{1,\\mathrm{{MSE}}}}-h_{{1,\\mathrm{{ValNLL}}}}={_format_float(best_mse['h1']-best_nll['h1'])}$."
        )

    # PNN error figures.
    for mix in (1, 2, 3):
        pnn_m = pnn[str(mix)]
        h1s = np.asarray(pnn_m["bandwidths_h1"], dtype=float)
        kde = pnn_m["kde"]
        kde_h1_mse = float(kde["h1_opt_mse"])

        # Determine if PNN opts are lower than KDE on this discrete grid.
        opts = []
        for arch in pnn_m["pnn"]["architectures"]:
            opts.append(float(arch["h1_opt_mse"]))

        median_opt = float(np.median(np.asarray(opts, dtype=float))) if opts else float("nan")

        corr = pnn_m["pnn"]["spearman_valnll_vs_mse"]
        corr_str = f"$\\rho={_format_float(corr['rho'])}$ (p={_format_float(corr['p'])}, n={int(corr['n'])})" if np.isfinite(corr["rho"]) else "(insufficient points)"

        add = (
            "\\par\\noindent\\textbf{Evidence.} "
            f"On the bandwidth grid $h_1\\in\\{{{', '.join(_format_float(x) for x in h1s)}\\}}$ from \\texttt{{{pnn_m['source_json']}}}, "
            f"the KDE reference attains its minimum grid-MSE at $h_1^{{\\mathrm{{KDE}}}}={_format_float(kde_h1_mse)}$. "
            f"Across PNN architectures, the median MSE-optimal bandwidth is $\\mathrm{{median}}(h_1^{{\\mathrm{{PNN}}}})={_format_float(median_opt)}$ on this grid. "
        )

        if pnn_m["pnn"]["undersmoothed_h1_2"]:
            # summarize best improvement at h1=2
            best_arch = min(
                pnn_m["pnn"]["undersmoothed_h1_2"].items(),
                key=lambda kv: float(kv[1]["delta_mean"]),
            )
            label, dct = best_arch
            t = dct["ttest_1samp_delta"]
            add += (
                f"At $h_1=2$ (undersmoothed slice), the best architecture ({label}) changes grid-MSE by "
                f"$\\Delta=\\mathbb{{E}}[\\mathrm{{MSE}}_\\mathrm{{PNN}}-\\mathrm{{MSE}}_\\mathrm{{KDE}}]={_format_float(dct['delta_mean'])}$ "
                f"with one-sample t-test $t={_format_float(t['t'])}$ (p={_format_float(t['p'])}, n={int(t['n'])}). "
            )

        add += f"Across all (architecture, $h_1$) points, ValNLL vs grid-MSE alignment is {corr_str}."
        addenda[f"fig:pnn-error-{mix}"] = add

    # PNN overlay figures: per-architecture best-by-ValNLL bandwidth and relative MSE.
    for mix in (1, 2, 3):
        pnn_m = pnn[str(mix)]
        h1s = np.asarray(pnn_m["bandwidths_h1"], dtype=float)
        kde_mse = np.asarray(pnn_m["kde"]["grid_mse"], dtype=float)
        pnn_mse = np.asarray(pnn_m["pnn"]["final_grid_mse"], dtype=float)
        pnn_nll = np.asarray(pnn_m["pnn"]["val_avg_nll"], dtype=float)
        labels = [a["label"] for a in pnn_m["pnn"]["architectures"]]

        # Build a concise table row list: label -> h1* (valnll) and resulting grid-mse.
        rows = []
        for i, lab in enumerate(labels):
            j = int(np.nanargmin(pnn_nll[i]))
            rows.append((lab, float(h1s[j]), float(pnn_nll[i, j]), float(pnn_mse[i, j])))

        # pick best and worst by valnll.
        best = min(rows, key=lambda t: t[2])
        worst = max(rows, key=lambda t: t[2])

        # KDE at the best arch's h1.
        j_best = int(np.where(h1s == best[1])[0][0])
        kde_here = float(kde_mse[j_best])

        addenda[f"fig:pnn-overlay-{mix}"] = (
            "\\par\\noindent\\textbf{Evidence.} "
            f"Per-architecture ValNLL selection yields (best) {best[0]} at $h_1={_format_float(best[1])}$ with "
            f"ValNLL $={_format_float(best[2])}$ and grid-MSE $={_format_float(best[3])}$; "
            f"(worst) {worst[0]} at $h_1={_format_float(worst[1])}$ with ValNLL $={_format_float(worst[2])}$. "
            f"At the selected $h_1={_format_float(best[1])}$, KDE has grid-MSE $={_format_float(kde_here)}$, "
            f"so $\\Delta\\mathrm{{MSE}}=\\mathrm{{MSE}}_\\mathrm{{PNN}}-\\mathrm{{MSE}}_\\mathrm{{KDE}}={_format_float(best[3]-kde_here)}$."
        )

    # Boundary penalty figures: summarize average delta across arch/h1 and whether min ValNLL improves.
    bd = boundary
    # compute mixture-level summary: for each mixture, proportion of configs where best lambda != 0 and improves.
    delta_rows = bd["delta_table"]
    mix_summ: dict[int, dict[str, float]] = {}
    for mix in (1, 2, 3):
        here = [r for r in delta_rows if int(r["mixture"]) == mix]
        if not here:
            continue
        deltas = np.array([float(r["delta_best_minus_0"]) for r in here], dtype=float)
        improves = int(np.sum(deltas < 0.0))
        mix_summ[mix] = {
            "n_configs": float(len(here)),
            "prop_improves": float(improves / max(1, len(here))),
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
        }

    for mix in (1, 2, 3):
        s = mix_summ.get(mix)
        if not s:
            continue
        addenda[f"fig:pnn-boundary-{mix}"] = (
            "\\par\\noindent\\textbf{Evidence.} "
            f"Using \\texttt{{{bd['source_csv']}}}, across all cached (architecture,$h_1$) settings for mixture {mix}, "
            f"the best $\\lambda$ improves mean ValNLL relative to $\\lambda=0$ in "
            f"{_format_float(100.0*s['prop_improves'])}\\% of configurations; "
            f"the average change is $\\mathbb{{E}}[\\Delta]= {_format_float(s['mean_delta'])}$ and the best observed change is "
            f"$\\min\\Delta={_format_float(s['min_delta'])}$ (negative = improvement)."
        )

    evidence = {
        "repo_root": str(REPO_ROOT),
        "figures_in_report": figures,
        "pw": pw,
        "pnn": pnn,
        "boundary": boundary,
        "latex_addenda_by_label": addenda,
    }
    return evidence


def write_outputs(evidence: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "evidence_per_figure.json"
    out_csv = RESULTS_DIR / "evidence_summary.csv"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=False)

    # Summary CSV: one row per label with first 160 chars of addendum.
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "addendum_preview"])
        writer.writeheader()
        for label, txt in sorted(evidence.get("latex_addenda_by_label", {}).items()):
            preview = re.sub(r"\s+", " ", txt).strip()[:160]
            writer.writerow({"label": label, "addendum_preview": preview})


def main() -> None:
    evidence = build_evidence()
    write_outputs(evidence)

    labels = evidence.get("latex_addenda_by_label", {})
    print(f"Wrote {RESULTS_DIR / 'evidence_per_figure.json'}")
    print(f"Wrote {RESULTS_DIR / 'evidence_summary.csv'}")
    print(f"Prepared LaTeX addenda for {len(labels)} figure labels")


if __name__ == "__main__":
    main()
