from __future__ import annotations

import csv
import unittest
from pathlib import Path

from .results_analysis import logs_dir


def _rows_for_n(path: Path, n: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(float(r["n"])) != int(n):
                continue
            rows.append(
                {
                    "h1": float(r["h1"]),
                    "grid_mse": float(r["grid_mse"]),
                    "val_avg_nll": float(r["val_avg_nll"]),
                }
            )
    rows.sort(key=lambda d: d["h1"])
    return rows


class TestPwBiasVarianceOptimalH1(unittest.TestCase):
    """Theory: KDE/PW exhibits a bias-variance tradeoff in bandwidth h1 with an interior optimum."""

    def test_optimum_not_at_extremes(self) -> None:
        for mixture in (1, 2, 3):
            path = logs_dir() / f"pw_errors_mixture{mixture}.csv"
            self.assertTrue(path.exists(), f"Missing PW log: {path}")

            # Choose a representative sample size (median n in the sweep).
            ns: set[int] = set()
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ns.add(int(float(r["n"])))
            n_sorted = sorted(ns)
            self.assertGreaterEqual(len(n_sorted), 5)
            n_mid = n_sorted[len(n_sorted) // 2]

            rows = _rows_for_n(path, n_mid)
            self.assertGreaterEqual(len(rows), 10)

            mse = [r["grid_mse"] for r in rows]
            nll = [r["val_avg_nll"] for r in rows]
            idx_mse = int(min(range(len(mse)), key=lambda i: mse[i]))
            idx_nll = int(min(range(len(nll)), key=lambda i: nll[i]))

            # "Interior" means not in the first/last few bandwidths.
            margin = 3
            self.assertGreater(idx_mse, margin, f"m={mixture}: MSE optimum too close to smallest h1")
            self.assertLess(idx_mse, len(rows) - 1 - margin, f"m={mixture}: MSE optimum too close to largest h1")
            self.assertGreater(idx_nll, margin, f"m={mixture}: NLL optimum too close to smallest h1")
            self.assertLess(idx_nll, len(rows) - 1 - margin, f"m={mixture}: NLL optimum too close to largest h1")
