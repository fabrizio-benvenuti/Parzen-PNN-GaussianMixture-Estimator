from __future__ import annotations

import csv
import unittest
from pathlib import Path

from .results_analysis import logs_dir


def _load_pw_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "mixture": float(r["mixture"]),
                    "n": float(r["n"]),
                    "h1": float(r["h1"]),
                    "grid_mse": float(r["grid_mse"]),
                    "val_avg_nll": float(r["val_avg_nll"]),
                }
            )
    return rows


class TestPwBestMseDecreasesWithN(unittest.TestCase):
    """Theory: as sample size n grows, the best achievable KDE/PW error should decrease."""

    def test_best_grid_mse_decreases_with_n(self) -> None:
        for mixture in (1, 2, 3):
            path = logs_dir() / f"pw_errors_mixture{mixture}.csv"
            self.assertTrue(path.exists(), f"Missing PW log: {path}")
            rows = _load_pw_rows(path)

            # Compute min MSE over h1 for each n.
            best_by_n: dict[int, float] = {}
            for r in rows:
                n = int(r["n"])
                mse = float(r["grid_mse"])
                best_by_n[n] = min(best_by_n.get(n, float("inf")), mse)

            ns = sorted(best_by_n.keys())
            self.assertGreaterEqual(len(ns), 10)

            k = 5
            low = [best_by_n[n] for n in ns[:k]]
            high = [best_by_n[n] for n in ns[-k:]]

            mean_low = sum(low) / len(low)
            mean_high = sum(high) / len(high)

            # Require a clear improvement on average.
            self.assertLess(
                mean_high,
                0.85 * mean_low,
                f"Expected best MSE to drop with n (mixture={mixture}): low={mean_low:.3e}, high={mean_high:.3e}",
            )
