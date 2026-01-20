from __future__ import annotations

import unittest

from .results_analysis import load_sweep_results, results_dir


class TestPwVsPnnComparison(unittest.TestCase):
    """Theory: a learned surrogate can match or improve PW/KDE selection metrics on the same domain."""

    def test_best_pnn_nll_not_worse_than_kde(self) -> None:
        improvements = 0
        for mixture in (1, 2, 3):
            sr = load_sweep_results(results_dir() / f"sweep_results_mixture{mixture}.json")

            best_kde = min(sr.kde_val_avg_nll)
            best_pnn = min(min(row) for row in sr.pnn_val_avg_nll)

            # The PNN is a surrogate trained on Parzen-style targets; it should be competitive with KDE.
            self.assertLessEqual(
                best_pnn,
                best_kde + 0.01,
                f"Mixture {mixture}: best PNN NLL ({best_pnn:.4f}) too worse than KDE ({best_kde:.4f})",
            )

            if best_pnn + 0.02 < best_kde:
                improvements += 1

        # In the saved artifacts, at least one mixture should show a noticeable NLL gain.
        self.assertGreaterEqual(improvements, 1)

    def test_best_pnn_oracle_mse_close_to_or_better_than_kde(self) -> None:
        for mixture in (1, 2, 3):
            sr = load_sweep_results(results_dir() / f"sweep_results_mixture{mixture}.json")

            best_kde = min(sr.kde_grid_mse)
            best_pnn = min(min(row) for row in sr.pnn_final_grid_mse)

            # Allow small numerical/modeling differences but enforce "same order".
            self.assertLessEqual(
                best_pnn,
                1.10 * best_kde,
                f"Mixture {mixture}: best PNN MSE ({best_pnn:.3e}) much worse than KDE ({best_kde:.3e})",
            )
