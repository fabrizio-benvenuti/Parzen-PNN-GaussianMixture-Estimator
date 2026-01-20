import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestQuantitativeBiasVarianceOptimalH1(unittest.TestCase):
    def test_optimal_h1_not_always_extreme(self):
        """Theory: bias-variance trade-off suggests an intermediate bandwidth minimizes NLL.

        With a coarse grid of h1 values, we should still see the minimizer *often* not at an extreme.
        """

        mixes = [
            load_sweep_results(results_dir() / "sweep_results_mixture1.json"),
            load_sweep_results(results_dir() / "sweep_results_mixture2.json"),
            load_sweep_results(results_dir() / "sweep_results_mixture3.json"),
        ]

        def is_extreme(idx: int, n: int) -> bool:
            return idx == 0 or idx == n - 1

        # KDE: allow one mixture to land on an edge due to coarse sweep.
        kde_non_extreme = 0
        for r in mixes:
            idx = min(range(len(r.kde_val_avg_nll)), key=lambda i: r.kde_val_avg_nll[i])
            if not is_extreme(idx, len(r.kde_val_avg_nll)):
                kde_non_extreme += 1
        self.assertGreaterEqual(
            kde_non_extreme,
            2,
            msg=f"Expected >=2/3 KDE optima to be non-extreme; got {kde_non_extreme}/3",
        )

        # PNN: min over architectures at each h1, then pick the best h1.
        pnn_non_extreme = 0
        for r in mixes:
            best_per_h1 = [min(r.pnn_val_avg_nll[a][j] for a in range(len(r.architectures))) for j in range(len(r.bandwidths_h1))]
            idx = min(range(len(best_per_h1)), key=lambda i: best_per_h1[i])
            if not is_extreme(idx, len(best_per_h1)):
                pnn_non_extreme += 1
        self.assertGreaterEqual(
            pnn_non_extreme,
            3,
            msg=f"Expected all PNN optima to be non-extreme; got {pnn_non_extreme}/3",
        )


if __name__ == "__main__":
    unittest.main()
