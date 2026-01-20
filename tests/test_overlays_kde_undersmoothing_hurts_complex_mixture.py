import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestOverlaysKDEUndersmoothingHurtsComplexMixture(unittest.TestCase):
    def test_kde_nll_at_low_h1_explodes_for_complex_mixture(self):
        """Theory: for small h1, KDE can assign near-zero density away from samples.

        The NLL then becomes very large if validation points fall in regions with insufficient kernel
        coverage. This risk increases with more multimodal/complex target densities.
        """

        m1 = load_sweep_results(results_dir() / "sweep_results_mixture1.json")
        m3 = load_sweep_results(results_dir() / "sweep_results_mixture3.json")

        # h1 list is [2,7,12,16] so low bandwidth corresponds to index 0.
        nll_m1 = m1.kde_val_avg_nll[0]
        nll_m3 = m3.kde_val_avg_nll[0]

        self.assertGreater(
            nll_m3 - nll_m1,
            2.0,
            msg=f"Expected Mixture3 KDE NLL at h1=2 to be much worse than Mixture1; got {nll_m3} vs {nll_m1}",
        )


if __name__ == "__main__":
    unittest.main()
