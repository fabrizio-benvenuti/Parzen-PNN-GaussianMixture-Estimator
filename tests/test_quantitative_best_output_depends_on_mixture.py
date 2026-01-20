import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestQuantitativeBestOutputDependsOnMixture(unittest.TestCase):
    def test_best_by_val_nll_uses_different_output_across_mixtures(self):
        """Theory: output constraints interact with peak sharpness/dynamic range.

        Scaled-sigmoid can act as an implicit cap (helpful for smoother targets), while ReLU is unbounded
        (helpful for sharper peaks). The saved sweeps should reflect a mixture-dependent best output.
        """

        m2 = load_sweep_results(results_dir() / "sweep_results_mixture2.json")
        m3 = load_sweep_results(results_dir() / "sweep_results_mixture3.json")

        self.assertIn("outSigmoid", m2.best_by_val_nll["label"], msg=f"Mixture2 best label: {m2.best_by_val_nll}")
        self.assertIn("outReLU", m3.best_by_val_nll["label"], msg=f"Mixture3 best label: {m3.best_by_val_nll}")


if __name__ == "__main__":
    unittest.main()
