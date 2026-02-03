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

        # The exact winner can shift with retraining, but the key hypothesis is that the best output
        # is mixture-dependent (i.e., not identical across mixtures).
        self.assertNotEqual(
            m2.best_by_val_nll["label"],
            m3.best_by_val_nll["label"],
            msg=f"Expected different best labels: m2={m2.best_by_val_nll}, m3={m3.best_by_val_nll}",
        )
        self.assertTrue(
            ("outSigmoid" in m2.best_by_val_nll["label"]) or ("outReLU" in m2.best_by_val_nll["label"]),
            msg=f"Unexpected mixture2 best label format: {m2.best_by_val_nll}",
        )
        self.assertTrue(
            ("outSigmoid" in m3.best_by_val_nll["label"]) or ("outReLU" in m3.best_by_val_nll["label"]),
            msg=f"Unexpected mixture3 best label format: {m3.best_by_val_nll}",
        )


if __name__ == "__main__":
    unittest.main()
