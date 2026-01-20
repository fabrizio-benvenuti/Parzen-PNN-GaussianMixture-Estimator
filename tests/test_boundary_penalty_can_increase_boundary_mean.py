import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestBoundaryPenaltyCanIncreaseBoundaryMean(unittest.TestCase):
    def test_boundary_mean_not_always_decreases(self):
        """Theory: with finite-domain renormalization, boundary penalty can be counteracted by rescaling.

        Therefore, the mean predicted density near the boundary shell is not guaranteed to decrease.
        The saved artifacts include at least one mixture where boundary_mean increases.
        """

        deltas = []
        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            demo = r.boundary_penalty_demo
            b0 = float(demo["lambda_0"]["boundary_mean"])
            b1 = float(demo["lambda_1"]["boundary_mean"])
            deltas.append(b1 - b0)

        self.assertTrue(any(d > 0 for d in deltas), msg=f"Expected at least one positive delta, got {deltas}")


if __name__ == "__main__":
    unittest.main()
