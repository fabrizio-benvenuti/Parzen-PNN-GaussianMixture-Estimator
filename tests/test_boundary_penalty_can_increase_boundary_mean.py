import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestBoundaryPenaltyCanIncreaseBoundaryMean(unittest.TestCase):
    def test_boundary_mean_decreases_in_saved_demo(self):
        """Sanity check on the saved artifacts.

        With boundary points sampled outside the convex hull, the stored demo shows the boundary penalty
        reducing mean unnormalized density on the boundary set for all mixtures.
        """

        deltas = []
        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            demo = r.boundary_penalty_demo
            b0 = float(demo["lambda_0"]["boundary_mean"])
            b1 = float(demo["lambda_1"]["boundary_mean"])
            deltas.append(b1 - b0)

        self.assertTrue(all(d < 0 for d in deltas), msg=f"Expected all deltas < 0, got {deltas}")


if __name__ == "__main__":
    unittest.main()
