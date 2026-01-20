import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestBoundaryPenaltyEffectSizeTracksBoundaryMass(unittest.TestCase):
    def test_larger_baseline_boundary_mass_changes_more(self):
        """Theory: penalty impact should be larger when there is more mass near the boundary.

        Using the saved demo, mixtures with larger baseline boundary_mean should show a larger absolute
        change when the penalty is turned on.
        """

        baseline = []
        change = []
        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            demo = r.boundary_penalty_demo
            b0 = float(demo["lambda_0"]["boundary_mean"])
            b1 = float(demo["lambda_1"]["boundary_mean"])
            baseline.append(b0)
            change.append(abs(b1 - b0))

        # Check monotonic ordering (baseline and change both increase with mixture id in the saved artifacts).
        self.assertLessEqual(baseline[0], baseline[1])
        self.assertLessEqual(baseline[1], baseline[2])
        self.assertLessEqual(change[0], change[1])
        self.assertLessEqual(change[1], change[2])


if __name__ == "__main__":
    unittest.main()
