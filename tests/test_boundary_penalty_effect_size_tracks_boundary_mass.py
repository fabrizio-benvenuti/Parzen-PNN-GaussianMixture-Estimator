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

        # Check that larger baseline boundary mass tends to produce a larger absolute change.
        # Use rank-order agreement to avoid assuming any particular ordering by mixture id.
        baseline_order = sorted(range(len(baseline)), key=lambda i: baseline[i])
        change_order = sorted(range(len(change)), key=lambda i: change[i])
        self.assertEqual(
            baseline_order,
            change_order,
            msg=f"Expected baseline and change to have same ordering. baseline={baseline}, change={change}",
        )


if __name__ == "__main__":
    unittest.main()
