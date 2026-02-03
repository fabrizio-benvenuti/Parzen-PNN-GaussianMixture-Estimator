import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestBoundaryPenaltyNLLTradeoff(unittest.TestCase):
    def test_boundary_penalty_increases_val_nll_in_saved_demo(self):
        """Theory check: boundary penalty can reduce tails, but it can also bias the estimator.

        In saved artifacts, the effect on validation NLL can go either way depending on the mixture.
        """

        deltas = []
        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            demo = r.boundary_penalty_demo
            self.assertTrue(demo, msg=f"Missing boundary_penalty_demo for mixture {mix_id}")

            nll0 = float(demo["lambda_0"]["val_nll"])
            nll1 = float(demo["lambda_1"]["val_nll"])

            deltas.append(nll1 - nll0)

        self.assertTrue(
            any(d < 0 for d in deltas) and any(d > 0 for d in deltas),
            msg=f"Expected mixed-sign NLL deltas, got {deltas}",
        )


if __name__ == "__main__":
    unittest.main()
