import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestBoundaryPenaltyNLLTradeoff(unittest.TestCase):
    def test_boundary_penalty_increases_val_nll_in_saved_demo(self):
        """Theory check: boundary penalty can reduce tails, but it can also bias the estimator.

        In the saved boundary_penalty_demo runs, validation NLL should reveal whether the penalty helps.
        Current saved artifacts show a small *increase* in NLL for all mixtures.
        """

        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            demo = r.boundary_penalty_demo
            self.assertTrue(demo, msg=f"Missing boundary_penalty_demo for mixture {mix_id}")

            nll0 = float(demo["lambda_0"]["val_nll"])
            nll1 = float(demo["lambda_1"]["val_nll"])

            self.assertGreater(
                nll1,
                nll0,
                msg=f"Expected penalty to increase NLL in saved demo for mixture {mix_id}: {nll0} -> {nll1}",
            )


if __name__ == "__main__":
    unittest.main()
