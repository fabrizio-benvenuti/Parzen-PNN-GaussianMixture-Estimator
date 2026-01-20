import statistics
import unittest

from tests.log_analysis import iter_all_logs, last_run
from tests.results_analysis import logs_dir


class TestOverlaysConvergenceSlowerLowH1(unittest.TestCase):
    def test_low_h1_converges_slower(self):
        """Theory: smaller bandwidth yields sharper targets (higher curvature), often slowing optimization.

        Important: convergence must be measured in a way that is comparable across runs.
        Using “within X% of the best achieved MSE” is biased because runs with worse best-MSE will hit
        that threshold earlier.

        Here we define convergence as the epoch at which EvalMSE achieves 90% of the total improvement:
        letting e0 be the initial EvalMSE and e* the minimum EvalMSE, we look for the first epoch where
        EvalMSE <= e* + 0.1 (e0 - e*).
        """

        def epoch_to_fraction_improvement(run, frac: float = 0.9):
            if not run:
                return None
            e0 = run[0].eval_mse
            e_star = min(pt.eval_mse for pt in run)
            target = e_star + (1.0 - frac) * (e0 - e_star)
            for pt in run:
                if pt.eval_mse <= target:
                    return pt.epoch
            return None

        low_epochs = []
        high_epochs = []

        for log in iter_all_logs(logs_dir()):
            run = last_run(log)
            t = epoch_to_fraction_improvement(run, frac=0.9)
            if t is None:
                continue

            if abs(log.config.h1 - 2.0) < 1e-9:
                low_epochs.append(t)
            elif abs(log.config.h1 - 16.0) < 1e-9:
                high_epochs.append(t)

        self.assertGreater(len(low_epochs), 5)
        self.assertGreater(len(high_epochs), 5)

        self.assertGreaterEqual(
            statistics.median(low_epochs),
            statistics.median(high_epochs),
            msg=(
                f"Expected slower convergence for h1=2 than h1=16; "
                f"medians {statistics.median(low_epochs)} vs {statistics.median(high_epochs)}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
