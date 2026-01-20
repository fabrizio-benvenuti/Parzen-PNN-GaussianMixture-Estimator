import statistics
import unittest

from tests.log_analysis import iter_all_logs, last_run, tail_std
from tests.results_analysis import logs_dir


class TestOverlaysTailVarianceHigherLowH1(unittest.TestCase):
    def test_evalmse_tail_is_noisier_low_h1(self):
        """Theory: undersmoothing (small h1) yields spiky targets, which can make training noisier.

        We quantify 'noise' as the standard deviation of EvalMSE over the last 5 logged checkpoints.
        """

        low = []
        high = []

        for log in iter_all_logs(logs_dir()):
            run = last_run(log)
            series = [pt.eval_mse for pt in run]
            if len(series) < 6:
                continue
            s = tail_std(series, k=5)
            if abs(log.config.h1 - 2.0) < 1e-9:
                low.append(s)
            elif abs(log.config.h1 - 16.0) < 1e-9:
                high.append(s)

        self.assertGreater(len(low), 5)
        self.assertGreater(len(high), 5)

        self.assertGreaterEqual(
            statistics.median(low),
            statistics.median(high),
            msg=f"Expected noisier tails for h1=2 than h1=16; medians {statistics.median(low)} vs {statistics.median(high)}",
        )


if __name__ == "__main__":
    unittest.main()
