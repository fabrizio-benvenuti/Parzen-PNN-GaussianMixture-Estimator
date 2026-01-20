import math
import unittest

import numpy as np

from tests.results_analysis import results_dir


class TestComplexityLOOCVQuadraticScaling(unittest.TestCase):
    def test_loglog_slope_near_2(self):
        """Theory: leave-one-out KDE target construction is O(n^2).

        Using the measured CSV, fit log(seconds) = a + b log(n). We expect b ~ 2.
        """

        csv_path = results_dir() / "training_complexity_loocv.csv"
        rows = csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]

        n = []
        t = []
        for line in rows:
            parts = line.split(",")
            n.append(float(parts[0]))
            t.append(float(parts[1]))

        x = np.log(np.asarray(n))
        y = np.log(np.asarray(t))
        b, a = np.polyfit(x, y, deg=1)

        self.assertGreater(b, 1.7, msg=f"slope too small: {b}")
        self.assertLess(b, 2.3, msg=f"slope too large: {b}")

        # Also check growth factor roughly matches 4x when doubling n.
        # We use consecutive pairs.
        for i in range(len(n) - 1):
            ratio_n = n[i + 1] / n[i]
            ratio_t = t[i + 1] / t[i]
            expected = ratio_n ** 2
            # allow wide tolerance because of constants/cache effects
            self.assertLess(abs(math.log(ratio_t / expected)), math.log(2.0))


if __name__ == "__main__":
    unittest.main()
