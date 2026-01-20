import re
import unittest
from pathlib import Path


class TestComplexityRuntimeSpeedup(unittest.TestCase):
    def test_reported_inference_speedup_is_large(self):
        """Theory: KDE inference scales with n, while PNN forward pass does not.

        This test sanity-checks the report's Table runtime numbers imply a large speedup.
        """

        tex_path = Path(__file__).resolve().parents[1] / "experiment_report.tex"
        tex = tex_path.read_text(encoding="utf-8")

        # Extract the two time numbers from the runtime table.
        # Example lines:
        # PW (Gaussian KDE) & 2000 & $10^4$ & 0.923\\
        # PNN forward (fixed $W$) & -- & $10^4$ & 7.66\,$\times 10^{-4}$\\
        m_kde = re.search(r"PW \(Gaussian KDE\)\s*&\s*2000\s*&\s*\$10\^4\$\s*&\s*([0-9.]+)", tex)
        self.assertIsNotNone(m_kde, msg="Could not find KDE runtime row")
        kde_s = float(m_kde.group(1))

        m_pnn = re.search(r"PNN forward \(fixed \$W\$\)\s*&\s*--\s*&\s*\$10\^4\$\s*&\s*([0-9.]+)\\,\$\\times 10\^\{-4\}\$", tex)
        self.assertIsNotNone(m_pnn, msg="Could not find PNN runtime row")
        pnn_coeff = float(m_pnn.group(1))
        pnn_s = pnn_coeff * 1e-4

        speedup = kde_s / pnn_s
        self.assertGreater(speedup, 1000.0, msg=f"Speedup too small: {speedup}x")


if __name__ == "__main__":
    unittest.main()
