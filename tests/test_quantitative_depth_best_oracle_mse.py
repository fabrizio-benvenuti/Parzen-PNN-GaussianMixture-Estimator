import unittest

from tests.results_analysis import load_sweep_results, results_dir


class TestQuantitativeDepthBestOracleMSE(unittest.TestCase):
    def test_deeper_arch_reaches_best_oracle_mse_each_mixture(self):
        """Theory: higher-capacity networks should be able to match a richer target mapping.

        We check that, in the saved sweeps, the minimum oracle grid MSE per mixture is achieved by a
        2-hidden-layer architecture ([30,20]).
        """

        for mix_id in (1, 2, 3):
            r = load_sweep_results(results_dir() / f"sweep_results_mixture{mix_id}.json")
            # Identify which architectures are deep.
            is_deep = [len(a["hidden_layers"]) >= 2 for a in r.architectures]

            best_mse = None
            best_is_deep = None
            for ai, _arch in enumerate(r.architectures):
                for hi in range(len(r.bandwidths_h1)):
                    mse = r.pnn_final_grid_mse[ai][hi]
                    if best_mse is None or mse < best_mse:
                        best_mse = mse
                        best_is_deep = is_deep[ai]

            self.assertIsNotNone(best_mse)
            self.assertTrue(best_is_deep, msg=f"Mixture {mix_id} best MSE={best_mse} was not deep")


if __name__ == "__main__":
    unittest.main()
