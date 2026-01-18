import unittest
import numpy as np
import torch

from estimator import ParzenNeuralNetwork, Plotter, MultivariateGaussian, GaussianMixture


class TestEstimator(unittest.TestCase):
    def test_forward_shape(self):
        model = ParzenNeuralNetwork(hidden_layers=[10], output_activation="relu")
        x = torch.randn(11, 2)
        y = model(x)
        self.assertEqual(tuple(y.shape), (11,))

    def test_training_loss_finite(self):
        # Small synthetic mixture for a lightweight training step.
        g = MultivariateGaussian([0, 0], [[1.0, 0.0], [0.0, 1.0]])
        mix = GaussianMixture([g], [1.0])
        samples = mix.sample_points_weighted(20, with_pdf=False)
        plotter = Plotter(-2, 2, -2, 2, 25)

        bandwidth = 0.3
        model = ParzenNeuralNetwork(hidden_layers=[10], output_activation="sigmoid", output_scale="auto")

        loss, eval_history, train_history = model.train_network(
            samples,
            plotter,
            bandwidth=bandwidth,
            mixture=mix,
            learning_rate=1e-3,
            epochs=10,
            boundary_points=None,
            lambda_boundary=0.0,
            verbose=False,
            loss_mode="relative",
        )
        self.assertTrue(torch.isfinite(loss).item())
        # history is optional but if present should be finite
        if len(eval_history) > 0:
            self.assertTrue(np.all(np.isfinite(np.asarray(eval_history))))
        if len(train_history) > 0:
            self.assertTrue(np.all(np.isfinite(np.asarray(train_history))))


if __name__ == "__main__":
    unittest.main()