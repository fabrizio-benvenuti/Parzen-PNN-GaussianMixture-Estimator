import unittest
import torch
from estimator import PNN  # Assuming PNN is the class for your estimator

class TestEstimator(unittest.TestCase):
    def setUp(self):
        self.model = PNN()
        self.samples = torch.randn(10, 5)  # Example input
        self.targets = torch.randn(10, 5)  # Example target

    def test_training_loss_not_nan(self):
        log_file = None  # Replace with actual log file if needed
        learning_rate = 0.01
        epochs = 10
        
        for epoch in range(epochs):
            loss = self.model.train_network(self.samples, log_file=log_file, learning_rate=learning_rate, epochs=1)
            self.assertFalse(torch.isnan(loss).any(), f"Loss is NaN at epoch {epoch}")

    def test_model_output_shape(self):
        outputs = self.model.forward(self.samples)
        self.assertEqual(outputs.shape, self.targets.shape, "Output shape does not match target shape")

if __name__ == '__main__':
    unittest.main()