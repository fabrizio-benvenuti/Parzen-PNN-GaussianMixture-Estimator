import unittest


def mlp_param_count(input_dim: int, hidden_layers: list[int], output_dim: int = 1) -> int:
    dims = [input_dim] + list(hidden_layers) + [output_dim]
    total = 0
    for a, b in zip(dims[:-1], dims[1:]):
        total += (a + 1) * b  # weights + biases
    return total


class TestComplexityParamCount(unittest.TestCase):
    def test_param_counts_match_report(self):
        """Theory: PNN inference is O(WT); W depends on architecture.

        These checks keep the report's parameter counts consistent with the actual MLP definition.
        """

        self.assertEqual(mlp_param_count(2, [20], 1), 81)
        self.assertEqual(mlp_param_count(2, [30, 20], 1), 731)


if __name__ == "__main__":
    unittest.main()
