import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class ParzenNeuralNetwork(nn.Module):
    r"""Parzen Neural Network implemented as an MLP density model on a compact 2D domain.

    This class models an *unnormalized* positive function \tilde p_\theta(x) via an MLP
    and turns it into a proper pdf on a fixed rectangular domain D by explicit normalization:

        p_\theta(x) = \tilde p_\theta(x) / Z_\theta,   Z_\theta = \int_D \tilde p_\theta(u) du

    In code, Z_\theta is approximated by a differentiable Riemann sum on the same plot grid.

    The "architecture" truly refers to the neural network structure:
      - hidden layer widths (list)
      - activation function
    """

    def __init__(self, input_dim: int = 2, hidden_layers=None, activation: nn.Module | None = None):
        super().__init__()
        if input_dim != 2:
            raise ValueError("This implementation currently assumes input_dim=2.")

        if hidden_layers is None:
            hidden_layers = [10]
        if activation is None:
            activation = nn.ReLU()

        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.eps = 1e-12

        layers = []
        prev = self.input_dim
        for width in self.hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(copy.deepcopy(self.activation))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.positive = nn.Softplus()

    @staticmethod
    def _grid_from_plotter(plotter) -> torch.Tensor:
        grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
        return torch.tensor(grid_points, dtype=torch.float32)

    @staticmethod
    def _dx_dy_from_plotter(plotter) -> tuple[float, float]:
        dx = float(plotter.x[1] - plotter.x[0]) if len(plotter.x) > 1 else 1.0
        dy = float(plotter.y[1] - plotter.y[0]) if len(plotter.y) > 1 else 1.0
        return dx, dy

    def unnormalized_density(self, x: torch.Tensor) -> torch.Tensor:
        # Positive, smooth, avoids log(0)
        out = self.net(x).squeeze(-1)
        return self.positive(out) + self.eps

    def log_Z(self, plotter) -> torch.Tensor:
        grid_tensor = self._grid_from_plotter(plotter)
        dx, dy = self._dx_dy_from_plotter(plotter)
        z = torch.sum(self.unnormalized_density(grid_tensor)) * (dx * dy)
        return torch.log(z + self.eps)

    def log_prob(self, x: torch.Tensor, plotter) -> torch.Tensor:
        log_unnorm = torch.log(self.unnormalized_density(x))
        return log_unnorm - self.log_Z(plotter)

    def estimate_pdf(self, plotter):
        grid_tensor = self._grid_from_plotter(plotter)
        with torch.no_grad():
            pdf_values = torch.exp(self.log_prob(grid_tensor, plotter)).cpu().numpy()
        return pdf_values.reshape(plotter.X.shape)

    def train_network(self, points, plotter, log_file=None, learning_rate=0.01, epochs=500):
        """Train by maximum likelihood on samples, with explicit normalization on D."""
        points_xy = np.asarray(points)[:, :2]
        points_tensor = torch.tensor(points_xy, dtype=torch.float32)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            nll = -torch.mean(self.log_prob(points_tensor, plotter))
            loss = nll

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch}: loss is NaN/Inf. Try reducing learning rate or epochs.")
                return

            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                message = f"Epoch {epoch}: Loss = {loss.item()}\n"
                if log_file:
                    log_file.write(message)
                print(message, end='')
        return loss.detach()

class ParzenWindowEstimator:
    def __init__(self, points, window_size=0.1):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Points must be a numpy array with shape (n, 2) or (n, 3).")
        self.points = points[:, :2]
        self.window_size = float(window_size)
        # For a 2D isotropic Gaussian kernel, normalization is 1/(2π h^2)
        self.kernel_norm = 1.0 / (2.0 * np.pi * self.window_size * self.window_size)

    def estimate_pdf(self, plotter):
        self.pdf = np.zeros(plotter.pos.shape[:2])
        num_points = self.points.shape[0]

        for point in self.points:
            diff = plotter.pos - point
            norm = np.sum((diff / self.window_size) ** 2, axis=2)
            self.pdf += np.exp(-0.5 * norm)

        self.pdf *= self.kernel_norm / num_points
        return self.pdf

    def calculate_error_metrics(self, original_pdf):
        if self.pdf.shape != original_pdf.shape:
            raise ValueError("The shapes of estimated_pdf and original_pdf must match.")
        mse = np.mean((self.pdf - original_pdf) ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(self.pdf - original_pdf))
        mean_abs_error = np.mean(np.abs(self.pdf - original_pdf))
        return {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Max Absolute Error': max_error,
            'Mean Absolute Error': mean_abs_error
        }

class MultivariateGaussian:
    def __init__(self, mu, covariance):
        if len(mu) != 2 or np.array(covariance).shape != (2, 2):
            raise ValueError("Mean must be a list of 2 values and covariance must be a 2x2 matrix.")
        if not np.all(np.linalg.eigvals(covariance) >= 0):
            raise ValueError("Covariance matrix must be positive semidefinite.")
        self._mu = np.array(mu)
        self._covariance = np.array(covariance)
        self._rv = multivariate_normal(self._mu, self._covariance)

    def get_mu(self):
        return self._mu

    def get_covariance(self):
        return self._covariance

    def get_distribution(self):
        return self._rv

class GaussianMixture:
    def __init__(self, gaussians, weights):
        if len(gaussians) != len(weights):
            raise ValueError("The number of Gaussian distributions must match the number of weights.")
        if not all(isinstance(g, MultivariateGaussian) for g in gaussians):
            raise TypeError("All elements in 'gaussians' must be instances of MultivariateGaussian.")
        dimensions = [g.get_mu().shape[0] for g in gaussians]
        if len(set(dimensions)) != 1:
            raise ValueError("All Gaussian distributions must have the same dimensionality.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("The sum of all weights must be 1.")
        self._gaussians = gaussians
        self._weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def get_gaussian_and_weight(self, index):
        if index < 0 or index >= len(self._gaussians):
            raise IndexError("Index out of range.")
        return self._gaussians[index], self._weights[index]

    def get_mesh(self, pos):
        total_pdf = np.zeros(pos.shape[:2])
        for gaussian, weight in zip(self._gaussians, self._weights):
            total_pdf += weight * gaussian.get_distribution().pdf(pos)
        return total_pdf

    def sample_points(self, num_points: int, with_pdf: bool = True):
        points = []
        for _ in range(num_points):
            idx = np.random.choice(len(self._gaussians), p=self._weights)
            sample = self._gaussians[idx].get_distribution().rvs()
            if with_pdf:
                pdf_value = sum(weight * gaussian.get_distribution().pdf(sample)
                                for gaussian, weight in zip(self._gaussians, self._weights))
                points.append([sample[0], sample[1], pdf_value])
            else:
                points.append([sample[0], sample[1]])
        return np.array(points, dtype=float)

class Plotter:
    def __init__(self, min_x, max_x, min_y, max_y, num_points):
        self.x = np.linspace(min_x, max_x, num_points)
        self.y = np.linspace(min_y, max_y, num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((self.X, self.Y))
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def add_surface(self, mesh, color='plasma', alpha=0.7):
        # mesh: a 2D array of pdf values that matches self.X and self.Y
        self.ax.plot_surface(self.X, self.Y, mesh, alpha=alpha, cmap=color)

    def add_points(self, points, color='red'):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Points must be a numpy array with shape (n, 2) or (n, 3).")
        if points.shape[1] == 2:
            self.ax.scatter(points[:, 0], points[:, 1], np.zeros(points.shape[0]), color=color)
        else:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)

    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(self.fig)

def gen_covariance_with_relative_widths(x_width, y_width, correlation=0.0):
    sigma_x = x_width
    sigma_y = y_width
    rho = correlation
    return np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2]
    ])

def create_and_display_mixtures(avgs):
	while True:
		gaussians = [MultivariateGaussian(mu, gen_covariance_with_relative_widths(np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), correlation=np.random.uniform(-0.5, 0.5))) for mu in avgs]
		weights1 = [1.0]
		weights2 = [0.3,0.3,0.4]
		weights3 = [0.2,0.2,0.2,0.2,0.2]

		mixture1 = GaussianMixture([gaussians[0]], weights1)
		mixture2 = GaussianMixture(gaussians[:3], weights2)
		mixture3 = GaussianMixture(gaussians, weights3)

		plotter = Plotter(-5,5,-5,5,100)
		plotter2 = Plotter(-5,5,-5,5,100)
		plotter3 = Plotter(-5,5,-5,5,100)
		mesh1 = mixture1.get_mesh(plotter.pos)
		mesh2 = mixture2.get_mesh(plotter.pos)
		mesh3 = mixture3.get_mesh(plotter.pos)

		plotter.ax.plot_surface(plotter.X, plotter.Y, mesh1, alpha=0.7, cmap='plasma')
		plotter2.ax.plot_surface(plotter.X, plotter.Y, mesh2, alpha=0.7,  cmap='plasma')
		plotter3.ax.plot_surface(plotter.X, plotter.Y, mesh3, alpha=0.7,  cmap='plasma')
		plotter.show()

		user_input = input("Do you approve the mixtures? (yes/no): ").strip().lower()
		if user_input == 'yes':
			print("Covariance Matrices and Weights:")
			print(f"Mixture 1 Weights: {weights1}")
			print(f"Gaussian 1 Covariance Matrix:\n{gaussians[0].get_covariance()}\n")

			print(f"Mixture 2 Weights: {weights2}")
			for i, gaussian in enumerate(gaussians[:3], start=1):
				print(f"Gaussian {i} Covariance Matrix:\n{gaussian.get_covariance()}\n")

			print(f"Mixture 3 Weights: {weights3}")
			for i, gaussian in enumerate(gaussians, start=1):
				print(f"Gaussian {i} Covariance Matrix:\n{gaussian.get_covariance()}\n")
			break
		else:
			print("Regenerating mixtures...")

def powspace(start, stop, power, num, allow_floats: bool):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    power_array = np.power(np.linspace(start, stop, num=num), power)
    if allow_floats:
        return power_array
    else:
        return np.array([int(i) for i in power_array])


def main():
    torch.autograd.set_detect_anomaly(True)

    weights1 = [1.0]
    weights2 = [0.3, 0.3, 0.4]
    weights3 = [0.2, 0.2, 0.2, 0.2, 0.2]
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    g2 = MultivariateGaussian([-2, -1], [[1.14822883, 0.19240818], [0.19240818, 1.23432651]])
    g3 = MultivariateGaussian([-1, 3], [[0.30198015, 0.13745508], [0.13745508, 1.69483031]])
    g4 = MultivariateGaussian([1.5, -0.5], [[0.85553671, -0.19601649], [-0.19601649, 0.7507167]])
    g5 = MultivariateGaussian([-3, 2], [[0.42437194, -0.17066673], [-0.17066673, 2.16117758]])
    mixture1 = GaussianMixture([g1], weights1)
    mixture2 = GaussianMixture([g1, g2, g3], weights2)
    mixture3 = GaussianMixture([g1, g2, g3, g4, g5], weights3)
    plotter = Plotter(-5, 5, -5, 5, 100)

    # Define parameter ranges
    num_samples_per_gaussian = powspace(50, 200, 10, 40, False)  # 50 to 200 samples
    window_sizes = powspace(0.05, 1.0, 10, 40, True)  # 0.05 to 1.0 window sizes
    # Five MLP architectures (true neural network architectures)
    architectures = [
        {"hidden_layers": [10], "activation": nn.ReLU()},
        {"hidden_layers": [20, 10], "activation": nn.Tanh()},
        {"hidden_layers": [50], "activation": nn.Sigmoid()},
        {"hidden_layers": [30, 20, 10], "activation": nn.LeakyReLU(0.1)},
        {"hidden_layers": [100, 50], "activation": nn.ELU()},
    ]

    def arch_label(arch: dict) -> str:
        layers = "-".join(str(x) for x in arch["hidden_layers"])
        act = arch["activation"].__class__.__name__
        return f"layers_{layers}_act_{act}"

    mixtures = [mixture1, mixture2, mixture3]

    # --------------------------
    # Parzen Window Evaluation (Gaussian KDE)
    for mixture_idx, mixture in enumerate(mixtures):
        print(f"Processing Gaussian Mixture {mixture_idx + 1} with Parzen Window")
        errors = []
        sampled_points = []
        sampled_window = []

        for num_samples in num_samples_per_gaussian:
            samples_xy = mixture.sample_points(num_samples, with_pdf=False)
            for window_size in window_sizes:
                parzen_estimator = ParzenWindowEstimator(samples_xy, window_size)
                _ = parzen_estimator.estimate_pdf(plotter)
                error = parzen_estimator.calculate_error_metrics(mixture.get_mesh(plotter.pos))
                errors.append(error['Mean Squared Error'])
                sampled_points.append(num_samples)
                sampled_window.append(window_size)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np.array(sampled_points), np.array(sampled_window), np.array(errors), c='r')
        ax.set_title(f"Parzen Window Errors (Mixture {mixture_idx + 1})")
        ax.set_xlabel("Sampled Points")
        ax.set_ylabel("Window Size")
        ax.set_zlabel("Mean Squared Error")
        error_fig_filename = f"figures/Parzen_errors_mixture{mixture_idx + 1}.jpeg"
        plt.savefig(error_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved error figure: {error_fig_filename}")

        best_idx = int(np.argmin(errors))
        best_num_samples = sampled_points[best_idx]
        best_window_size = sampled_window[best_idx]
        print(f"Best Parzen parameters for mixture {mixture_idx+1}: samples = {best_num_samples}, window = {best_window_size}")

        samples_best = mixture.sample_points(best_num_samples, with_pdf=False)
        parzen_best = ParzenWindowEstimator(samples_best, best_window_size)
        estimated_pdf_best = parzen_best.estimate_pdf(plotter)
        real_pdf = mixture.get_mesh(plotter.pos)

        fig_overlay = plt.figure(figsize=(16, 9))
        ax_overlay = fig_overlay.add_subplot(projection='3d')
        ax_overlay.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.5, cmap='viridis')
        ax_overlay.plot_surface(plotter.X, plotter.Y, estimated_pdf_best, alpha=0.5, cmap='plasma')
        ax_overlay.set_title(
            f"Parzen Overlay (Mixture {mixture_idx+1})\nSamples = {best_num_samples}, Window = {best_window_size:.3f}"
        )
        overlay_fig_filename = f"figures/Parzen_overlay_mixture{mixture_idx + 1}.jpeg"
        plt.savefig(overlay_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_overlay)
        print(f"Saved overlay figure: {overlay_fig_filename}")

    # --------------------------
    # Parzen Neural Network (PNN) Evaluation
    for mixture_idx, mixture in enumerate(mixtures):
        print(f"Processing Gaussian Mixture {mixture_idx + 1} with PNN")

        errors_nn_arr = []
        arch_labels = []

        # Training uses ONLY sample locations (x,y) via maximum likelihood.
        samples_xy = mixture.sample_points(100, with_pdf=False)
        for arch in architectures:
            label = arch_label(arch)
            log_filename = f"logs/mixture{mixture_idx+1}_{label}.txt"
            with open(log_filename, "w") as log_file:
                pnn = ParzenNeuralNetwork(
                    input_dim=2,
                    hidden_layers=arch["hidden_layers"],
                    activation=arch["activation"],
                )
                pnn.train_network(samples_xy, plotter, log_file=log_file, learning_rate=0.01, epochs=500)

            estimated_pdf_nn = pnn.estimate_pdf(plotter)
            error_nn = float(np.mean((estimated_pdf_nn - mixture.get_mesh(plotter.pos)) ** 2))
            arch_labels.append(label)
            errors_nn_arr.append(error_nn)
            print(f"Mixture {mixture_idx+1}, {label}: Error = {error_nn}")

        # Simple error plot across architectures
        fig_nn = plt.figure(figsize=(16, 9))
        ax_nn = fig_nn.add_subplot(111)
        x = np.arange(len(arch_labels))
        ax_nn.bar(x, np.array(errors_nn_arr), color='green')
        ax_nn.set_xticks(x)
        ax_nn.set_xticklabels(arch_labels, rotation=30, ha='right')
        ax_nn.set_title(f"PNN Errors (Mixture {mixture_idx + 1}, architectures)")
        ax_nn.set_xlabel("Architecture")
        ax_nn.set_ylabel("Mean Squared Error")
        nn_error_fig_filename = f"figures/PNN_errors_mixture{mixture_idx+1}_architectures.jpeg"
        plt.tight_layout()
        plt.savefig(nn_error_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_nn)
        print(f"Saved PNN error figure: {nn_error_fig_filename}")

        best_idx_nn = int(np.argmin(errors_nn_arr))
        best_arch_label = arch_labels[best_idx_nn]
        best_arch = architectures[best_idx_nn]
        print(f"Best PNN parameters for mixture {mixture_idx+1}: arch = {best_arch_label}")

        # Retrain best architecture for overlay
        samples_best_nn = mixture.sample_points(100, with_pdf=False)
        pnn_best = ParzenNeuralNetwork(
            input_dim=2,
            hidden_layers=best_arch["hidden_layers"],
            activation=best_arch["activation"],
        )
        pnn_best.train_network(samples_best_nn, plotter, learning_rate=0.01, epochs=500)
        estimated_pdf_nn_best = pnn_best.estimate_pdf(plotter)
        real_pdf = mixture.get_mesh(plotter.pos)

        fig_nn_overlay = plt.figure(figsize=(16, 9))
        ax_nn_overlay = fig_nn_overlay.add_subplot(projection='3d')
        ax_nn_overlay.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.5, cmap='viridis')
        ax_nn_overlay.plot_surface(plotter.X, plotter.Y, estimated_pdf_nn_best, alpha=0.5, cmap='plasma')
        ax_nn_overlay.set_title(
            f"PNN Overlay (Mixture {mixture_idx+1})\nArch = {best_arch_label}"
        )
        nn_overlay_fig_filename = f"figures/PNN_overlay_mixture{mixture_idx+1}_{best_arch_label}.jpeg"
        plt.savefig(nn_overlay_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_nn_overlay)
        print(f"Saved PNN overlay figure: {nn_overlay_fig_filename}")


if __name__ == "__main__":
    main()
