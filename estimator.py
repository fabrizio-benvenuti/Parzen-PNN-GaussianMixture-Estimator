import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class ParzenNeuralNetwork(nn.Module):
    def __init__(self, input_dim=2, num_kernels=50, bandwidth=1.0):
        super(ParzenNeuralNetwork, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float32))
        self.kernel_centers = nn.Parameter(torch.randn(num_kernels, input_dim))
        self.kernel_weights = nn.Parameter(torch.randn(num_kernels))

    def gaussian_kernel(self, x, center):
        return torch.exp(-torch.sum((x - center) ** 2, dim=1) / (2 * self.bandwidth ** 2))

    def forward(self, x):
        pdf_values = torch.zeros(x.shape[0], device=x.device)
        for center, weight in zip(self.kernel_centers, self.kernel_weights):
            pdf_values += weight * self.gaussian_kernel(x, center)
        pdf_values = pdf_values / (self.bandwidth * np.sqrt(2 * np.pi) * len(self.kernel_weights))
        return pdf_values.clamp(min=0)  # Ensure non-negative output for PDF

    def estimate_pdf(self, points, plotter):
        points_tensor = torch.tensor(points[:, :2], dtype=torch.float32)
        grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        with torch.no_grad():
            pdf_values = self.forward(grid_tensor).numpy()
            
        pdf_mesh = pdf_values.reshape(plotter.X.shape)
        return pdf_mesh

    def train_network(self, points, log_file=None, learning_rate=0.01, epochs=500):
        points_tensor = torch.tensor(points[:, :2], dtype=torch.float32)
        targets_tensor = torch.tensor(points[:, 2], dtype=torch.float32)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(points_tensor)
            loss = criterion(outputs, targets_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                message = f"Epoch {epoch}: Loss = {loss.item()}\n"
                if log_file:
                    log_file.write(message)
                print(message, end='')

class ParzenWindowEstimator:
    def __init__(self, points, window_size=0.1):
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array with shape (n, 3).")
        self.points = points
        self.window_size = window_size
        self.kernel_volume = (window_size ** 2) * (2 * np.pi)

    def estimate_pdf(self, plotter):
        self.pdf = np.zeros(plotter.pos.shape[:2])
        num_points = self.points.shape[0]

        for point in self.points:
            diff = plotter.pos - point[:2]
            norm = np.sum((diff / self.window_size) ** 2, axis=2)
            self.pdf += np.exp(-0.5 * norm)

        self.pdf /= (num_points * self.kernel_volume)
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

    def sample_points(self, num_points: int):
        points = []
        for _ in range(num_points):
            idx = np.random.choice(len(self._gaussians), p=self._weights)
            sample = self._gaussians[idx].get_distribution().rvs()
            pdf_value = sum(weight * gaussian.get_distribution().pdf(sample)
                            for gaussian, weight in zip(self._gaussians, self._weights))
            points.append([sample[0], sample[1], pdf_value])
        return np.array(points)

class Plotter:
    def __init__(self, min_x, max_x, min_y, max_y, num_points):
        self.x = np.linspace(min_x, max_x, num_points)
        self.y = np.linspace(min_y, max_y, num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((self.X, self.Y))
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def add_surface(self, rv, color=None):
        if not hasattr(rv, 'pdf'):
            raise TypeError("Expected an object with a 'pdf' method, such as a scipy.stats.multivariate_normal object.")
        Z = rv.pdf(self.pos)
        self.ax.plot_surface(self.X, self.Y, Z, alpha=0.7, color=color)

    def add_points(self, points, color='red'):
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array with shape (n, 3).")
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

# Example usage
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
def powspace(start, stop, power, num, allow_floats: bool):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    power_array = np.power(np.linspace(start, stop, num=num), power)
    if allow_floats:
        return power_array
    else:
        return np.array([int(i) for i in power_array])

num_samples_per_gaussian = powspace(50, 200, 10, 40, False)  # 50 to 200 samples
window_sizes = powspace(0.05, 1.0, 10, 40, True)  # 0.05 to 1.0 window sizes
architectures = [
    {"hidden_layers": [10], "activation": nn.ReLU()},
    {"hidden_layers": [20, 10], "activation": nn.Tanh()},
    {"hidden_layers": [50], "activation": nn.Sigmoid()},
    {"hidden_layers": [30, 20, 10], "activation": nn.LeakyReLU(0.1)},
    {"hidden_layers": [100, 50], "activation": nn.ELU()}
]
num_kernels = np.linspace(10, 100, 10, dtype=int)  # Number of kernels
bandwidths = np.linspace(0.01, 1.0, 10)  # Bandwidths for kernels

# Initialize mixtures
mixtures = [mixture1, mixture2, mixture3]

# --- Parzen Window Evaluation ---
for mixture_idx, mixture in enumerate(mixtures):
    print(f"Processing Gaussian Mixture {mixture_idx + 1} with Parzen Window")
    errors = []
    sampled_points = []
    sampled_window = []
    for num_samples in num_samples_per_gaussian:
        samples = mixture.sample_points(num_samples)
        for window_size in window_sizes:
            parzen_estimator = ParzenWindowEstimator(samples, window_size)
            estimated_pdf = parzen_estimator.estimate_pdf(plotter)
            error = parzen_estimator.calculate_error_metrics(mixture.get_mesh(plotter.pos))
            errors.append(error['Mean Squared Error'])
            sampled_points.append(num_samples)
            sampled_window.append(window_size)
    # Create a 3D scatter plot for errors
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(sampled_points), np.array(sampled_window), np.array(errors), c='r')
    ax.set_title(f"Parzen Window Errors (Mixture {mixture_idx + 1})")
    ax.set_xlabel("Sampled Points")
    ax.set_ylabel("Window Size")
    ax.set_zlabel("Mean Squared Error")
    fig_filename = f"figures/Parzen_errors_mixture{mixture_idx + 1}.jpeg"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_filename}")

# --- Parzen Neural Network (PNN) Evaluation ---
for mixture_idx, mixture in enumerate(mixtures):
    print(f"Processing Gaussian Mixture {mixture_idx + 1} with PNN")
    for arch in architectures:
        arch_str = f"layers_{'_'.join(map(str, arch['hidden_layers']))}_act_{arch['activation'].__class__.__name__}"
        errors_nn_arr = []
        num_kernel_nn = []
        bandwidths_nn = []
        # For each architecture, we will log training output for each combination
        for num_kernels_val in num_kernels:
            for bandwidth in bandwidths:
                # Create a unique log file for this combination
                log_filename = f"logs/mixture{mixture_idx+1}_{arch_str}_nk_{num_kernels_val}_bw_{bandwidth:.3f}.txt"
                with open(log_filename, "w") as log_file:
                    # Sample points once for training
                    samples = mixture.sample_points(100)
                    pnn = ParzenNeuralNetwork(input_dim=2, num_kernels=num_kernels_val, bandwidth=bandwidth)
                    pnn.train_network(samples, log_file=log_file, learning_rate=0.01, epochs=500)
                # After training, evaluate error on a fresh sample
                samples_eval = mixture.sample_points(100)
                estimated_pdf_nn = pnn.estimate_pdf(samples_eval, plotter)
                errors_nn = np.mean((estimated_pdf_nn - mixture.get_mesh(plotter.pos)) ** 2)
                num_kernel_nn.append(num_kernels_val)
                bandwidths_nn.append(bandwidth)
                errors_nn_arr.append(errors_nn)
                print(f"Mixture {mixture_idx+1}, {arch_str}, Kernels {num_kernels_val}, Bandwidth {bandwidth:.3f}: Error = {errors_nn}")
        # Create a 3D scatter plot for PNN errors per architecture
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np.array(num_kernel_nn), np.array(bandwidths_nn), np.array(errors_nn_arr), c='green')
        ax.set_title(f"PNN Errors (Mixture {mixture_idx + 1}, Architecture {arch_str})")
        ax.set_xlabel("Number of Kernels")
        ax.set_ylabel("Bandwidth")
        ax.set_zlabel("Mean Squared Error")
        fig_filename = f"figures/PNN_errors_mixture{mixture_idx+1}_{arch_str}.jpeg"
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure: {fig_filename}")
