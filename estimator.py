import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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

	def sample_points(self, num_points):
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
		self.fig = plt.figure()
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

	def show(self):
		plt.show()

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
weights1 = [1.0]
weights2 = [0.3,0.3,0.4]
weights3 = [0.2,0.2,0.2,0.2,0.2]
g1=MultivariateGaussian([1, 2],[[1.62350208,-0.13337813],[-0.13337813,0.63889251]])
g2=MultivariateGaussian([-2, -1],[[1.14822883,0.19240818],[0.19240818,1.23432651]])
g3=MultivariateGaussian([-1, 3],[[0.30198015,0.13745508],[0.13745508,1.69483031]])
g4=MultivariateGaussian([1.5, -0.5],[[0.85553671,-0.19601649],[-0.19601649,0.7507167]])
g5=MultivariateGaussian([-3, 2],[[0.42437194,-0.17066673],[-0.17066673,2.16117758]])
mixture1 = GaussianMixture([g1], weights1)
mixture2 = GaussianMixture([g1,g2,g3], weights2)
mixture3 = GaussianMixture([g1,g2,g3,g4,g5], weights3)
plotter = Plotter(-5,5,-5,5,100)
plotter2 = Plotter(-5,5,-5,5,100)
plotter3 = Plotter(-5,5,-5,5,100)
mesh1 = mixture1.get_mesh(plotter.pos)
mesh2 = mixture2.get_mesh(plotter2.pos)
mesh3 = mixture3.get_mesh(plotter3.pos)
plotter.ax.plot_surface(plotter.X, plotter.Y, mesh1, alpha=0.7, cmap='plasma')
plotter2.ax.plot_surface(plotter.X, plotter.Y, mesh2, alpha=0.7,  cmap='plasma')
plotter3.ax.plot_surface(plotter.X, plotter.Y, mesh3, alpha=0.7,  cmap='plasma')
plotter.show()