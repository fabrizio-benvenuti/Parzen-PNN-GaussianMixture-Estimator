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

def gen_pos_semi_definite():
    matrixSize = 2
    A = np.random.rand(matrixSize, matrixSize)
    return np.dot(A, A.T)

def create_and_display_mixtures(avgs):
    while True:
        gaussians = [MultivariateGaussian(mu, gen_pos_semi_definite()) for mu in avgs]
        covariances = [g.get_covariance() for g in gaussians]
        weights1 = [1.0]
        weights2 = np.random.rand(3)
        weights3 = np.random.rand(5)
        weights2 /= np.sum(weights2)  # Normalize
        weights3 /= np.sum(weights3)  # Normalize

        mixture1 = GaussianMixture([gaussians[0]], weights1)
        mixture2 = GaussianMixture(gaussians[:3], weights2)
        mixture3 = GaussianMixture(gaussians, weights3)

        plotter = Plotter(-3, 3, -3, 3, 100)
        plotter2 = Plotter(-3, 3, -3, 3, 100)
        plotter3 = Plotter(-3, 3, -3, 3, 100)
        mesh1 = mixture1.get_mesh(plotter.pos)
        mesh2 = mixture2.get_mesh(plotter.pos)
        mesh3 = mixture3.get_mesh(plotter.pos)

        plotter.ax.plot_surface(plotter.X, plotter.Y, mesh1, alpha=0.7, color='green')
        plotter2.ax.plot_surface(plotter.X, plotter.Y, mesh2, alpha=0.7, color='blue')
        plotter3.ax.plot_surface(plotter.X, plotter.Y, mesh3, alpha=0.7, color='red')
        plotter.show()
        plotter2.show()
        plotter3.show()

        user_input = input("Do you approve the mixtures? (yes/no): ").strip().lower()
        if user_input == 'yes':
            print("Covariance Matrices:")
            for i, cov in enumerate(covariances):
                print(f"Gaussian {i + 1} Covariance Matrix:\n{cov}\n")
            print("Weights:")
            print(f"Mixture 1 Weights: {weights1}")
            print(f"Mixture 2 Weights: {weights2}")
            print(f"Mixture 3 Weights: {weights3}")
            break
        else:
            print("Regenerating mixtures...")

# Example usage:
#averages = [[1, 2], [0, 3], [-1, 2], [-2, 1], [2, -1]]
#create_and_display_mixtures(averages)
gaussian1 = MultivariateGaussian([1, 2], [[1.64553961, 0.89438437],[0.89438437, 0.50355942]])
gaussian2 = MultivariateGaussian([1, -1], [[0.25808108 ,0.26465206],[0.26465206, 0.38207475]])

gaussian3 = MultivariateGaussian([-1, 2], [[0.93008787, 0.16909737],[0.16909737, 0.2199979 ]])
gaussian4 = MultivariateGaussian([-2, 1], [[0.88367637, 0.29129515],[0.29129515, 0.14613662]])
gaussian5 = MultivariateGaussian([-1, 2], [[0.21661693, 0.33837229],[0.33837229, 0.8277334 ]])
mixture = GaussianMixture([gaussian1], [1.0])
mixture2 = GaussianMixture([gaussian1, gaussian2, gaussian3], [0.44854643, 0.16644709, 0.38500648])
mixture3 = GaussianMixture([gaussian1, gaussian2, gaussian3, gaussian4,gaussian5], [0.31735217, 0.05439902, 0.10245388, 0.27751252, 0.24828241])

plotter = Plotter(-4, 5, -3, 5, 100)
mesh = mixture.get_mesh(plotter.pos)
plotter.ax.plot_surface(plotter.X, plotter.Y, mesh, alpha=1, cmap='plasma')
plotter2 = Plotter(-4, 5, -3, 5, 100)
mesh2 = mixture2.get_mesh(plotter.pos)
plotter2.ax.plot_surface(plotter.X, plotter.Y, mesh2, alpha=1, cmap='plasma')
plotter3 = Plotter(-4, 5, -3, 5, 100)
mesh3 = mixture3.get_mesh(plotter.pos)
plotter3.ax.plot_surface(plotter.X, plotter.Y, mesh3, alpha=1, cmap='plasma')

plotter.show()
plotter2.show()
plotter3.show()