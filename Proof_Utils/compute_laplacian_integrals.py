#!/usr/bin/env python3
"""
Compute exact Laplacian integrals for Gaussian mixture density estimation analysis.

For a 2D Gaussian mixture p(x) = Σ w_k N(x; μ_k, Σ_k), this script computes:
- ∫(Δp)² dx  (curvature integral for MSE)
- ∫(Δp)²/p · p dx  (curvature integral for NLL)
- ρ = ratio of the two integrals
- (2ρ)^(1/6) = predicted bandwidth ratio from curvature alone

where Δp = ∂²p/∂x₁² + ∂²p/∂x₂² is the Laplacian of the density.
"""

import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal
import sys


class MultivariateGaussian:
    def __init__(self, mu, covariance):
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
        self._gaussians = gaussians
        self._weights = np.array(weights) / np.sum(weights)

    def pdf(self, x):
        """Evaluate mixture density at point x."""
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        for gaussian, weight in zip(self._gaussians, self._weights):
            result += weight * gaussian.get_distribution().pdf(x)
        return result if x.shape[0] > 1 else result[0]

    def laplacian(self, x):
        """
        Compute Laplacian Δp = ∂²p/∂x₁² + ∂²p/∂x₂² at point x.
        
        For a single Gaussian N(x; μ, Σ), the Laplacian is:
        Δp = p(x) · [(x-μ)ᵀ Σ⁻¹ H Σ⁻¹ (x-μ) - tr(Σ⁻¹ H) - tr(Σ⁻¹)²]
        
        where H is the Hessian of the Gaussian (before multiplication by p).
        
        For 2D, we can compute the second derivatives directly:
        ∂²p/∂x₁² = p(x) · [Σ⁻¹₁₁(Σ⁻¹₁₁(x₁-μ₁)² + Σ⁻¹₁₂(x₁-μ₁)(x₂-μ₂) - 1) + 
                           Σ⁻¹₁₂(Σ⁻¹₁₁(x₁-μ₁)(x₂-μ₂) + Σ⁻¹₁₂(x₂-μ₂)² - 0)]
        
        Simplified approach: use the formula
        ∂²/∂xᵢ² [p(x)] = p(x) · [(Σ⁻¹ᵢᵢ)² (xᵢ-μᵢ)² + (Σ⁻¹ᵢⱼ)² (xⱼ-μⱼ)² 
                                   + 2Σ⁻¹ᵢᵢΣ⁻¹ᵢⱼ(xᵢ-μᵢ)(xⱼ-μⱼ) - Σ⁻¹ᵢᵢ]
        """
        x = np.atleast_1d(x)
        laplacian = 0.0
        
        for gaussian, weight in zip(self._gaussians, self._weights):
            mu = gaussian.get_mu()
            Sigma = gaussian.get_covariance()
            Sigma_inv = np.linalg.inv(Sigma)
            
            # Evaluate Gaussian density
            p_x = gaussian.get_distribution().pdf(x)
            
            # Centered coordinates
            dx = x - mu
            
            # Compute ∂²p/∂x₁²
            d2p_dx1_sq = p_x * (
                (Sigma_inv[0, 0]**2) * dx[0]**2
                + (Sigma_inv[0, 1]**2) * dx[1]**2
                + 2 * Sigma_inv[0, 0] * Sigma_inv[0, 1] * dx[0] * dx[1]
                - Sigma_inv[0, 0]
            )
            
            # Compute ∂²p/∂x₂²
            d2p_dx2_sq = p_x * (
                (Sigma_inv[1, 0]**2) * dx[0]**2
                + (Sigma_inv[1, 1]**2) * dx[1]**2
                + 2 * Sigma_inv[1, 0] * Sigma_inv[1, 1] * dx[0] * dx[1]
                - Sigma_inv[1, 1]
            )
            
            # Laplacian for this component
            laplacian += weight * (d2p_dx1_sq + d2p_dx2_sq)
        
        return laplacian


def compute_integrals_monte_carlo(mixture, n_samples=10_000_000):
    """
    Compute curvature integrals using Monte Carlo sampling.
    
    Returns:
        integral_Delta_p_sq: ∫(Δp)² dx
        integral_Delta_p_sq_over_p: ∫(Δp)²/p · p dx = ∫(Δp)²/p dx (weighted)
        rho: ratio of the two integrals
    """
    print(f"\nMonte Carlo integration with {n_samples:,} samples...")
    
    # Sample from the mixture
    samples = []
    for _ in range(n_samples):
        # Choose component according to weights
        k = np.random.choice(len(mixture._gaussians), p=mixture._weights)
        # Sample from chosen component
        sample = mixture._gaussians[k].get_distribution().rvs()
        samples.append(sample)
    
    samples = np.array(samples)
    
    # Compute Laplacian at each sample
    print("Computing Laplacian at sample points...")
    laplacians = np.array([mixture.laplacian(x) for x in samples])
    
    # Compute p(x) at each sample
    print("Computing density at sample points...")
    densities = np.array([mixture.pdf(x) for x in samples])
    
    # Monte Carlo estimates
    # ∫(Δp)² dx ≈ E_x~p[(Δp(x))²/p(x)] = mean((Δp)²/p)
    integral_Delta_p_sq = np.mean(laplacians**2 / densities)
    
    # ∫(Δp)²/p · p dx = ∫(Δp)²/p dx ≈ E_x~p[(Δp(x))²/p(x)²] = mean((Δp)²/p²)
    integral_Delta_p_sq_over_p = np.mean(laplacians**2 / densities**2)
    
    # Actually, let me reconsider the formula:
    # For ∫(Δp)² dx, we need to integrate over all space
    # Using importance sampling with proposal q = p:
    # ∫(Δp)² dx = ∫(Δp)²/p · p dx = E_x~p[(Δp)²/p]
    
    # For ∫(Δp)²/p · p dx = ∫(Δp)²/p dx, this is the same as above but weighted differently
    # Actually this integral weights by 1/p, which amplifies low-density regions
    
    # Let me use numerical integration instead for accuracy
    return integral_Delta_p_sq, integral_Delta_p_sq_over_p


def compute_integrals_numerical(mixture, bounds_scale=4.0, n_grid=200):
    """
    Compute curvature integrals using numerical quadrature on a grid.
    
    Args:
        bounds_scale: Extend grid to ±bounds_scale standard deviations from mean
        n_grid: Number of grid points per dimension
    
    Returns:
        integral_Delta_p_sq: ∫(Δp)² dx
        integral_Delta_p_sq_over_p: ∫(Δp)²/p · p dx
        rho: ratio of the two integrals
    """
    print(f"\nNumerical integration with {n_grid}×{n_grid} grid...")
    
    # Determine integration bounds based on mixture means and covariances
    mus = np.array([g.get_mu() for g in mixture._gaussians])
    mean_center = np.mean(mus, axis=0)
    
    # Find maximum spread
    max_spread = 0.0
    for g in mixture._gaussians:
        Sigma = g.get_covariance()
        eigenvals = np.linalg.eigvalsh(Sigma)
        max_std = np.sqrt(np.max(eigenvals))
        mu = g.get_mu()
        spread = np.linalg.norm(mu - mean_center) + bounds_scale * max_std
        max_spread = max(max_spread, spread)
    
    # Create grid
    x1_range = mean_center[0] + np.linspace(-max_spread, max_spread, n_grid)
    x2_range = mean_center[1] + np.linspace(-max_spread, max_spread, n_grid)
    dx1 = x1_range[1] - x1_range[0]
    dx2 = x2_range[1] - x2_range[0]
    dA = dx1 * dx2
    
    print(f"Integration bounds: x1 ∈ [{x1_range[0]:.2f}, {x1_range[-1]:.2f}], "
          f"x2 ∈ [{x2_range[0]:.2f}, {x2_range[-1]:.2f}]")
    
    # Evaluate on grid
    print("Evaluating density and Laplacian on grid...")
    integral_Delta_p_sq = 0.0
    integral_Delta_p_sq_over_p = 0.0
    
    for i, x1 in enumerate(x1_range):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_grid}")
        for x2 in x2_range:
            x = np.array([x1, x2])
            
            p_x = mixture.pdf(x)
            if p_x < 1e-300:  # Avoid numerical issues
                continue
            
            Delta_p_x = mixture.laplacian(x)
            
            # ∫(Δp)² dx
            integral_Delta_p_sq += (Delta_p_x**2) * dA
            
            # ∫(Δp)²/p · p dx = ∫(Δp)²/p dx
            integral_Delta_p_sq_over_p += (Delta_p_x**2 / p_x) * dA
    
    print("Integration complete.")
    return integral_Delta_p_sq, integral_Delta_p_sq_over_p


def analyze_mixture(mixture, name):
    """Analyze a single mixture and print results."""
    print(f"\n{'='*70}")
    print(f"ANALYZING {name}")
    print(f"{'='*70}")
    
    # Show mixture structure
    n_components = len(mixture._gaussians)
    print(f"\nMixture structure: {n_components} Gaussian component(s)")
    print(f"Weights: {mixture._weights}")
    for i, g in enumerate(mixture._gaussians):
        print(f"  Component {i+1}: μ = {g.get_mu()}, Σ diagonal = {np.diag(g.get_covariance())}")
    
    # Compute integrals using numerical integration
    I_Delta_p_sq, I_Delta_p_sq_over_p = compute_integrals_numerical(
        mixture, bounds_scale=5.0, n_grid=300
    )
    
    # Compute rho
    rho = I_Delta_p_sq / I_Delta_p_sq_over_p if I_Delta_p_sq_over_p > 0 else 0.0
    
    # Compute predicted ratio
    predicted_ratio = (2 * rho) ** (1/6)
    
    # Print results
    print(f"\n{'Results:':-^70}")
    print(f"∫(Δp)² dx                    = {I_Delta_p_sq:.6e}")
    print(f"∫(Δp)²/p dx                  = {I_Delta_p_sq_over_p:.6e}")
    print(f"ρ = ratio                    = {rho:.6f}")
    print(f"(2ρ)^(1/6) [curvature only] = {predicted_ratio:.6f}")
    print(f"{'-'*70}")
    
    return {
        'name': name,
        'n_components': n_components,
        'I_Delta_p_sq': I_Delta_p_sq,
        'I_Delta_p_sq_over_p': I_Delta_p_sq_over_p,
        'rho': rho,
        'predicted_ratio_curvature': predicted_ratio,
    }


def main():
    """Compute Laplacian integrals for all three mixtures."""
    
    # Define the three mixtures exactly as in estimator.py
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
    
    results = []
    results.append(analyze_mixture(mixture1, "Mixture 1 (1 component, unimodal)"))
    results.append(analyze_mixture(mixture2, "Mixture 2 (3 components, trimodal)"))
    results.append(analyze_mixture(mixture3, "Mixture 3 (5 components, 5-modal)"))
    
    # Print summary
    print(f"\n\n{'='*70}")
    print(f"{'SUMMARY TABLE':^70}")
    print(f"{'='*70}")
    print(f"{'Mixture':<15} {'Components':<12} {'ρ':<12} {'(2ρ)^(1/6)':<12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['name']:<15} {r['n_components']:<12} {r['rho']:<12.6f} {r['predicted_ratio_curvature']:<12.6f}")
    print(f"{'='*70}")
    
    # Export to JSON for easy integration into LaTeX
    import json
    output_file = 'results/laplacian_integrals.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
