"""
Non-asymptotic peak error validation using direct numerical integration.

This script addresses the fundamental limitation identified in EXTENDED_THEORY_RESULTS.md:
The Taylor expansion approach assumes h → 0, but adaptive bandwidth h₁/√(n-1) gives
h ~ O(1) in practice, invalidating asymptotic assumptions.

NEW APPROACH:
Instead of Taylor expansion, we compute the variance exactly using numerical integration:

    Var[∇p̂_h(x*)] = (1/n) ∫ ||∇K_h(x* - y)||² p(y) dy

This approach:
1. ✓ Works for ANY bandwidth h (no h → 0 assumption)
2. ✓ Accounts for actual mixture geometry
3. ✓ No empirical calibration factors needed
4. ✓ Valid for both small and large bandwidth regimes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import minimize
import pandas as pd
import json
import os
from typing import Tuple, Dict, List

from estimator import (
    GaussianMixture,
    MultivariateGaussian,
    ParzenWindowEstimator,
    Plotter,
    _effective_bandwidth,
    compute_kde,
)

from validate_peak_error_formulas import (
    mixture_pdf,
    mixture_gradient,
    mixture_hessian,
    mixture_laplacian,
    find_mixture_peaks,
    estimate_kde_peak,
)


def gradient_kernel_norm_squared_2d(u: np.ndarray, h: float) -> float:
    """
    Compute ||∇K_h(u)||² for 2D Gaussian kernel.
    
    K_h(u) = (1/(2πh²)) exp(-||u||²/(2h²))
    ∇K_h(u) = -(u/h²) K_h(u)
    ||∇K_h(u)||² = (||u||²/h⁴) K_h²(u)
    
    Args:
        u: 2D displacement vector
        h: Bandwidth
    
    Returns:
        ||∇K_h(u)||²
    """
    u = np.asarray(u, dtype=float)
    u_norm_sq = np.dot(u, u)
    
    # K_h(u) = (1/(2πh²)) exp(-u²/(2h²))
    K_h = (1.0 / (2.0 * np.pi * h**2)) * np.exp(-u_norm_sq / (2.0 * h**2))
    
    # ||∇K_h||² = (u²/h⁴) K_h²
    grad_K_norm_sq = (u_norm_sq / h**4) * K_h**2
    
    return float(grad_K_norm_sq)


def compute_variance_numerical_integration(
    x_peak: np.ndarray,
    mixture: GaussianMixture,
    h: float,
    integration_limit: float = 10.0
) -> float:
    """
    Compute Var[∇p̂_h(x*)] using direct numerical integration (per sample).
    
    The variance of the gradient estimate is:
    Var[∇p̂_h(x*)] = E[||∇K_h(x* - Y)||²] where Y ~ p
    
    But this gives variance of the gradient VECTOR. For peak location error,
    we need to project onto the direction of the gradient at the peak.
    
    However, at the peak, ∇p(x*) = 0, so we use the covariance matrix:
    Cov[∇p̂_h(x*)] = ∫ ∇K_h(x*-y) ⊗ ∇K_h(x*-y) p(y) dy
    
    For the location error magnitude, we use the trace:
    Var[||∇p̂_h(x*)||²] ≈ tr(Cov[∇p̂_h(x*)])
    
    This simplifies to: ∫ ||∇K_h(x* - y)||² p(y) dy
    
    Args:
        x_peak: Peak location [x, y]
        mixture: Gaussian mixture
        h: Bandwidth (actual, not base h₁)
        integration_limit: Integration radius in standard deviations
    
    Returns:
        Trace of covariance (per sample)
    """
    x_peak = np.asarray(x_peak, dtype=float)
    
    # Define the integrand: ||∇K_h(x* - y)||² p(y)
    def integrand(y1, y2):
        y = np.array([y1, y2])
        u = x_peak - y  # Displacement from y to peak
        
        grad_K_norm_sq = gradient_kernel_norm_squared_2d(u, h)
        p_y = mixture_pdf(y, mixture)
        
        return grad_K_norm_sq * p_y
    
    # Determine integration bounds - needs to be large for large h
    limit = max(integration_limit, 3 * h)
    x_min, x_max = x_peak[0] - limit, x_peak[0] + limit
    y_min, y_max = x_peak[1] - limit, x_peak[1] + limit
    
    # Perform numerical integration
    try:
        result, error = dblquad(
            integrand,
            x_min, x_max,
            lambda x: y_min, lambda x: y_max,
            epsabs=1e-8,
            epsrel=1e-6
        )
        return float(result)
    except Exception as e:
        print(f"Integration failed: {e}")
        return np.nan


def compute_variance_monte_carlo(
    x_peak: np.ndarray,
    mixture: GaussianMixture,
    h: float,
    n_samples: int = 100000
) -> float:
    """
    Compute Var[∇p̂_h(x*)] using Monte Carlo integration.
    
    This is faster and more robust than dblquad for complex mixtures.
    
    Var[∇p̂_h(x*)] ≈ (1/N) Σᵢ ||∇K_h(x* - yᵢ)||²
    
    where yᵢ ~ p(y)
    
    Args:
        x_peak: Peak location
        mixture: Gaussian mixture  
        h: Bandwidth
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Variance per sample
    """
    x_peak = np.asarray(x_peak, dtype=float)
    
    # Sample from mixture
    y_samples = mixture.sample_points_weighted(n_samples // len(mixture._gaussians), with_pdf=False)
    
    # Compute ||∇K_h(x* - y)||² for each sample
    grad_norms_sq = []
    for y in y_samples:
        u = x_peak - y
        grad_norm_sq = gradient_kernel_norm_squared_2d(u, h)
        grad_norms_sq.append(grad_norm_sq)
    
    # Mean is the variance per sample
    variance = float(np.mean(grad_norms_sq))
    
    return variance


def predict_peak_error_numerical(
    x_peak: np.ndarray,
    mixture: GaussianMixture,
    h1: float,
    n: int,
    method: str = 'monte_carlo',
    n_mc_samples: int = 100000
) -> Dict[str, float]:
    """
    Predict peak location error using numerical integration (no Taylor expansion).
    
    The peak location error arises from solving ∇p̂_h(x̂) = 0.
    Using implicit function theorem: δx ≈ -H⁻¹ · δ(∇p̂_h)
    
    where H = Hessian at true peak = -Λ (peak sharpness matrix).
    
    Therefore:
        Var[δx] = H⁻¹ · Cov[∇p̂_h(x*)] · H⁻¹
        RMSE² = tr(Var[δx]) = tr(Λ⁻¹ · Cov[∇p̂_h(x*)] · Λ⁻¹)
    
    For isotropic kernel (Gaussian):
        Cov[∇p̂_h(x*)] = (σ²_∇/n) · I
        where σ²_∇ = ∫ ||∇K_h(x* - y)||² p(y) dy
    
    So: RMSE² = (σ²_∇/n) · tr(Λ⁻²)
    
    Args:
        x_peak: Peak location
        mixture: Gaussian mixture
        h1: Base bandwidth
        n: Total sample size
        method: 'monte_carlo' or 'integration'
        n_mc_samples: Number of MC samples if using Monte Carlo
    
    Returns:
        Dictionary with predicted RMSE and components
    """
    from validate_peak_error_formulas import mixture_hessian
    
    # Effective bandwidth
    h = _effective_bandwidth(h1, n)
    
    # Compute Hessian at peak: H = -Λ
    H = mixture_hessian(x_peak, mixture)
    Lambda = -H  # Peak sharpness matrix
    
    # Check if it's a valid peak
    eigvals = np.linalg.eigvalsh(Lambda)
    if not np.all(eigvals > 0):
        print(f"Warning: Non-positive eigenvalues at peak: {eigvals}")
        Lambda = np.eye(2) * np.mean(np.abs(eigvals))
    
    Lambda_inv = np.linalg.inv(Lambda)
    Lambda_inv_sq = Lambda_inv @ Lambda_inv
    tr_Lambda_inv_sq = np.trace(Lambda_inv_sq)
    
    # Compute variance of gradient numerically
    if method == 'monte_carlo':
        sigma_grad_sq = compute_variance_monte_carlo(x_peak, mixture, h, n_mc_samples)
    else:
        sigma_grad_sq = compute_variance_numerical_integration(x_peak, mixture, h)
    
    # For Gaussian kernel, Cov[∇p̂] ≈ (σ²_∇/n) I (approximately isotropic)
    # So: Var[δx] = (σ²_∇/n) Λ⁻²
    # And: RMSE² = tr(Var[δx]) = (σ²_∇/n) tr(Λ⁻²)
    
    variance_location = (sigma_grad_sq / n) * tr_Lambda_inv_sq
    rmse = float(np.sqrt(variance_location))
    
    return {
        'h1': h1,
        'n': n,
        'h_eff': h,
        'sigma_grad_sq': sigma_grad_sq,
        'tr_Lambda_inv_sq': tr_Lambda_inv_sq,
        'variance_location': variance_location,
        'predicted_rmse': rmse,
        'predicted_mean': float(np.sqrt(np.pi / 2.0) * rmse),  # Rayleigh distribution mean
        'method': method,
        'lambda_eigenvalues': eigvals.tolist(),
    }


def validate_peak_error_numerical(
    mixture: GaussianMixture,
    h1_values: List[float],
    n_values: List[int],
    n_trials: int = 10,
    seed: int = 42,
    method: str = 'monte_carlo'
) -> Dict:
    """
    Validate peak error predictions using numerical integration approach.
    
    This is the NON-ASYMPTOTIC version that works for finite bandwidth.
    
    Args:
        mixture: Gaussian mixture
        h1_values: Base bandwidth values
        n_values: Sample sizes (per component)
        n_trials: Number of trials per configuration
        seed: Random seed
        method: 'monte_carlo' or 'integration'
    
    Returns:
        Validation results dictionary
    """
    np.random.seed(seed)
    
    # Find true peaks
    peaks = find_mixture_peaks(mixture, n_starts=30, seed=seed)
    print(f"Found {len(peaks)} peaks in mixture")
    
    if len(peaks) == 0:
        print("Warning: No peaks found!")
        return {}
    
    # Use primary peak
    peak_densities = [mixture_pdf(pk, mixture) for pk in peaks]
    primary_peak_idx = np.argmax(peak_densities)
    true_peak = peaks[primary_peak_idx]
    print(f"Primary peak at {true_peak} with density {peak_densities[primary_peak_idx]:.6f}")
    
    # Run experiments
    results = []
    
    print(f"\nRunning {len(h1_values) * len(n_values) * n_trials} experiments...")
    print(f"Method: {method}")
    
    for n_per_comp in n_values:
        n_total = n_per_comp * len(mixture._gaussians)
        
        for h1 in h1_values:
            # Compute numerical prediction
            pred = predict_peak_error_numerical(
                true_peak, mixture, h1, n_total, method=method
            )
            
            # Run empirical trials
            peak_errors = []
            for trial in range(n_trials):
                # Sample training data
                train_xy = mixture.sample_points_weighted(n_per_comp, with_pdf=False)
                
                # Estimate peak
                est_peak = estimate_kde_peak(train_xy, h1, true_peak)
                
                # Compute error
                error = np.linalg.norm(est_peak - true_peak)
                peak_errors.append(error)
            
            # Observed statistics
            mean_error = float(np.mean(peak_errors))
            std_error = float(np.std(peak_errors))
            rmse_error = float(np.sqrt(np.mean(np.array(peak_errors)**2)))
            
            results.append({
                'n_per_comp': n_per_comp,
                'n_total': n_total,
                'h1': h1,
                'h_eff': pred['h_eff'],
                'observed_mean_error': mean_error,
                'observed_std_error': std_error,
                'observed_rmse': rmse_error,
                'predicted_rmse': pred['predicted_rmse'],
                'predicted_mean': pred['predicted_mean'],
                'rmse_ratio': rmse_error / pred['predicted_rmse'] if pred['predicted_rmse'] > 0 else np.nan,
                'mean_ratio': mean_error / pred['predicted_mean'] if pred['predicted_mean'] > 0 else np.nan,
                'sigma_grad_sq': pred['sigma_grad_sq'],
                'tr_Lambda_inv_sq': pred['tr_Lambda_inv_sq'],
                'variance_location': pred['variance_location'],
                'method': method,
            })
            
            print(f"n={n_total}, h1={h1:.1f}: obs={rmse_error:.4f}, pred={pred['predicted_rmse']:.4f}, ratio={results[-1]['rmse_ratio']:.3f}")
    
    return {
        'mixture_info': {
            'n_components': len(mixture._gaussians),
            'n_peaks': len(peaks),
            'primary_peak': true_peak.tolist(),
        },
        'method': method,
        'results': results,
    }


def compare_methods(
    mixture: GaussianMixture,
    h1_values: List[float],
    n_values: List[int],
    n_trials: int = 10,
    seed: int = 42
) -> Dict:
    """
    Compare Taylor expansion vs numerical integration approaches.
    
    Returns:
        Dictionary with results from both methods
    """
    from validate_peak_error_formulas import validate_peak_error_formula
    
    print("=" * 80)
    print("COMPARING TAYLOR EXPANSION VS NUMERICAL INTEGRATION")
    print("=" * 80)
    
    print("\n1. TAYLOR EXPANSION METHOD (Old Theory)")
    print("-" * 80)
    taylor_results = validate_peak_error_formula(
        mixture, h1_values, n_values, n_trials, seed
    )
    
    print("\n2. NUMERICAL INTEGRATION METHOD (New Theory)")
    print("-" * 80)
    numerical_results = validate_peak_error_numerical(
        mixture, h1_values, n_values, n_trials, seed, method='monte_carlo'
    )
    
    return {
        'taylor': taylor_results,
        'numerical': numerical_results,
    }


def plot_comparison(comparison_results: Dict, output_path: str = 'figures/method_comparison.jpeg'):
    """
    Create comparison plots between Taylor and numerical methods.
    """
    taylor_df = pd.DataFrame(comparison_results['taylor']['results'])
    numerical_df = pd.DataFrame(comparison_results['numerical']['results'])
    
    # Get unique h1 values
    h1_values = sorted(taylor_df['h1'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, h1 in enumerate(h1_values[:4]):  # Plot first 4 h1 values
        if i >= 4:
            break
        
        ax = axes[i // 2, i % 2]
        
        # Filter data for this h1
        taylor_h1 = taylor_df[taylor_df['h1'] == h1]
        numerical_h1 = numerical_df[numerical_df['h1'] == h1]
        
        # Plot observed
        ax.plot(taylor_h1['n_total'], taylor_h1['observed_rmse'], 
                'ko-', label='Observed', linewidth=2, markersize=8)
        
        # Plot predictions
        ax.plot(taylor_h1['n_total'], taylor_h1['predicted_rmse'],
                'b^--', label='Taylor (4-term)', linewidth=1.5, markersize=6)
        ax.plot(numerical_h1['n_total'], numerical_h1['predicted_rmse'],
                'rs--', label='Numerical', linewidth=1.5, markersize=6)
        
        ax.set_xlabel('Sample size (n)', fontsize=11)
        ax.set_ylabel('Peak location RMSE', fontsize=11)
        ax.set_title(f'h₁ = {h1:.1f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    """Run validation with numerical integration method."""
    
    print("=" * 80)
    print("NON-ASYMPTOTIC PEAK ERROR VALIDATION")
    print("Direct Numerical Integration (No Taylor Expansion)")
    print("=" * 80)
    
    # Create test mixture
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    mixture1 = GaussianMixture([g1], [1.0])
    
    # Test parameters
    h1_values = [2.0, 4.0, 7.0, 12.0]
    n_values = [20, 40, 80, 160]
    n_trials = 20
    
    # Run comparison
    comparison = compare_methods(mixture1, h1_values, n_values, n_trials)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    taylor_df = pd.DataFrame(comparison['taylor']['results'])
    numerical_df = pd.DataFrame(comparison['numerical']['results'])
    
    print("\nTAYLOR EXPANSION (4-term):")
    print(f"  Median RMSE ratio: {taylor_df['rmse_ratio'].median():.3f}")
    print(f"  Mean RMSE ratio: {taylor_df['rmse_ratio'].mean():.3f}")
    print(f"  Std RMSE ratio: {taylor_df['rmse_ratio'].std():.3f}")
    
    print("\nNUMERICAL INTEGRATION:")
    print(f"  Median RMSE ratio: {numerical_df['rmse_ratio'].median():.3f}")
    print(f"  Mean RMSE ratio: {numerical_df['rmse_ratio'].mean():.3f}")
    print(f"  Std RMSE ratio: {numerical_df['rmse_ratio'].std():.3f}")
    
    # Improvement
    taylor_median_error = abs(taylor_df['rmse_ratio'].median() - 1.0)
    numerical_median_error = abs(numerical_df['rmse_ratio'].median() - 1.0)
    improvement = (taylor_median_error - numerical_median_error) / taylor_median_error * 100
    
    print(f"\nImprovement: {improvement:+.1f}%")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    with open("results/peak_error_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    print("\nSaved results to results/peak_error_comparison.json")
    
    # Create plots
    plot_comparison(comparison)
    
    # Detailed breakdown by h1
    print("\n" + "=" * 80)
    print("BREAKDOWN BY BANDWIDTH")
    print("=" * 80)
    
    for h1 in h1_values:
        taylor_h1 = taylor_df[taylor_df['h1'] == h1]
        numerical_h1 = numerical_df[numerical_df['h1'] == h1]
        
        print(f"\nh₁ = {h1:.1f}:")
        print(f"  Taylor median ratio: {taylor_h1['rmse_ratio'].median():.3f}")
        print(f"  Numerical median ratio: {numerical_h1['rmse_ratio'].median():.3f}")
        
        if taylor_h1['rmse_ratio'].median() > 0:
            relative_improvement = (
                (taylor_h1['rmse_ratio'].median() - numerical_h1['rmse_ratio'].median()) 
                / taylor_h1['rmse_ratio'].median() * 100
            )
            print(f"  Improvement: {relative_improvement:+.1f}%")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
