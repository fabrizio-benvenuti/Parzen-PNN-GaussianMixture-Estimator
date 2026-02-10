"""
Deep dive into the variance formula derivation.

The key claim in Lemma 12.3 is:

    Cov[∇δp(x*)] = (1/n) * Var[∇K_h(x* - X)]
                 = (C_∇K)/(nh⁴) * p(x*) * I

Let's verify this step-by-step.
"""

import numpy as np
from scipy.stats import multivariate_normal

def test_gradient_variance_empirically(n=100, h=1.0, n_trials=1000):
    """
    Empirically compute the variance of ∇δp at a point using Monte Carlo.
    
    We'll use a simple case: true density is N(0, I), estimate at x* = 0.
    """
    p_true = lambda x: multivariate_normal.pdf(x, mean=[0, 0], cov=np.eye(2))
    
    # Gradient of Gaussian kernel
    def grad_K_h(u):
        K = multivariate_normal.pdf(u, mean=[0, 0], cov=h**2 * np.eye(2))
        return -(u / h**2) * K
    
    # Collect gradient estimates
    grad_estimates = []
    
    for _ in range(n_trials):
        # Sample n points from true distribution
        X = np.random.multivariate_normal([0, 0], np.eye(2), size=n)
        
        # Estimate gradient at x* = 0
        grad_sum = np.zeros(2)
        for x_i in X:
            grad_sum += grad_K_h(0 - x_i)
        
        grad_est = grad_sum / n
        grad_estimates.append(grad_est)
    
    grad_estimates = np.array(grad_estimates)
    
    # Compute empirical covariance
    cov_empirical = np.cov(grad_estimates.T)
    
    # Expected value (should be near zero at peak)
    mean_empirical = np.mean(grad_estimates, axis=0)
    
    # Theoretical prediction from formula
    # Cov[∇δp] = (C_∇K)/(nh⁴) * p(x*) * I
    C_grad_K = 1.0 / (4.0 * np.pi)  # Corrected constant
    p_at_peak = p_true([0, 0])
    cov_theoretical = (C_grad_K / (n * h**4)) * p_at_peak * np.eye(2)
    
    print(f"Test with n={n}, h={h}:")
    print(f"  Mean (should be ~0): {mean_empirical}")
    print(f"  Empirical Cov diagonal: {np.diag(cov_empirical)}")
    print(f"  Theoretical Cov diagonal: {np.diag(cov_theoretical)}")
    print(f"  Ratio: {np.diag(cov_empirical) / np.diag(cov_theoretical)}")
    print()
    
    return cov_empirical, cov_theoretical

def test_peak_displacement_empirically(n=100, h1=4.0, n_trials=100):
    """
    Test the full peak displacement formula empirically.
    
    For a single Gaussian, the peak is at the mean.
    We'll see if the observed peak displacement matches the theory.
    """
    from estimator import GaussianMixture, MultivariateGaussian, _effective_bandwidth
    
    # Single Gaussian mixture
    g1 = MultivariateGaussian([0, 0], np.eye(2))
    mixture = GaussianMixture([g1], [1.0])
    true_peak = np.array([0, 0])
    
    h = _effective_bandwidth(h1, n)
    
    # For single Gaussian at origin:
    # β² = 0 (by symmetry)
    # σ² = C_∇K * p(0) * tr(Λ⁻²)
    # where Λ⁻¹ = Σ (for Gaussian with Σ = I)
    # so tr(Λ⁻²) = tr(I²) = 2
    
    C_grad_K = 1.0 / (4.0 * np.pi)
    p_at_peak = 1.0 / (2.0 * np.pi)  # N(0, I) density at origin
    tr_Lambda_inv_sq = 2.0  # tr(I²) = 2
    
    beta_sq = 0.0
    sigma_sq = C_grad_K * p_at_peak * tr_Lambda_inv_sq
    
    # Theoretical prediction
    bias_term = beta_sq * h1**4 / (4 * (n - 1)**2)
    var_term = sigma_sq * (n - 1)**2 / (n * h1**4)
    pred_rmse = np.sqrt(bias_term + var_term)
    
    print(f"\nPeak displacement test with n={n}, h1={h1}, h_eff={h:.4f}:")
    print(f"  σ² = {sigma_sq:.6f}")
    print(f"  Predicted RMSE = {pred_rmse:.6f}")
    print(f"  Variance term = {var_term:.6f}")
    
    # Now test empirically by sampling
    peak_errors = []
    
    for _ in range(n_trials):
        # Sample n points
        X = mixture.sample_points_weighted(n, with_pdf=False)
        
        # Estimate peak using KDE
        def neg_kde(x):
            kde_val = 0
            for x_i in X:
                u = x - x_i
                K = np.exp(-0.5 * np.dot(u, u) / h**2) / (2 * np.pi * h**2)
                kde_val += K
            return -kde_val / n
        
        from scipy.optimize import minimize
        result = minimize(neg_kde, [0, 0], method='BFGS')
        est_peak = result.x
        
        error = np.linalg.norm(est_peak - true_peak)
        peak_errors.append(error)
    
    obs_rmse = np.sqrt(np.mean(np.array(peak_errors)**2))
    obs_mean = np.mean(peak_errors)
    
    print(f"  Observed RMSE = {obs_rmse:.6f}")
    print(f"  Observed Mean = {obs_mean:.6f}")
    print(f"  Ratio (obs/pred) = {obs_rmse / pred_rmse:.3f}")
    
    return obs_rmse, pred_rmse

if __name__ == "__main__":
    print("=" * 80)
    print("EMPIRICAL VALIDATION OF VARIANCE FORMULA")
    print("=" * 80)
    print()
    
    print("Test 1: Gradient variance at peak")
    print("-" * 80)
    test_gradient_variance_empirically(n=50, h=1.0, n_trials=5000)
    test_gradient_variance_empirically(n=100, h=1.0, n_trials=5000)
    test_gradient_variance_empirically(n=100, h=0.5, n_trials=5000)
    
    print("=" * 80)
    print("Test 2: Full peak displacement")
    print("-" * 80)
    test_peak_displacement_empirically(n=50, h1=4.0, n_trials=100)
    test_peak_displacement_empirically(n=100, h1=4.0, n_trials=100)
    test_peak_displacement_empirically(n=100, h1=2.0, n_trials=100)
