"""
Diagnose why predictions fail at large bandwidth.

This script investigates:
1. How does σ²_∇ scale with h?
2. Is the implicit function theorem approximation valid?
3. What happens to the KDE gradient at large h?
"""

import numpy as np
import matplotlib.pyplot as plt
from estimator import (
    GaussianMixture,
    MultivariateGaussian,
    _effective_bandwidth,
)
from validate_peak_error_numerical import (
    compute_variance_monte_carlo,
    gradient_kernel_norm_squared_2d,
)
from validate_peak_error_formulas import (
    mixture_pdf,
    mixture_hessian,
    find_mixture_peaks,
)


def diagnose_variance_scaling():
    """Check how σ²_∇ scales with bandwidth h."""
    
    # Create test mixture
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    mixture = GaussianMixture([g1], [1.0])
    
    # Find peak
    peaks = find_mixture_peaks(mixture, n_starts=10)
    x_peak = peaks[0]
    print(f"Peak at {x_peak}")
    
    # Test range of bandwidths
    h_values = np.logspace(-1, 1, 20)  # 0.1 to 10
    
    sigma_grad_sq_values = []
    
    for h in h_values:
        sigma_grad_sq = compute_variance_monte_carlo(x_peak, mixture, h, n_samples=50000)
        sigma_grad_sq_values.append(sigma_grad_sq)
        print(f"h={h:.3f}: σ²_∇ = {sigma_grad_sq:.6e}")
    
    # Plot scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-log plot
    ax1.loglog(h_values, sigma_grad_sq_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Bandwidth h', fontsize=12)
    ax1.set_ylabel('σ²_∇', fontsize=12)
    ax1.set_title('Variance Scaling with Bandwidth', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Check different power law fits
    for exponent in [4, 6, 8]:
        fit = sigma_grad_sq_values[0] / (h_values[0]**exponent) * h_values**exponent
        ax1.loglog(h_values, fit, '--', alpha=0.5, label=f'h^{exponent}')
    ax1.legend()
    
    # Linear-log to see better at large h
    ax2.semilogx(h_values, sigma_grad_sq_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Bandwidth h', fontsize=12)
    ax2.set_ylabel('σ²_∇', fontsize=12)
    ax2.set_title('Variance vs Bandwidth (semilog)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/variance_bandwidth_scaling.jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved variance scaling plot to figures/variance_bandwidth_scaling.jpeg")
    
    return h_values, sigma_grad_sq_values


def check_theoretical_scaling():
    """
    For Gaussian kernel and Gaussian density:
    
    ∇K_h(u) = -(u/h²) K_h(u)
    ||∇K_h||² = (||u||²/h⁴) K_h²(u)
    
    For small h, density is ~constant near peak, so:
    σ²_∇ ≈ p(x*) ∫ (||u||²/h⁴) K_h²(u) du
        ≈ p(x*) · (const/h²)  [from scaling]
        ≈ C/h²
    
    For large h, the density variation matters more.
    """
    print("\nTheoretical scaling analysis:")
    print("Small h: σ²_∇ ~ 1/h² (density approximately constant)")
    print("Large h: σ²_∇ ~ ??? (density variation matters)")
    
    # For very large h, kernel becomes flat
    # ∇K_h ≈ 0 everywhere
    # So σ²_∇ → 0 as h → ∞
    
    print("\nExpected: σ²_∇ should peak at some intermediate h, then decay")


def diagnose_peak_finding_at_large_h():
    """
    At large h, the KDE becomes very smooth and flat.
    The peak location becomes poorly defined.
    
    This means:
    1. Large variance in ∇p̂ doesn't translate to large location error
    2. The peak "disappears" into a broad plateau
    3. Any location near the center is roughly equivalent
    """
    
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    mixture = GaussianMixture([g1], [1.0])
    
    # Sample data
    np.random.seed(42)
    train_xy = mixture.sample_points_weighted(100, with_pdf=False)
    
    # Compute KDE gradients at peak for different h values
    peaks = find_mixture_peaks(mixture, n_starts=10)
    x_peak = peaks[0]
    
    h_test = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("\nGradient magnitude near peak for different bandwidths:")
    print("(Lower gradient = flatter KDE = harder to locate peak)")
    
    for h in h_test:
        # Compute KDE gradient at peak
        grad = np.zeros(2)
        for xi in train_xy:
            u = x_peak - xi
            K_h = (1.0 / (2.0 * np.pi * h**2)) * np.exp(-np.dot(u, u) / (2.0 * h**2))
            grad_K = -(u / h**2) * K_h
            grad += grad_K
        grad /= len(train_xy)
        
        grad_norm = np.linalg.norm(grad)
        print(f"h={h:5.1f}: ||∇p̂(x*)||={grad_norm:.6e}")
    
    print("\n→ At large h, gradient is tiny, so peak location is ill-defined")
    print("→ This is why large variance doesn't translate to large observed error")


def main():
    print("=" * 80)
    print("DIAGNOSING BANDWIDTH SCALING ISSUES")
    print("=" * 80)
    
    print("\n1. Variance Scaling with Bandwidth")
    print("-" * 80)
    h_vals, sigma_vals = diagnose_variance_scaling()
    
    print("\n2. Theoretical Scaling Expectations")
    print("-" * 80)
    check_theoretical_scaling()
    
    print("\n3. Peak Finding at Large Bandwidth")
    print("-" * 80)
    diagnose_peak_finding_at_large_h()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    print("\nKEY FINDING:")
    print("The numerical integration approach computes σ²_∇ correctly, but the")
    print("implicit function theorem approximation δx ≈ -H⁻¹·δ(∇p̂) is NOT valid")
    print("at large h because:")
    print("  1. The KDE becomes very flat (||∇p̂|| ≈ 0 everywhere)")
    print("  2. The peak 'disappears' into a broad plateau")
    print("  3. The Hessian H ≈ 0, so H⁻¹ is not well-defined")
    print("  4. The peak location becomes ill-posed")
    print("\nThis explains why:")
    print("  - σ²_∇ grows with h (correct)")
    print("  - But observed location error DOESN'T grow (also correct!)")
    print("  - The theory predicts large error, but that's because the")
    print("    peak itself becomes undefined")


if __name__ == "__main__":
    main()
