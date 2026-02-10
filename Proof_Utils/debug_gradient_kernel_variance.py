"""
Debug: Check the gradient kernel variance constant C_∇K

The theory claims:
    Cov[∇δp(x*)] = (C_∇K)/(nh⁴) * p(x*) * I

where C_∇K = 1/(2π) for 2D Gaussian kernel.

Let's derive this independently!
"""

import numpy as np
from scipy.stats import multivariate_normal

def gaussian_kernel_2d(u, h):
    """
    2D Gaussian kernel: K_h(u) = (1/(2πh²)) * exp(-||u||²/(2h²))
    """
    return multivariate_normal.pdf(u, mean=[0, 0], cov=h**2 * np.eye(2))

def gradient_gaussian_kernel_2d(u, h):
    """
    Gradient of 2D Gaussian kernel:
    ∇K_h(u) = -(u/h²) * K_h(u)
    
    Returns a 2D vector.
    """
    K = gaussian_kernel_2d(u, h)
    return -(u / h**2) * K

def compute_C_grad_K_numerical(h=1.0, grid_size=1000):
    """
    Numerically compute C_∇K = ∫||∇K_h(u)||² du
    
    For isotropic kernel, this should be independent of h (up to scaling).
    """
    # Create grid
    u_max = 5 * h  # Integrate to 5 standard deviations
    u_vals = np.linspace(-u_max, u_max, grid_size)
    du = u_vals[1] - u_vals[0]
    
    # Compute integral
    integral = 0.0
    for u1 in u_vals:
        for u2 in u_vals:
            u = np.array([u1, u2])
            grad_K = gradient_gaussian_kernel_2d(u, h)
            integral += np.dot(grad_K, grad_K) * du**2
    
    return integral

def compute_C_grad_K_analytical():
    """
    Analytical computation of C_∇K for 2D Gaussian kernel.
    
    K_h(u) = (1/(2πh²)) * exp(-||u||²/(2h²))
    ∇K_h(u) = -(u/h²) * K_h(u)
    
    ||∇K_h(u)||² = (||u||²/h⁴) * K_h(u)²
    
    ∫||∇K_h(u)||² du = ∫(||u||²/h⁴) * [(1/(2πh²))²] * exp(-||u||²/h²) du
                      = (1/(4π²h⁸)) * ∫||u||² * exp(-||u||²/h²) du
    
    Let v = u/h, so u = hv, du = h² dv (in 2D), ||u||² = h²||v||²:
    
    = (1/(4π²h⁸)) * ∫(h²||v||²) * exp(-||v||²) * h² dv
    = (1/(4π²h⁴)) * ∫||v||² * exp(-||v||²) dv
    
    For 2D: ∫||v||² * exp(-||v||²) dv = ∫∫(v1² + v2²) * exp(-(v1² + v2²)) dv1 dv2
    
    In polar coords (r, θ): v1² + v2² = r², dv = r dr dθ
    
    = ∫₀^{2π} ∫₀^∞ r² * exp(-r²) * r dr dθ
    = 2π * ∫₀^∞ r³ * exp(-r²) dr
    
    Let w = r², dw = 2r dr, r dr = dw/2, r³ dr = r² * r dr = w * dw/2:
    
    = 2π * ∫₀^∞ (w/2) * exp(-w) dw
    = π * ∫₀^∞ w * exp(-w) dw
    = π * Γ(2)  [Gamma function]
    = π * 1!
    = π
    
    Therefore:
    C_∇K = (1/(4π²h⁴)) * π * h⁴ = 1/(4π)
    
    **Wait! The LaTeX says C_∇K = 1/(2π), but I get 1/(4π)!**
    
    Let me recheck...
    
    Actually, wait. The covariance formula in the LaTeX is:
    
    Cov[∇δp(x)] = (1/n) * Var[∇K_h(x - X)]
                = (1/n) * (E[||∇K_h(x - X)||²] - E[∇K_h(x - X)]²)
    
    If we're at a peak where E[∇K_h] ≈ 0, then:
    
    Cov[∇δp(x)] = (1/n) * ∫||∇K_h(x - y)||² p(y) dy
    
    For p(y) ≈ p(x) (locally constant):
    
    ≈ (p(x)/n) * ∫||∇K_h(u)||² du  [where u = x - y]
    
    So C_∇K should be ∫||∇K_h(u)||² du.
    
    But wait, there's also h scaling! Let me recalculate more carefully...
    """
    # From above derivation:
    return 1.0 / (4.0 * np.pi)

def compute_C_grad_K_corrected():
    """
    Let me reconsider the LaTeX derivation.
    
    From Lemma 12.3:
    "For Gaussian kernel in R²:
     Cov[∇δp(x*)] = (C_∇K)/(nh⁴) * p(x*) * I"
    
    This suggests C_∇K is dimensionless or has dimensions of h⁴.
    
    Let me check the gradient kernel properties:
    
    K_h(u) = (1/(2πh²)) * exp(-||u||²/(2h²))
    ∇K_h(u) = -(u/h²) * K_h(u) = -(u/(2πh⁴)) * exp(-||u||²/(2h²))
    
    The variance of ∇K_h components scales as:
    Var[∇K_h,i] ∝ ∫(u_i²/h⁴) * K_h(u)² du
    
    Since K_h² scales as 1/h⁴, we have:
    Var[∇K_h,i] ∝ (1/h⁴) * ∫u_i² * (1/h⁴) * exp(...) du
                 ∝ (1/h⁸) * h² (from integral of u_i²)
                 = 1/h⁶  ???
    
    No wait, let me be more careful with the integral...
    
    Actually, let's just compute it numerically and see what h-scaling we get!
    """
    results = {}
    for h in [0.5, 1.0, 2.0, 4.0]:
        C_num = compute_C_grad_K_numerical(h=h, grid_size=200)
        results[h] = C_num
        print(f"h = {h:4.1f}: C_∇K (numerical) = {C_num:.6f}, scaled by h⁴ = {C_num * h**4:.6f}")
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("GRADIENT KERNEL VARIANCE CONSTANT")
    print("=" * 80)
    print()
    
    print("Analytical derivation:")
    C_analytical = compute_C_grad_K_analytical()
    print(f"  C_∇K = 1/(4π) = {C_analytical:.6f}")
    print(f"  LaTeX claims: C_∇K = 1/(2π) = {1.0/(2.0*np.pi):.6f}")
    print()
    
    print("Numerical verification:")
    compute_C_grad_K_corrected()
    print()
    
    print("Conclusion:")
    print("-" * 80)
    print("If numerical * h⁴ is constant, then the formula Cov ∝ 1/(nh⁴) is correct.")
    print("The actual constant value determines σ².")
