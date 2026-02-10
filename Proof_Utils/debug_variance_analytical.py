"""
Investigate the discrepancy in gradient variance formula.

The empirical variance is ~0.22 times the theoretical prediction.
1/0.22 ≈ 4.5

This suggests we might be off by a factor related to:
- sqrt(2) squared = 2
- 2*sqrt(2) ≈ 2.83
- 2*2 = 4

Let me check if there's an error in how the variance of the gradient is computed.
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

def analytical_gradient_variance_component():
    """
    Compute Var[∇K_h,x component] analytically for unit Gaussian kernel.
    
    For K_h(u) = (1/(2πh²)) * exp(-||u||²/(2h²))
    ∇K_h(u) = -(u/h²) * K_h(u)
    
    So ∇K_h,x(u) = -(u_x/h²) * (1/(2πh²)) * exp(-(u_x² + u_y²)/(2h²))
    
    Var[∇K_h,x(U)] where U ~ N(0, I):
    = E[(∇K_h,x(U))²] - (E[∇K_h,x(U)])²
    
    E[∇K_h,x(U)] should be 0 by symmetry, so:
    Var[∇K_h,x(U)] = E[(∇K_h,x(U))²]
                    = ∫∫ (∇K_h,x(u))² * p(u) du
    
    where p(u) = (1/(2π)) * exp(-||u||²/2) is the true density.
    
    This is:
    ∫∫ (u_x/h²)² * K_h(u)² * p(u) du
    
    where K_h(u) = (1/(2πh²)) * exp(-||u||²/(2h²))
    
    So:
    = ∫∫ (u_x²/h⁴) * (1/(2πh²))² * exp(-||u||²/h²) * (1/(2π)) * exp(-||u||²/2) du
    = (1/(4π²h⁸ * 2π)) * ∫∫ u_x² * exp(-||u||²/h²) * exp(-||u||²/2) du
    = (1/(8π³h⁸)) * ∫∫ u_x² * exp(-||u||²(1/h² + 1/2)) du
    
    Let α = 1/h² + 1/2 = (2 + h²)/(2h²)
    
    = (1/(8π³h⁸)) * ∫∫ u_x² * exp(-α||u||²) du
    
    In 2D, ∫∫ u_x² * exp(-α||u||²) du:
    By symmetry, ∫ u_x² = ∫ u_y², and together they equal ∫ ||u||².
    
    So ∫∫ u_x² * exp(-α||u||²) du = (1/2) * ∫∫ ||u||² * exp(-α||u||²) du
    
    In polar coordinates:
    ∫₀^{2π} ∫₀^∞ r² * exp(-α r²) * r dr dθ
    = 2π * ∫₀^∞ r³ * exp(-α r²) dr
    
    Let w = α r², dw = 2α r dr, r dr = dw/(2α), r³ dr = (w/α) * dw/(2α) = w/(2α²) dw
    
    = 2π * ∫₀^∞ (w/(2α²)) * exp(-w) dw
    = (π/α²) * Γ(2)
    = π/α²
    
    So:
    ∫∫ u_x² * exp(-α||u||²) du = (1/2) * (π/α²) = π/(2α²)
    
    Therefore:
    Var[∇K_h,x(U)] = (1/(8π³h⁸)) * π/(2α²)
                    = 1/(16π²h⁸α²)
    
    where α = (2 + h²)/(2h²)
    
    So:
    Var[∇K_h,x(U)] = 1/(16π²h⁸) * (2h²)²/(2 + h²)²
                    = (4h⁴)/(16π²h⁸(2 + h²)²)
                    = 1/(4π²h⁴(2 + h²)²)
    
    For large h: α → 1/(2h²), so Var → 4h⁴/(16π²h⁸) = 1/(4π²h⁴)
    For h=1: α = 3/(2), Var = 1/(4π² * 1 * 9) = 1/(36π²)
    
    Wait, this doesn't have the claimed form! Let me reconsider...
    """
    
    # Actually, let me just compute it numerically and see what the pattern is
    results = {}
    
    for h in [0.5, 1.0, 2.0]:
        # Numerical integration
        def integrand(uy, ux):
            u = np.array([ux, uy])
            # Gradient component
            K = np.exp(-np.dot(u, u)/(2*h**2)) / (2*np.pi*h**2)
            grad_x = -(ux/h**2) * K
            
            # True density
            p = np.exp(-np.dot(u, u)/2) / (2*np.pi)
            
            return grad_x**2 * p
        
        # Integrate over large region
        limit = 5 * max(h, 1)
        var_x, _ = dblquad(integrand, -limit, limit, -limit, limit)
        
        results[h] = {
            'var_x': var_x,
            'var_x_times_h4': var_x * h**4,
        }
        
        print(f"h={h}: Var[∇K_x] = {var_x:.6e}, Var * h⁴ = {var_x * h**4:.6e}")
    
    return results

def test_with_n_samples():
    """
    The claimed formula is:
        Cov[∇δp] = (1/n) * Var[∇K_h(x - X)]
    
    But wait - is this the variance when X is a SINGLE sample, or when we average?
    
    If ∇δp = (1/n) Σ ∇K_h(x - X_i) - E[∇K_h], then:
        Var[∇δp] = (1/n²) * n * Var[∇K_h]  (for iid samples)
                 = (1/n) * Var[∇K_h]
    
    So the 1/n factor is correct. But maybe the issue is that we need to use
    Var[∇K_h(x - X)] where X ~ p, not just ∫||∇K||².
    
    Let me compute this properly.
    """
    print("\n" + "=" * 80)
    print("Computing Var[∇K_h(0 - X)] where X ~ N(0, I)")
    print("=" * 80)
    
    for h in [0.5, 1.0, 2.0]:
        def integrand_x_comp(uy, ux):
            u = np.array([ux, uy])
            # ∇K_h at origin with sample at u
            K = np.exp(-np.dot(u, u)/(2*h**2)) / (2*np.pi*h**2)
            grad_x = -(-ux/h**2) * K  # Note: grad w.r.t. origin, u = 0 - X = -X
            
            # True density at u
            p = np.exp(-np.dot(u, u)/2) / (2*np.pi)
            
            return grad_x**2 * p
        
        limit = 5 * max(h, 1)
        var_x, _ = dblquad(integrand_x_comp, -limit, limit, -limit, limit)
        
        # Also compute E[∇K]² (should be 0 by symmetry, but let's check)
        def integrand_mean_x(uy, ux):
            u = np.array([ux, uy])
            K = np.exp(-np.dot(u, u)/(2*h**2)) / (2*np.pi*h**2)
            grad_x = (ux/h**2) * K  # grad of K_h(0 - u) = grad of K_h(-u) w.r.t. first arg
            p = np.exp(-np.dot(u, u)/2) / (2*np.pi)
            return grad_x * p
        
        mean_x, _ = dblquad(integrand_mean_x, -limit, limit, -limit, limit)
        
        print(f"\nh={h}:")
        print(f"  E[∇K_h,x] = {mean_x:.6e} (should be ~0)")
        print(f"  Var[∇K_h,x] = {var_x:.6e}")
        print(f"  Var * h⁴ = {var_x * h**4:.6e}")
        print(f"  Var * n * h⁴ (for n=100) = {var_x * 100 * h**4:.6e}")

if __name__ == "__main__":
    print("=" * 80)
    print("ANALYTICAL GRADIENT VARIANCE")
    print("=" * 80)
    analytical_gradient_variance_component()
    
    test_with_n_samples()
