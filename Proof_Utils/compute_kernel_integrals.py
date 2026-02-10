"""
Compute C_H integral numerically for 2D Gaussian kernel.

C_H = ∫ ||∇K(u)||² u u^T du

This is needed for the curvature correction term in variance.
"""
import numpy as np
from scipy.integrate import nquad

def gaussian_kernel_2d(u):
    """2D Gaussian kernel K(u) = (1/2π) exp(-||u||²/2)"""
    norm_sq = np.dot(u, u)
    return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * norm_sq)

def grad_kernel_2d(u):
    """Gradient of 2D Gaussian kernel: ∇K(u) = -u * K(u)"""
    K_u = gaussian_kernel_2d(u)
    return -u * K_u

def grad_kernel_squared_norm(u):
    """||∇K(u)||²"""
    grad = grad_kernel_2d(u)
    return np.dot(grad, grad)

def integrand_C_H_11(u1, u2):
    """Integrand for C_H[1,1] = ∫ ||∇K||² u₁² du"""
    u = np.array([u1, u2])
    return grad_kernel_squared_norm(u) * u1**2

def integrand_C_H_22(u1, u2):
    """Integrand for C_H[2,2] = ∫ ||∇K||² u₂² du"""
    u = np.array([u1, u2])
    return grad_kernel_squared_norm(u) * u2**2

def integrand_C_H_12(u1, u2):
    """Integrand for C_H[1,2] = ∫ ||∇K||² u₁u₂ du"""
    u = np.array([u1, u2])
    return grad_kernel_squared_norm(u) * u1 * u2

def compute_C_grad_K():
    """
    Compute C_∇K = ∫ ||∇K(u)||² du numerically.
    
    For 2D Gaussian kernel, this should be 1/(4π).
    """
    def integrand(u1, u2):
        u = np.array([u1, u2])
        return grad_kernel_squared_norm(u)
    
    # Integrate over all space (use ±∞ with reasonable limits)
    result, error = nquad(integrand, [[-10, 10], [-10, 10]])
    
    return result, error

def compute_C_H_trace():
    """
    Compute tr(C_H) where C_H = ∫ ||∇K||² u u^T du.
    
    For isotropic kernel: C_H[1,1] = C_H[2,2] and C_H[1,2] = 0 by symmetry.
    So tr(C_H) = 2 * C_H[1,1].
    """
    # Compute C_H[1,1]
    result_11, error_11 = nquad(integrand_C_H_11, [[-10, 10], [-10, 10]])
    
    # Compute C_H[2,2] (should be same as C_H[1,1] by symmetry)
    result_22, error_22 = nquad(integrand_C_H_22, [[-10, 10], [-10, 10]])
    
    # Compute C_H[1,2] (should be ~0 by symmetry)
    result_12, error_12 = nquad(integrand_C_H_12, [[-10, 10], [-10, 10]])
    
    return {
        'C_H_11': result_11,
        'C_H_22': result_22,
        'C_H_12': result_12,
        'trace': result_11 + result_22,
        'errors': {'11': error_11, '22': error_22, '12': error_12}
    }

def main():
    print("=" * 80)
    print("NUMERICAL COMPUTATION OF KERNEL INTEGRALS")
    print("=" * 80)
    
    print("\n1. Computing C_∇K = ∫ ||∇K(u)||² du...")
    C_grad_K, err_grad_K = compute_C_grad_K()
    theoretical = 1.0 / (4.0 * np.pi)
    print(f"   Numerical result: {C_grad_K:.10f}")
    print(f"   Integration error: {err_grad_K:.2e}")
    print(f"   Theoretical value: {theoretical:.10f}")
    print(f"   Relative difference: {abs(C_grad_K - theoretical)/theoretical * 100:.4f}%")
    
    print("\n2. Computing C_H matrix components...")
    C_H_results = compute_C_H_trace()
    print(f"   C_H[1,1] = {C_H_results['C_H_11']:.10f} ± {C_H_results['errors']['11']:.2e}")
    print(f"   C_H[2,2] = {C_H_results['C_H_22']:.10f} ± {C_H_results['errors']['22']:.2e}")
    print(f"   C_H[1,2] = {C_H_results['C_H_12']:.10f} ± {C_H_results['errors']['12']:.2e}")
    print(f"   tr(C_H) = {C_H_results['trace']:.10f}")
    
    print("\n3. Relationship to C_∇K:")
    ratio = C_H_results['trace'] / C_grad_K
    print(f"   tr(C_H) / C_∇K = {ratio:.6f}")
    
    print("\n4. For Hessian H_p = -Λ at peak:")
    print(f"   The curvature correction term involves:")
    print(f"   σ₂² = -(tr(C_H) / 2) * p(x*) * tr(Λ⁻¹)")
    print(f"   σ₂² = -{C_H_results['trace']/2:.6f} * p(x*) * tr(Λ⁻¹)")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
