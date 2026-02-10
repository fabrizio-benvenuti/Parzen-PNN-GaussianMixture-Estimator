"""
Compute all geometric and theoretical terms for Gaussian mixtures.

This script computes and tabulates:
1. Curvature concentration ratio ρ
2. Peak geometry parameters (β², σ², Λ, etc.)
3. MISE and NLL expansion coefficients
4. Optimal bandwidth predictions

Output: Comprehensive tables matching the theoretical document.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from estimator import (
    GaussianMixture,
    MultivariateGaussian,
    Plotter,
)
from validate_peak_error_formulas import (
    mixture_pdf,
    mixture_laplacian,
    mixture_hessian,
    mixture_gradient_laplacian,
    find_mixture_peaks,
    compute_peak_geometry_parameters,
    compute_curvature_concentration_ratio,
)


def compute_mise_coefficients(mixture: GaussianMixture, plotter: Plotter) -> Dict[str, float]:
    """
    Compute coefficients for MISE expansion:
    
    MISE(h) = C_b * h^4 + C_v / (n*h^2)
    
    where:
        C_b = (1/4) * ∫(Δp)² dx
        C_v = 1/(2π) (for 2D Gaussian kernel)
    """
    grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
    dx, dy = plotter.x[1] - plotter.x[0], plotter.y[1] - plotter.y[0]
    dA = dx * dy
    
    # Compute Laplacian at each grid point
    laplacian_vals = np.array([mixture_laplacian(pt, mixture) for pt in grid_points])
    
    # C_b = (1/4) * ∫(Δp)² dx
    C_b = 0.25 * float(np.sum(laplacian_vals**2) * dA)
    
    # C_v = 1/(4π) for 2D Gaussian kernel (CORRECTED from 1/(2π))
    C_v = 1.0 / (4.0 * np.pi)
    
    return {
        'C_b': C_b,
        'C_v': C_v,
        'laplacian_integral': float(np.sum(laplacian_vals**2) * dA),
    }


def compute_nll_coefficients(mixture: GaussianMixture, plotter: Plotter) -> Dict[str, float]:
    """
    Compute coefficients for Expected NLL expansion:
    
    E[NLL(h)] = const + C̃_b * h^4 + C̃_v / (n*h^2)
    
    where:
        C̃_b = (1/8) * ∫(Δp)²/p * p dx
        C̃_v = 1/(4π)
    """
    grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
    dx, dy = plotter.x[1] - plotter.x[0], plotter.y[1] - plotter.y[0]
    dA = dx * dy
    
    # Compute Laplacian and density at each grid point
    laplacian_vals = np.array([mixture_laplacian(pt, mixture) for pt in grid_points])
    density_vals = np.array([mixture_pdf(pt, mixture) for pt in grid_points])
    
    # C̃_b = (1/8) * ∫(Δp)²/p * p dx
    safe_density = np.maximum(density_vals, 1e-300)
    weighted_laplacian_sq = (laplacian_vals**2 / safe_density) * density_vals
    C_tilde_b = 0.125 * float(np.sum(weighted_laplacian_sq) * dA)
    
    # C̃_v = 1/(8π) for 2D Gaussian kernel (CORRECTED: was 1/(4π), now properly derived)
    # For NLL, the correct constant differs from MISE due to the 1/p weighting
    C_tilde_v = 1.0 / (8.0 * np.pi)
    
    return {
        'C_tilde_b': C_tilde_b,
        'C_tilde_v': C_tilde_v,
        'weighted_laplacian_integral': float(np.sum(weighted_laplacian_sq) * dA),
    }


def compute_optimal_bandwidths(mise_coef: Dict, nll_coef: Dict, n: int) -> Dict[str, float]:
    """
    Compute optimal bandwidths for fixed sample size n.
    
    h_MSE^* = (C_v / (2*n*C_b))^(1/6)
    h_NLL^* = (C̃_v / (2*n*C̃_b))^(1/6)
    """
    C_b = mise_coef['C_b']
    C_v = mise_coef['C_v']
    C_tilde_b = nll_coef['C_tilde_b']
    C_tilde_v = nll_coef['C_tilde_v']
    
    h_mse = (C_v / (2.0 * n * C_b)) ** (1.0/6.0)
    h_nll = (C_tilde_v / (2.0 * n * C_tilde_b)) ** (1.0/6.0)
    
    return {
        'n': n,
        'h_mse_optimal': h_mse,
        'h_nll_optimal': h_nll,
        'ratio_h_nll_h_mse': h_nll / h_mse,
    }


def create_comprehensive_table(
    mixtures: List[GaussianMixture],
    mixture_names: List[str],
    plotter: Plotter,
    n_sample: int = 100
) -> pd.DataFrame:
    """
    Create a comprehensive table with all geometric terms for each mixture.
    """
    rows = []
    
    for mixture, name in zip(mixtures, mixture_names):
        print(f"\nProcessing {name}...")
        
        # Basic info
        n_components = len(mixture._gaussians)
        
        # Find peaks
        peaks = find_mixture_peaks(mixture, n_starts=30)
        n_peaks = len(peaks)
        
        # Primary peak (highest density)
        if n_peaks > 0:
            peak_densities = [mixture_pdf(pk, mixture) for pk in peaks]
            primary_peak_idx = np.argmax(peak_densities)
            primary_peak = peaks[primary_peak_idx]
            peak_density = peak_densities[primary_peak_idx]
            
            # Peak geometry
            peak_geom = compute_peak_geometry_parameters(primary_peak, mixture)
            beta_sq = peak_geom['beta_squared']
            sigma_sq = peak_geom['sigma_squared']
            peak_sharpness = peak_geom['peak_sharpness']
        else:
            primary_peak = [np.nan, np.nan]
            peak_density = np.nan
            beta_sq = np.nan
            sigma_sq = np.nan
            peak_sharpness = np.nan
        
        # Curvature concentration
        curv = compute_curvature_concentration_ratio(mixture, plotter)
        rho = curv['rho']
        curvature_factor = curv['curvature_factor']
        
        # MISE coefficients
        mise_coef = compute_mise_coefficients(mixture, plotter)
        C_b = mise_coef['C_b']
        C_v = mise_coef['C_v']
        
        # NLL coefficients
        nll_coef = compute_nll_coefficients(mixture, plotter)
        C_tilde_b = nll_coef['C_tilde_b']
        C_tilde_v = nll_coef['C_tilde_v']
        
        # Optimal bandwidths (for reference sample size)
        opt_bw = compute_optimal_bandwidths(mise_coef, nll_coef, n_sample)
        
        rows.append({
            'Mixture': name,
            'N_Components': n_components,
            'N_Peaks': n_peaks,
            'Primary_Peak_X': primary_peak[0],
            'Primary_Peak_Y': primary_peak[1],
            'Peak_Density': peak_density,
            'Beta_Squared': beta_sq,
            'Sigma_Squared': sigma_sq,
            'Peak_Sharpness': peak_sharpness,
            'Rho': rho,
            'Curvature_Factor_2rho^1/6': curvature_factor,
            'C_b_MISE': C_b,
            'C_v_MISE': C_v,
            'C_tilde_b_NLL': C_tilde_b,
            'C_tilde_v_NLL': C_tilde_v,
            f'h_MSE_opt_n{n_sample}': opt_bw['h_mse_optimal'],
            f'h_NLL_opt_n{n_sample}': opt_bw['h_nll_optimal'],
            'Ratio_h_NLL/h_MSE': opt_bw['ratio_h_nll_h_mse'],
            'Laplacian_Integral_Numerator': curv['numerator'],
            'Laplacian_Integral_Denominator': curv['denominator'],
        })
    
    df = pd.DataFrame(rows)
    return df


def plot_curvature_distributions(
    mixtures: List[GaussianMixture],
    mixture_names: List[str],
    plotter: Plotter
):
    """
    Visualize the curvature distributions for each mixture.
    """
    n_mixtures = len(mixtures)
    fig, axes = plt.subplots(n_mixtures, 3, figsize=(18, 6*n_mixtures))
    
    if n_mixtures == 1:
        axes = axes.reshape(1, -1)
    
    for i, (mixture, name) in enumerate(zip(mixtures, mixture_names)):
        grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
        
        # Compute fields
        density_vals = np.array([mixture_pdf(pt, mixture) for pt in grid_points]).reshape(plotter.X.shape)
        laplacian_vals = np.array([mixture_laplacian(pt, mixture) for pt in grid_points]).reshape(plotter.X.shape)
        laplacian_sq = laplacian_vals**2
        
        # Plot 1: Density
        im1 = axes[i, 0].contourf(plotter.X, plotter.Y, density_vals, levels=20, cmap='viridis')
        axes[i, 0].set_title(f'{name}\nDensity p(x)')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Plot 2: Laplacian squared (Δp)²
        im2 = axes[i, 1].contourf(plotter.X, plotter.Y, laplacian_sq, levels=20, cmap='hot')
        axes[i, 1].set_title(f'{name}\n(Δp)² [uniform weighting for MSE]')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Plot 3: Weighted Laplacian squared (Δp)²/p
        safe_density = np.maximum(density_vals, 1e-300)
        weighted_laplacian_sq = laplacian_sq / safe_density
        # Clip extreme values for visualization
        vmax = np.percentile(weighted_laplacian_sq, 99)
        im3 = axes[i, 2].contourf(plotter.X, plotter.Y, np.clip(weighted_laplacian_sq, 0, vmax), 
                                   levels=20, cmap='hot')
        axes[i, 2].set_title(f'{name}\n(Δp)²/p [density-weighted for NLL]')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig('figures/curvature_distributions.jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved curvature distributions to figures/curvature_distributions.jpeg")


def plot_peak_geometry(
    mixtures: List[GaussianMixture],
    mixture_names: List[str],
    plotter: Plotter
):
    """
    Visualize peak locations and their geometric properties.
    """
    n_mixtures = len(mixtures)
    fig, axes = plt.subplots(1, n_mixtures, figsize=(6*n_mixtures, 6))
    
    if n_mixtures == 1:
        axes = [axes]
    
    for i, (mixture, name) in enumerate(zip(mixtures, mixture_names)):
        # Density field
        grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
        density_vals = np.array([mixture_pdf(pt, mixture) for pt in grid_points]).reshape(plotter.X.shape)
        
        # Contour plot
        axes[i].contourf(plotter.X, plotter.Y, density_vals, levels=20, cmap='viridis', alpha=0.7)
        
        # Find and mark peaks
        peaks = find_mixture_peaks(mixture, n_starts=30)
        for peak in peaks:
            geom = compute_peak_geometry_parameters(peak, mixture)
            axes[i].plot(peak[0], peak[1], 'r*', markersize=20, markeredgecolor='white', markeredgewidth=1)
            
            # Annotate with β² and σ²
            label = f"β²={geom['beta_squared']:.2e}\nσ²={geom['sigma_squared']:.2e}"
            axes[i].annotate(label, xy=(peak[0], peak[1]), xytext=(10, 10),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_title(f'{name}\nPeaks (red stars) with geometry')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/peak_geometry.jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved peak geometry to figures/peak_geometry.jpeg")


def main():
    """Compute and tabulate all geometric terms for the mixtures."""
    
    print("=" * 80)
    print("COMPUTING MIXTURE GEOMETRY AND THEORETICAL PARAMETERS")
    print("=" * 80)
    
    # Create mixtures
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    g2 = MultivariateGaussian([-2, -1], [[1.14822883, 0.19240818], [0.19240818, 1.23432651]])
    g3 = MultivariateGaussian([-1, 3], [[0.30198015, 0.13745508], [0.13745508, 1.69483031]])
    g4 = MultivariateGaussian([1.5, -0.5], [[0.85553671, -0.19601649], [-0.19601649, 0.7507167]])
    g5 = MultivariateGaussian([-3, 2], [[0.42437194, -0.17066673], [-0.17066673, 2.16117758]])
    
    mixture1 = GaussianMixture([g1], [1.0])
    mixture2 = GaussianMixture([g1, g2, g3], [0.3, 0.3, 0.4])
    mixture3 = GaussianMixture([g1, g2, g3, g4, g5], [0.2, 0.2, 0.2, 0.2, 0.2])
    
    mixtures = [mixture1, mixture2, mixture3]
    mixture_names = ['Mixture_1', 'Mixture_2', 'Mixture_3']
    
    # High-resolution grid for accurate integration
    plotter = Plotter(-5, 5, -5, 5, 200)
    
    # Create comprehensive table
    print("\nComputing comprehensive geometry table...")
    df = create_comprehensive_table(mixtures, mixture_names, plotter, n_sample=100)
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/mixture_geometry_comprehensive.csv"
    df.to_csv(csv_path, index=False, float_format='%.8e')
    print(f"\nSaved comprehensive table to {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GEOMETRY TABLE")
    print("=" * 80)
    
    # Select key columns for display
    display_cols = [
        'Mixture', 'N_Components', 'N_Peaks',
        'Rho', 'Curvature_Factor_2rho^1/6',
        'Beta_Squared', 'Sigma_Squared',
        'C_b_MISE', 'C_tilde_b_NLL',
        'Ratio_h_NLL/h_MSE'
    ]
    
    print(df[display_cols].to_string(index=False))
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    os.makedirs("figures", exist_ok=True)
    
    print("\nPlotting curvature distributions...")
    plot_curvature_distributions(mixtures, mixture_names, plotter)
    
    print("Plotting peak geometry...")
    plot_peak_geometry(mixtures, mixture_names, plotter)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nCurvature Concentration Ratio (ρ):")
    for _, row in df.iterrows():
        print(f"  {row['Mixture']}: ρ = {row['Rho']:.6f}, (2ρ)^(1/6) = {row['Curvature_Factor_2rho^1/6']:.6f}")
    
    print("\nPeak Error Parameters:")
    for _, row in df.iterrows():
        print(f"  {row['Mixture']}: β² = {row['Beta_Squared']:.6e}, σ² = {row['Sigma_Squared']:.6e}")
    
    print("\nOptimal Bandwidth Ratio (fixed n=100):")
    for _, row in df.iterrows():
        print(f"  {row['Mixture']}: h_NLL/h_MSE = {row['Ratio_h_NLL/h_MSE']:.6f}")
    
    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - figures/curvature_distributions.jpeg")
    print(f"  - figures/peak_geometry.jpeg")


if __name__ == "__main__":
    main()
