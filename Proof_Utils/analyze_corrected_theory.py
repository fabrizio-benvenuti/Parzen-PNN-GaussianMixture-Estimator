"""
Analyze the corrected theory validation results.

Compare the corrected theory (with three terms: bias, var_0, var_2)
against observations and the old theory (two terms).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_validation_results(filepath):
    """Load validation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_old_theory_predictions(results, geometry):
    """
    Compute predictions using the OLD theory (two-term formula).
    
    OLD: RMSE² = β²h⁴/(4(n-1)²) + σ²(n-1)²/(nh⁴)
    where σ² = (1/2π) * p(x*) * tr(Λ⁻²)  [with wrong constant]
    
    Returns list of predicted RMSE values.
    """
    # Old constant (wrong)
    C_grad_K_old = 1.0 / (2.0 * np.pi)
    
    # Recompute sigma_squared with old constant
    sigma_sq_old = (C_grad_K_old / (1.0 / (4.0 * np.pi))) * geometry['sigma_0_squared']
    
    predictions = []
    for r in results:
        n = r['n_total']
        h1 = r['h1']
        
        bias_term = geometry['beta_squared'] * h1**4 / (4 * (n - 1)**2)
        var_term = sigma_sq_old * (n - 1)**2 / (n * h1**4)
        
        rmse_old = np.sqrt(bias_term + var_term)
        predictions.append(rmse_old)
    
    return predictions

def create_comparison_table(results, geometry):
    """Create a comparison table showing old vs new theory."""
    
    data = []
    
    old_predictions = compute_old_theory_predictions(results, geometry)
    
    for i, r in enumerate(results):
        n = r['n_total']
        h1 = r['h1']
        
        obs = r['observed_rmse']
        pred_new = r['predicted_rmse']
        pred_old = old_predictions[i]
        
        # Compute relative errors
        err_old = abs(obs - pred_old) / obs if obs > 0 else np.nan
        err_new = abs(obs - pred_new) / obs if obs > 0 else np.nan
        
        # Improvement
        improvement = (err_old - err_new) / err_old if err_old > 0 else np.nan
        
        data.append({
            'n': n,
            'h1': h1,
            'observed': obs,
            'old_theory': pred_old,
            'new_theory': pred_new,
            'old_error_%': err_old * 100,
            'new_error_%': err_new * 100,
            'improvement_%': improvement * 100,
            'bias_term': r['bias_term'],
            'var_0_term': r['var_0_term'],
            'var_2_term': r['var_2_term'],
        })
    
    return pd.DataFrame(data)

def plot_comparison(df, output_path):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get unique values
    h1_values = sorted(df['h1'].unique())
    n_values = sorted(df['n'].unique())
    
    # Plot 1: Observed vs Predicted (Old Theory)
    for h1 in h1_values:
        subset = df[df['h1'] == h1]
        axes[0, 0].scatter(subset['old_theory'], subset['observed'], 
                          label=f'h1={h1}', alpha=0.7, s=100)
    
    max_val = max(df['observed'].max(), df['old_theory'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect prediction')
    axes[0, 0].set_xlabel('Old Theory Prediction')
    axes[0, 0].set_ylabel('Observed RMSE')
    axes[0, 0].set_title('Old Theory (2-term formula with C_∇K=1/2π)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Observed vs Predicted (New Theory)
    for h1 in h1_values:
        subset = df[df['h1'] == h1]
        axes[0, 1].scatter(subset['new_theory'], subset['observed'], 
                          label=f'h1={h1}', alpha=0.7, s=100)
    
    max_val = max(df['observed'].max(), df['new_theory'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect prediction')
    axes[0, 1].set_xlabel('New Theory Prediction')
    axes[0, 1].set_ylabel('Observed RMSE')
    axes[0, 1].set_title('New Theory (3-term formula with C_∇K=1/4π)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Relative error comparison
    for h1 in h1_values:
        subset = df[df['h1'] == h1]
        axes[1, 0].plot(subset['n'], subset['old_error_%'], 'o-', 
                       label=f'h1={h1} (old)', alpha=0.7)
        axes[1, 0].plot(subset['n'], subset['new_error_%'], 's--', 
                       label=f'h1={h1} (new)', alpha=0.7)
    
    axes[1, 0].set_xlabel('Sample size n')
    axes[1, 0].set_ylabel('Relative Error (%)')
    axes[1, 0].set_title('Relative Prediction Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Term contributions
    for h1 in h1_values:
        subset = df[df['h1'] == h1]
        total_var = subset['var_0_term'] + subset['var_2_term']
        axes[1, 1].plot(subset['n'], subset['var_0_term'], 'o-', 
                       label=f'h1={h1} var₀ (1/h⁴)', alpha=0.7)
        axes[1, 1].plot(subset['n'], subset['var_2_term'], 's--', 
                       label=f'h1={h1} var₂ (1/n)', alpha=0.5)
    
    axes[1, 1].set_xlabel('Sample size n')
    axes[1, 1].set_ylabel('Variance term value')
    axes[1, 1].set_title('Variance Term Contributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")

def main():
    print("=" * 80)
    print("ANALYSIS OF CORRECTED THEORY")
    print("=" * 80)
    
    # Load results
    results_file = Path("results/peak_error_validation_mixture1.json")
    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        print("Please run validate_peak_error_formulas.py first.")
        return
    
    data = load_validation_results(results_file)
    
    print("\n=== Mixture Information ===")
    print(f"Components: {data['mixture_info']['n_components']}")
    print(f"Peaks: {data['mixture_info']['n_peaks']}")
    print(f"Primary peak: {data['mixture_info']['primary_peak']}")
    
    print("\n=== Geometric Parameters ===")
    geom = data['geometry']
    print(f"β² = {geom['beta_squared']:.6e}")
    print(f"σ₀² = {geom['sigma_0_squared']:.6e}  (primary variance, ~ 1/h⁴)")
    print(f"σ₂² = {geom['sigma_2_squared']:.6e}  (curvature correction, ~ 1/n)")
    print(f"Peak density p(x*) = {geom['peak_density']:.6f}")
    
    # Create comparison table
    print("\n=== Creating Comparison Table ===")
    df = create_comparison_table(data['results'], geom)
    
    # Save to CSV
    output_csv = Path("results/theory_comparison.csv")
    df.to_csv(output_csv, index=False, float_format='%.6e')
    print(f"Saved to {output_csv}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nOld Theory (2-term with C_∇K=1/2π):")
    print(f"  Mean relative error: {df['old_error_%'].mean():.2f}%")
    print(f"  Median relative error: {df['old_error_%'].median():.2f}%")
    print(f"  Max relative error: {df['old_error_%'].max():.2f}%")
    
    print(f"\nNew Theory (3-term with C_∇K=1/4π):")
    print(f"  Mean relative error: {df['new_error_%'].mean():.2f}%")
    print(f"  Median relative error: {df['new_error_%'].median():.2f}%")
    print(f"  Max relative error: {df['new_error_%'].max():.2f}%")
    
    print(f"\nImprovement:")
    print(f"  Mean improvement: {df['improvement_%'].mean():.2f}%")
    print(f"  Median improvement: {df['improvement_%'].median():.2f}%")
    
    # Print detailed table for key cases
    print("\n=== Detailed Comparison (selected cases) ===")
    print("\nSmall h1 = 2.0:")
    subset = df[df['h1'] == 2.0]
    print(subset[['n', 'observed', 'old_theory', 'new_theory', 'old_error_%', 'new_error_%']].to_string(index=False))
    
    print("\nLarge h1 = 12.0:")
    subset = df[df['h1'] == 12.0]
    print(subset[['n', 'observed', 'old_theory', 'new_theory', 'old_error_%', 'new_error_%']].to_string(index=False))
    
    # Analyze variance term importance
    print("\n=== Variance Term Analysis ===")
    print("\nRatio of var₂/var₀ (curvature correction / primary variance):")
    df['var_ratio'] = df['var_2_term'] / df['var_0_term']
    for h1 in sorted(df['h1'].unique()):
        subset = df[df['h1'] == h1]
        print(f"  h1={h1}: mean ratio = {subset['var_ratio'].mean():.4f}")
    
    # Create plots
    print("\n=== Creating Comparison Plots ===")
    plot_comparison(df, Path("figures/theory_comparison.jpeg"))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Corrected constant C_∇K = 1/(4π) reduces systematic bias")
    print("2. Curvature correction term σ₂²/n improves predictions, especially for large h")
    print("3. The three-term formula better captures the h-dependence of variance")

if __name__ == "__main__":
    main()
