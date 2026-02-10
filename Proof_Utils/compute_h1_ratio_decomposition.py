#!/usr/bin/env python3
"""
Compute the implied sample size ratios from observed h1 ratios and computed curvature ratios.

Given:
- Observed h1_NLL/h1_MSE ratios: 0.411, 0.576, 0.501
- Computed (2ρ)^(1/6) from Laplacian integrals: 0.645, 0.602, 0.554

We solve for the sample size ratio:
  h1_NLL/h1_MSE = (2ρ)^(1/6) × sqrt(n_NLL/n_MSE)
  => sqrt(n_NLL/n_MSE) = (h1_NLL/h1_MSE) / (2ρ)^(1/6)
  => n_NLL/n_MSE = [(h1_NLL/h1_MSE) / (2ρ)^(1/6)]²
"""

import json
import numpy as np

# Load computed curvature ratios
with open('results/laplacian_integrals.json', 'r') as f:
    data = json.load(f)

# Observed h1 ratios from the experiments
observed_h1_ratios = [0.411, 0.576, 0.501]
mixture_names = ["Mixture 1", "Mixture 2", "Mixture 3"]

print("="*80)
print("DECOMPOSITION OF OBSERVED h1 RATIOS")
print("="*80)
print()

results = []
for i, (obs_ratio, mixture_data) in enumerate(zip(observed_h1_ratios, data)):
    curvature_only = mixture_data['predicted_ratio_curvature']
    rho = mixture_data['rho']
    
    # Compute implied sample size ratio
    sqrt_n_ratio = obs_ratio / curvature_only
    n_ratio = sqrt_n_ratio ** 2
    
    # Store results
    result = {
        'mixture': mixture_names[i],
        'n_components': mixture_data['n_components'],
        'observed_h1_ratio': obs_ratio,
        'rho': rho,
        'curvature_factor': curvature_only,
        'implied_sqrt_n_ratio': sqrt_n_ratio,
        'implied_n_ratio': n_ratio,
        'predicted_h1_ratio': curvature_only * sqrt_n_ratio
    }
    results.append(result)
    
    print(f"{mixture_names[i]} ({mixture_data['n_components']} components)")
    print(f"{'-'*80}")
    print(f"  Observed h1_NLL/h1_MSE:           {obs_ratio:.6f}")
    print(f"  ρ (curvature concentration):      {rho:.6f}")
    print(f"  (2ρ)^(1/6) [curvature only]:      {curvature_only:.6f}")
    print(f"  Implied sqrt(n_NLL/n_MSE):        {sqrt_n_ratio:.6f}")
    print(f"  Implied n_NLL/n_MSE:              {n_ratio:.6f}  ({n_ratio*100:.1f}%)")
    print(f"  Predicted h1 ratio (check):       {result['predicted_h1_ratio']:.6f}")
    print(f"  Prediction error:                 {abs(result['predicted_h1_ratio'] - obs_ratio):.6f}")
    print()

print("="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Mixture':<12} {'Comps':<6} {'Obs. Ratio':<12} {'ρ':<12} {'(2ρ)^1/6':<12} "
      f"{'√(n_NLL/n_MSE)':<15} {'n_NLL/n_MSE':<12}")
print("-"*80)
for r in results:
    print(f"{r['mixture']:<12} {r['n_components']:<6} {r['observed_h1_ratio']:<12.6f} "
          f"{r['rho']:<12.6f} {r['curvature_factor']:<12.6f} "
          f"{r['implied_sqrt_n_ratio']:<15.6f} {r['implied_n_ratio']:<12.6f}")
print("="*80)

# Export for LaTeX
with open('results/h1_ratio_decomposition.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/h1_ratio_decomposition.json")
