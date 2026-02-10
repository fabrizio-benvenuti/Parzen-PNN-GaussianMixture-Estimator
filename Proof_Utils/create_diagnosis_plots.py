"""
Create visualizations showing where theory diverges from observations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load validation results
with open('results/peak_error_validation_mixture1.json') as f:
    peak_data = json.load(f)

with open('results/bandwidth_ratio_validation.json') as f:
    ratio_data = json.load(f)

df_peak = pd.DataFrame(peak_data['results'])

# Create figure with multiple panels
fig = plt.figure(figsize=(16, 12))

# Panel 1: Peak Error - Observed vs Predicted for different h1
ax1 = plt.subplot(2, 3, 1)
for h1 in sorted(df_peak['h1'].unique()):
    subset = df_peak[df_peak['h1'] == h1]
    ax1.loglog(subset['n_total'], subset['observed_rmse'], 'o-', label=f'h1={h1} (obs)', markersize=6)
    ax1.loglog(subset['n_total'], subset['predicted_rmse'], 's--', label=f'h1={h1} (theory)', alpha=0.5)

ax1.set_xlabel('Sample size n', fontsize=11)
ax1.set_ylabel('RMSE peak error', fontsize=11)
ax1.set_title('Peak Location Error:\nObserved vs Theory', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Panel 2: Ratio of Observed/Predicted
ax2 = plt.subplot(2, 3, 2)
for h1 in sorted(df_peak['h1'].unique()):
    subset = df_peak[df_peak['h1'] == h1]
    ax2.semilogx(subset['n_total'], subset['rmse_ratio'], 'o-', label=f'h1={h1}', markersize=6)

ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect match')
ax2.set_xlabel('Sample size n', fontsize=11)
ax2.set_ylabel('Observed / Predicted', fontsize=11)
ax2.set_title('Ratio: Where Theory Fails\n(should be 1.0)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 12])

# Panel 3: Bias vs Variance terms
ax3 = plt.subplot(2, 3, 3)
for h1 in [2.0, 12.0]:  # Show extreme cases
    subset = df_peak[df_peak['h1'] == h1]
    ax3.loglog(subset['n_total'], subset['var_term'], 'o-', label=f'h1={h1} var', markersize=6)

ax3.set_xlabel('Sample size n', fontsize=11)
ax3.set_ylabel('Variance term value', fontsize=11)
ax3.set_title('Variance Term Behavior\n(theory claims it should scale as (n-1)²/h1⁴)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Error pattern by h1 (for fixed n)
ax4 = plt.subplot(2, 3, 4)
for n_val in [20, 50, 100, 200]:
    subset = df_peak[df_peak['n_total'] == n_val]
    ratio_obs_pred = subset['observed_rmse'] / subset['predicted_rmse']
    ax4.semilogx(subset['h1'], ratio_obs_pred, 'o-', label=f'n={n_val}', markersize=6)

ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect match')
ax4.set_xlabel('Base bandwidth h1', fontsize=11)
ax4.set_ylabel('Observed / Predicted', fontsize=11)
ax4.set_title('Ratio vs h1: Systematic Bias\n(underpredicts at high h1, overpredicts at low h1)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 12])

# Panel 5: Bandwidth ratio validation
ax5 = plt.subplot(2, 3, 5)
mixtures = [v['mixture'] for v in ratio_data['validations']]
observed = [v['observed_ratio'] for v in ratio_data['validations']]
predicted = [v['predicted_ratio'] for v in ratio_data['validations']]

x = np.arange(len(mixtures))
width = 0.35
ax5.bar(x - width/2, observed, width, label='Observed', color='steelblue')
ax5.bar(x + width/2, predicted, width, label='Theory', color='coral')

ax5.set_xlabel('Mixture', fontsize=11)
ax5.set_ylabel('h_NLL / h_MSE ratio', fontsize=11)
ax5.set_title('Bandwidth Ratio Formula:\nMassive Discrepancy!', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['Mix 1', 'Mix 2', 'Mix 3'], fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(1.0, color='black', linestyle=':', alpha=0.3)

# Panel 6: Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
DIAGNOSIS SUMMARY
════════════════════════════════════════════

Issues Found:
────────────────────────────────────────────
1. WRONG CONSTANT
   • LaTeX: C_∇K = 1/(2π)
   • Correct: C_∇K = 1/(4π)
   • Factor of 2 error

2. VARIANCE FORMULA INVALID ⚠️
   • Claims: Var ∝ p(x*)/(nh⁴)
   • Reality: Doesn't factor when p non-uniform
   • Empirical variance is 0.22× theory
   
3. PEAK ERROR FORMULA WRONG
   • Small h1: overpredicts by 2-10×
   • Large h1: underpredicts by 2-7×
   • Not just a constant factor!
   
4. BANDWIDTH RATIO WRONG
   • Theory predicts 1.12
   • Observations: 0.41-0.58
   • 95-173% relative error

Root Cause:
────────────────────────────────────────────
Lemma 12.3 incorrectly factors:
  ∫||∇K_h(x-y)||² p(y) dy ≈ p(x*) ∫||∇K||²

This is only valid if p is constant!

Conclusion:
────────────────────────────────────────────
Theory needs fundamental revision.
Implementation is correct.
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figures/theory_vs_observations_discrepancy.jpeg', dpi=300, bbox_inches='tight')
print("Saved discrepancy analysis to figures/theory_vs_observations_discrepancy.jpeg")

plt.close()

# Also create a simple summary plot for the README
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Peak error ratio
ax = axes[0]
for h1 in sorted(df_peak['h1'].unique()):
    subset = df_peak[df_peak['h1'] == h1]
    ax.semilogx(subset['n_total'], subset['rmse_ratio'], 'o-', label=f'h1={h1}', markersize=8, linewidth=2)

ax.axhline(1.0, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Perfect match')
ax.fill_between([20, 200], 0.8, 1.2, alpha=0.2, color='green', label='±20% error')
ax.set_xlabel('Sample size n', fontsize=14, fontweight='bold')
ax.set_ylabel('Observed / Predicted', fontsize=14, fontweight='bold')
ax.set_title('Peak Error: Theory vs Observations', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.set_ylim([0, 12])
ax.tick_params(labelsize=11)

# Right: Bandwidth ratio
ax = axes[1]
mixtures = ['Mixture 1', 'Mixture 2', 'Mixture 3']
observed = [0.411, 0.576, 0.501]
predicted = [1.122, 1.122, 1.122]
errors = [abs(o - p)/o * 100 for o, p in zip(observed, predicted)]

x = np.arange(len(mixtures))
width = 0.35
bars1 = ax.bar(x - width/2, observed, width, label='Observed', color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, predicted, width, label='Theory', color='coral', edgecolor='black', linewidth=1.5)

# Add error percentage labels
for i, (b1, b2, err) in enumerate(zip(bars1, bars2, errors)):
    ax.text(i, max(observed[i], predicted[i]) + 0.05, f'{err:.0f}% error', 
            ha='center', fontsize=11, fontweight='bold', color='red')

ax.set_ylabel('h_NLL / h_MSE ratio', fontsize=14, fontweight='bold')
ax.set_title('Bandwidth Ratio: Theory vs Observations', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mixtures, fontsize=11)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
ax.set_ylim([0, 1.5])
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('figures/theory_validation_summary.jpeg', dpi=300, bbox_inches='tight')
print("Saved summary to figures/theory_validation_summary.jpeg")

print("\nDone! See DIAGNOSIS.md for detailed analysis.")
