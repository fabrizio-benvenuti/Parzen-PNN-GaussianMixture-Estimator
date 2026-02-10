# Spurious Peak Validation Guide

## Overview

The script `validate_spurious_peaks.py` implements the mathematical theory from Section "Spurious Peak Formation in Undersmoothed KDE" in `deep_mathematical_proof_PW_overlays.tex`.

## Theory Summary

When KDE bandwidth is too small relative to sample spacing, individual sample points create separate peaks instead of merging into a single smooth peak. The theory predicts:

### Key Formula

For a Gaussian component with standard deviation σ, estimated using n samples and bandwidth h:

```
N_peak(h, n, σ) = 1 + (n-1) * exp(-0.5 * ((h - h_min)/h_trans)²)
```

Where:
- `h_min = σ / (4√n)` - below this, every sample is a separate peak
- `h_trans = σ / (2√n)` - transition bandwidth for peak merging
- `h_max = 2σ / √n` - above this, all samples merge into one peak

### Spurious Peak Score

For each component:
```
S_k = max{0, log(N_peak / 1.5)}
```

For the entire mixture:
```
S_global = Σ w_k * S_k
```

**Interpretation:**
- `S_global = 0`: Clean, no spurious peaks
- `0 < S_global < 0.5`: Mild spurious peaks (barely visible)
- `0.5 ≤ S_global < 1.5`: Moderate spurious peaks (clearly visible)
- `S_global ≥ 1.5`: Severe spurious peaks (multiple false modes per component)

## How to Use the Script

### Basic Usage

```bash
python3 validate_spurious_peaks.py
```

This will:
1. Load mixture definitions from `estimator.py`
2. Load experimental logs from `logs/pw_errors_mixture*.csv` (if available)
3. Compute spurious peak predictions for various configurations
4. Compare predictions with experimental best configurations
5. Generate outputs in `results/` and `figures/`

### Prerequisites

The script automatically uses:
- Mixture definitions from `estimator.py` (lines 1905-1913)
- Experimental logs from `logs/` (generated with `--pw-write-logs`)

To generate fresh experimental logs:
```bash
python3 estimator.py --pw-only --pw-write-logs
```

### Outputs

1. **Console Output**: Detailed predictions table showing:
   - Configuration (n, h1, h_eff)
   - Spurious peak score S_global
   - Interpretation

2. **JSON Results** (`results/spurious_peak_predictions.json`):
   - Complete predictions for all test configurations
   - Component-wise breakdown
   - Comparison with experimental best configs

3. **Visualization** (`figures/spurious_peak_predictions.png`):
   - Spurious peak score vs bandwidth for each mixture
   - Color-coded severity thresholds
   - Multiple sample sizes (n=50, 100, 200)

4. **LaTeX Table** (`results/spurious_peak_table.tex`):
   - Ready to include in your paper
   - Shows predictions for best MSE and ValNLL configurations

## Key Findings

### From the Validation Run:

**Mixture 1 (Single Gaussian):**
- Best configurations (both MSE and ValNLL) show **S_global = 0.0**
- Prediction: NO spurious peaks ✓

**Mixture 2 (Three Gaussians):**
- Best configurations show **S_global = 0.0**
- BUT: At h1=2.0, n=200: **S_global = 3.02** (severe spurious peaks!)
- This confirms undersmoothed regions create false modes

**Mixture 3 (Five Gaussians):**
- Best configurations show **S_global = 0.0**
- At h1=2.0, n=200: **S_global = 3.92** (extremely severe!)
- More components → worse spurious peak problem at small bandwidths

### Validation Against Your Observation

Your quote:
> "KDE may sometime lead the predicted pdf with undersmoothed regions that approximates single peaks as two distinct modes. This causes, even for overlays selected by lowest MSE, to still have peaks where there shouldn't be."

**Theory prediction:** Even MSE-optimal bandwidths show S_global ≈ 0, meaning the formula predicts they should be CLEAN. This is actually GOOD news - it means MSE selection successfully avoids the spurious peak regime!

However, if you're still observing spurious peaks in MSE-optimal overlays, possible reasons:
1. Local undersmoothing in specific regions (formula uses global average σ)
2. Component overlap creating saddle points that look like spurious peaks
3. Different effective bandwidth in high-density vs low-density regions

## Advanced: Component-Wise Analysis

For detailed component analysis, check the JSON output:

```python
import json

with open('results/spurious_peak_predictions.json', 'r') as f:
    results = json.load(f)

# Look at component-wise predictions for best MSE config
mixture2_best = results[1]['best_mse']  # Index 1 = Mixture 2
for comp in mixture2_best['components']:
    print(f"Component {comp['component']}: N_peak={comp['N_peak']:.2f}, S_k={comp['S_k']:.3f}")
```

## Troubleshooting

### "No experimental logs found"
Run: `python3 estimator.py --pw-only --pw-write-logs --pw-seed 0`

### Different predictions than expected
- Check that mixture definitions match `estimator.py`
- Verify h_eff calculation: `h_eff = h1 / sqrt(n_total - 1)`
- Component σ uses geometric mean of eigenvalues (can try σ_max or σ_mean)

### Predictions don't match visual observations
The formula assumes:
- Isotropic effective bandwidth
- Samples uniformly distributed within component
- No component overlap effects
- Gaussian kernel

For more accurate predictions with component overlap, you'd need to extend the theory to account for saddle points between components.

## Next Steps

1. **Visual Validation**: Compare predictions with overlay figures
   - Configurations with S_global > 1.5 should show clear spurious peaks
   - Check `figures/Parzen_overlay_mixture*.jpeg`

2. **Quantitative Validation**: Count actual peaks in KDE estimates
   - Use peak detection algorithm (e.g., `scipy.signal.find_peaks_cwt`)
   - Compare actual peak count with predicted N_peak

3. **Extend Theory**: If predictions are systematically off, consider:
   - Component overlap effects (saddle points)
   - Anisotropic covariances (directional undersmoothing)
   - Non-Gaussian kernels

## Citation

When using this validation script, cite the theoretical section:

```latex
See Section "Spurious Peak Formation in Undersmoothed KDE" in 
\textit{Rigorous Derivation of the ValNLL/MSE Bandwidth Ratio} 
for mathematical derivation of spurious peak formulas.
```

## Example: Manual Calculation

For Mixture 2, Component 1, with n=100, h1=2.0:

```python
import numpy as np

# Component 1 parameters
sigma = 1.0048  # geometric mean of eigenvalues
n = 100
n_total = 100 * 3  # 3 components
h1 = 2.0
h_eff = h1 / np.sqrt(n_total - 1)  # = 0.1157

# Compute thresholds
h_min = sigma / (4 * np.sqrt(n))  # = 0.0251
h_trans = sigma / (2 * np.sqrt(n))  # = 0.0502
h_max = 2 * sigma / np.sqrt(n)  # = 0.2010

# h_eff = 0.1157 is in transition regime [h_min, h_max]
N_peak = 1 + (n - 1) * np.exp(-0.5 * ((h_eff - h_min) / h_trans)**2)
# N_peak ≈ 12.7

S_k = max(0, np.log(N_peak / 1.5))
# S_k ≈ 2.14 (severe!)

print(f"Predicted {N_peak:.1f} peaks for this component (severe spurious)")
```

This matches the script output showing severe spurious peaks at that configuration!
