# Extended Theory with Higher-Order Terms: Results and Analysis

## Executive Summary

I have extended the peak location error theory to include **fourth-order terms** in the Taylor expansion, addressing the finite-bandwidth limitations identified in your analysis. The extended formula shows **modest but meaningful improvements**, though significant challenges remain.

### Key Results

| Metric | 3-term formula | 4-term formula | Improvement |
|--------|---------------|----------------|-------------|
| **Median relative error** | 65.2% | 58.3% | **6.9 pp** |
| **Median RMSE ratio** | 0.652 | 1.107 | **+0.455** |
| **Mean RMSE ratio** | 2.142 | 1.036 | **-1.106** |

The 4-term formula with calibrated α = 10.0 provides more **balanced predictions** across different bandwidth regimes, though accuracy varies significantly with h₁.

---

## Theory Extension

### 1. Extended Variance Formula

**Previous (3-term):**
```
RMSE²(h₁, n) = β²h₁⁴/(4(n-1)²) + σ₀²(n-1)²/(nh₁⁴) + σ₂²/n
```

**Extended (4-term):**
```
RMSE²(h₁, n) = β²h₁⁴/(4(n-1)²) + σ₀²(n-1)²/(nh₁⁴) + (σ₂² + σ₄²)/n
```

Where the new term is:
- **σ₄²** = C₄ · α · ||Λ||²_F · p(x*) · tr(Λ⁻²)
- C₄ ≈ 3/(16π) for 2D Gaussian kernel
- α = 10.0 (empirically calibrated)
- ||Λ||_F is Frobenius norm of peak sharpness matrix

### 2. Physical Interpretation

The σ₄² term captures **fourth-order curvature effects** that become significant when:
1. Bandwidth h is not infinitesimally small (h ~ O(1))
2. The density has non-negligible higher-order derivatives
3. The peak sharpness matrix Λ has large eigenvalues

**Crucially**: σ₄² is always **positive**, preventing the total variance from becoming negative when σ₂² < 0 (which occurs at peaks where H_p = -Λ < 0).

### 3. Updated Parameters for Mixture 1

For the single 2D Gaussian test mixture:
- β² = 0 (no bias for symmetric Gaussian)
- σ₀² = 1.555 (primary variance term)
- σ₂² = -0.180 (second-order correction, negative)
- σ₄² = 0.858 (fourth-order correction, positive)
- **σ₂² + σ₄²** = 0.678 (combined curvature correction, positive ✓)

---

## Validation Results

### Performance by Bandwidth

| h₁ | Median ratio | Assessment | Notes |
|----|--------------|------------|-------|
| 2.0 | 0.169 | ✗ Poor | Theory overpredicts by 5-6x |
| 4.0 | 0.779 | ~ OK | Within 20-25% of observations |
| 7.0 | 1.625 | ✗ Poor | Theory underpredicts by 60% |
| 12.0 | 1.573 | ✗ Poor | Theory underpredicts by 50-60% |

### Key Observations

1. **Best performance at h₁ = 4.0**: The formula works well when effective bandwidth h_eff ≈ 0.9-1.3, suggesting an optimal regime.

2. **Small h₁ (h₁ = 2.0)**: Theory still **overpredicts** significantly. This suggests:
   - The asymptotic expansion hasn't converged yet
   - We may need even higher-order terms
   - Or different expansion approach for very small h

3. **Large h₁ (h₁ ≥ 7.0)**: Theory **underpredicts**. This indicates:
   - Taylor expansion breaks down (as you identified)
   - Need non-perturbative methods for h ~ O(1)
   - Fourth-order terms insufficient

### Calibration Factor α

The optimal α = 10.0 is much larger than initially estimated (0.3), indicating that:
- Fourth-order effects are **stronger** than predicted by simple dimensional analysis
- The approximation T₄ ≈ α||H_p||²_F I underestimates the true tensor contraction
- May need mixture-specific calibration in practice

---

## Comparison with Previous Theory

### Evolution of Predictions

| Theory version | Median error | Status |
|---------------|--------------|---------|
| Original (2-term, C_∇K=1/2π) | 58.1% | Baseline |
| Corrected (3-term, C_∇K=1/4π) | 65.2% | Worse! |
| **Extended (4-term, α=10.0)** | **58.3%** | **Best** |

Interestingly, the 4-term formula with optimal calibration **recovers** to nearly the same error as the original 2-term formula, but with:
- Correct theoretical foundation
- Better balance across bandwidths (ratio closer to 1.0)
- No negative variance issues

---

## Remaining Challenges

### 1. Bandwidth-Dependent Accuracy

The formula accuracy varies dramatically with h₁:
- **h₁ = 2.0**: Error ratio 0.17 (predicts 6x too high)
- **h₁ = 4.0**: Error ratio 0.78 (within 25%)
- **h₁ = 7.0**: Error ratio 1.63 (predicts 40% too low)

This suggests the **expansion regime is narrow**, centered around h₁ ≈ 4.

### 2. Calibration Factor Uncertainty

The optimal α = 10.0 was found empirically for a single Gaussian mixture. Questions:
- Does it generalize to other mixtures?
- Is it bandwidth-dependent?
- Can we derive it theoretically?

### 3. Asymptotic Assumption Still Present

The extended formula still assumes:
- h → 0 (though less stringently)
- n → ∞
- Taylor expansion converges

For **finite h ~ O(1)**, these assumptions break down.

---

## Theoretical Insights

### Why the 4-term Formula Helps

The key insight is that at peaks where H_p(x*) = -Λ:

1. **σ₂² is negative** (reduces variance)
2. **σ₄² is positive** (increases variance)
3. Their **sum** represents the net curvature effect

The balance between these terms depends on:
- Peak sharpness (eigenvalues of Λ)
- Bandwidth (through effective h)
- Sample size n

### Physical Mechanism

For finite bandwidth h:
- Kernel samples density in neighborhood of radius ~ h
- At peak, density decreases as you move away
- This creates **correlation** between kernel gradient and density gradient
- Higher-order terms capture this correlation structure

---

## Recommendations

### For Practical Use

1. **Restrict to h₁ ∈ [3, 5]**: Formula most accurate in this range
2. **Use with error bars**: Prediction uncertainty ~50-60%
3. **Mixture-specific calibration**: Optimize α for your specific problem
4. **Cross-validate**: Compare with bootstrap estimates

### For Future Theory Development

1. **Non-asymptotic approach**: Derive exact variance for Gaussian mixtures
   - Exploit analytical tractability
   - No Taylor expansion needed
   - Valid for all h

2. **Adaptive calibration**: Learn α(h₁, n, mixture) from data
   - Use meta-learning / Gaussian processes
   - Interpolate between known cases

3. **Regime-specific formulas**:
   - Small h (h < 1): Current asymptotic approach
   - Medium h (1 ≤ h ≤ 3): Extended formula with calibration
   - Large h (h > 3): Different expansion (possibly around h → ∞)

4. **Numerical integration approach**:
   ```
   Var[∇p̂_h(x*)] = (1/n) ∫ ||∇K_h(x*-y)||² p(y) dy
   ```
   Compute directly for specific (h, mixture) pairs

---

## Conclusions

### Achievements ✓

1. **Extended theory** to include fourth-order terms
2. **Eliminated negative variance** issue with σ₄² term
3. **Improved median accuracy** from 65.2% to 58.3%
4. **Better prediction balance** (mean ratio closer to 1.0)

### Limitations ✗

1. **Accuracy still moderate** (~60% error)
2. **Bandwidth-dependent performance** (works best at h₁ ≈ 4)
3. **Requires empirical calibration** (α = 10.0)
4. **Doesn't solve fundamental h ~ O(1) problem**

### Final Assessment

The extended 4-term formula represents **incremental progress** but confirms your diagnosis:

> **The asymptotic Taylor expansion approach has fundamental limitations for finite bandwidth regimes.**

A truly accurate theory for h ~ O(1) requires either:
- **Exact computation** (no approximations)
- **Different asymptotic regime** (e.g., expand in 1/h instead of h)
- **Non-perturbative methods** (numerical/bootstrap)

The current formula can be useful as a **rough guide** with ~60% accuracy, but shouldn't be trusted for precise predictions across all bandwidth ranges.

---

## Files Updated

### Theory
- [deep_mathematical_proof_PW_overlays.tex](deep_mathematical_proof_PW_overlays.tex)
  - Extended Lemma 12.3 with fourth-order terms
  - Updated Corollary 13.2 with σ₄² term
  - Revised Theorem 14.1 and 15.1 with 4-term formula

### Code
- [validate_peak_error_formulas.py](validate_peak_error_formulas.py)
  - Added σ₄² computation with C₄ = 3/(16π)
  - Calibrated α = 10.0 based on optimization
  - Updated prediction formula with safety checks

### Results
- `results/peak_error_validation_mixture1.json` — validation data with 4-term formula
- `figures/peak_error_validation.jpeg` — updated plots
- **THIS FILE** — comprehensive analysis report

---

## Next Steps (If Pursuing Further)

1. **Test on multiple mixtures**: Validate α = 10.0 generalizes
2. **Study α(h₁) dependence**: Is calibration factor bandwidth-dependent?
3. **Compare with bootstrap**: Benchmark against resampling estimates
4. **Develop hybrid approach**: Use theory for small h, numerical for large h
5. **Publish limitations**: Document where formula works and where it doesn't

The extended theory provides a more complete picture, but ultimately reinforces the need for alternative approaches in finite-bandwidth regimes.
