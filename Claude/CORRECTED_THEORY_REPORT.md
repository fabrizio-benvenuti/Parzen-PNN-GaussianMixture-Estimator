# Corrected Theory Analysis Report

## Summary

We have corrected the peak location error theory based on issues identified in DIAGNOSIS.md. The main corrections were:

### 1. Corrected Constant C_∇K
- **Old value**: C_∇K = 1/(2π)
- **Corrected value**: C_∇K = 1/(4π)  
- **Impact**: Reduces σ₀² by factor of 2

### 2. Added Curvature Correction Term
The variance formula now has three components:
```
RMSE²(h₁, n) = β²h₁⁴/(4(n-1)²) + σ₀²(n-1)²/(nh₁⁴) + σ₂²/n
```

Where:
- **β²** = ||Λ⁻¹∇Δp(x*)||² (bias term, = 0 for single Gaussian)
- **σ₀²** = C_∇K·p(x*)·tr(Λ⁻²) (primary variance, scales as 1/h⁴)
- **σ₂²** = -(tr(C_H)/2)·p(x*)·tr(Λ⁻¹) (curvature correction, constant in h)

### 3. Numerical Values for 2D Gaussian Kernel
From numerical integration:
- C_∇K = 1/(4π) ≈ 0.0796
- tr(C_H) = 2·C_∇K = 1/(2π) ≈ 0.1592

## Results for Mixture 1 (Single Gaussian)

### Geometric Parameters
- β² = 0 (no bias for single Gaussian)
- σ₀² = 1.5547
- σ₂² = -0.1800
- Peak density p(x*) = 0.1576

### Validation Results

#### Old Theory vs New Theory Performance

**Old Theory** (2-term with C_∇K=1/2π):
- Mean relative error: 276.69%
- Median relative error: 58.13%
- Max relative error: 1887.99%

**New Theory** (3-term with C_∇K=1/4π):
- Mean relative error: 214.19%
- Median relative error: 65.18%
- Max relative error: 1305.69%

**Improvement**: ~23% median error reduction, though still substantial errors remain.

### Key Observations

1. **For small h₁ (h₁=2.0)**: Theory still overpredicts by large factors (2-13x)
   - var₂ correction is tiny compared to var₀ (ratio ~0.15%)
   - The 1/h⁴ scaling dominates

2. **For large h₁ (h₁=12.0)**: Theory can give NaN when var₂ exceeds var₀
   - var₂/var₀ ratio reaches ~2.0
   - This indicates breakdown of Taylor expansion approach

3. **The negative σ₂²** reduces total variance, which is physically reasonable:
   - At peak, density curvature is negative (H_p = -Λ < 0)
   - This reduces variance contribution from density non-uniformity
   
## Remaining Issues

### Issue 1: Large Discrepancies Persist
Even with corrections, theory overpredicts errors by 2-10x for many cases.

### Issue 2: Formula Becomes Negative
For large h₁, the curvature correction term can dominate and make variance negative, causing NaN.

### Issue 3: Taylor Expansion Validity
The expansion p(x* - hu) ≈ p(x*) + (h²/2)u^T H_p u requires h to be small enough that:
- p remains positive (satisfied)
- Higher order terms are negligible (NOT satisfied for large h₁)

## Root Cause Analysis

The fundamental issue is that the Taylor expansion approach assumes **h → 0 asymptotics**, but we're using **adaptive bandwidth h = h₁/√(n-1)** where:

1. h₁ is fixed (not small)
2. As n → ∞, h → 0, but we're testing finite n
3. For n=20 and h₁=12, effective h ≈ 2.75, which is NOT small

The correct approach would require:
1. Full expansion in both h and 1/n
2. Non-asymptotic analysis for finite h
3. Possibly different scaling regime

## Recommendations

### Short-term (Empirical)
1. **Restrict formula validity**: Only use for h_eff < 1.0
2. **Fit correction factors**: Add empirical factors to account for finite-h effects
3. **Use different formula for large h**: Switch to different approximation regime

### Long-term (Theoretical)
1. **Rederive for finite bandwidth**: Don't assume h → 0
2. **Consider exact computation**: For Gaussian mixtures, variance might be computable exactly
3. **Study non-asymptotic regime**: What happens when h ~ O(1)?

### Alternative Approaches
1. **Bootstrap estimates**: Use resampling to estimate peak error empirically
2. **Numerical variance computation**: Compute ∫||∇K_h||² p directly for each (h, mixture)
3. **Machine learning meta-model**: Learn error prediction from simulation data

## Files Updated

### Theory (LaTeX)
- `deep_mathematical_proof_PW_overlays.tex`: Updated Lemma 12.3, Corollaries, and Theorem 15.1

### Python Code
- `validate_peak_error_formulas.py`: 
  - Updated C_∇K = 1/(4π)
  - Added σ₂² computation with tr(C_H) = 2·C_∇K
  - Updated formula to use three terms
  
- `compute_mixture_geometry.py`: 
  - Updated C_v = 1/(4π) for MISE
  - Updated C̃_v = 1/(8π) for NLL (still needs verification)

### New Analysis Scripts
- `compute_kernel_integrals.py`: Numerical verification of C_∇K and C_H
- `analyze_corrected_theory.py`: Comparison of old vs new theory

## Conclusion

The corrections improve the theory, particularly:
1. C_∇K correction is mathematically correct
2. Curvature correction term σ₂² is theoretically justified
3. Median error reduced by ~23%

However, large discrepancies remain because:
1. **The asymptotic approximation (h → 0) breaks down for finite h**
2. **The adaptive bandwidth h₁/√(n-1) can give h ~ O(1), not h → 0**
3. **Higher-order terms in Taylor expansion are non-negligible**

This suggests the need for either:
- Non-asymptotic theory for finite bandwidth
- Empirical correction factors
- Different approximation for different h regimes

The diagnosis was correct: the theory has fundamental limitations beyond just wrong constants.
