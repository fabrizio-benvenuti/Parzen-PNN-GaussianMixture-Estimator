# DIAGNOSIS: Discrepancies Between Theory and Observations

## Summary of Issues Found

### Issue 1: Wrong Constant C_∇K

**Location**: LaTeX document, Lemma 12.3

**Claimed**: C_∇K = 1/(2π) for 2D Gaussian kernel

**Correct**: C_∇K = 1/(4π)

**Evidence**:
- Analytical derivation: ∫||∇K_h(u)||² du = 1/(4π) when properly scaled
- Numerical verification confirms: numerical integral * h⁴ = 0.07958 ≈ 1/(4π)

**Impact**: Factor of 2 error in σ², but this alone doesn't explain the large discrepancies observed.

---

### Issue 2: Variance Formula Assumes Uniform Density (MAIN ISSUE)

**Location**: LaTeX document, Lemma 12.3

**Claimed**:
```
Cov[∇δp(x*)] = (C_∇K)/(nh⁴) * p(x*) * I
```

**Problem**: This formula assumes the variance scales exactly as `1/h⁴`, which is only true when:
1. The true density p(y) is locally constant around x*
2. Or equivalently, when computing ∫||∇K_h(u)||² du (no weighting by density)

**Reality**: When X ~ p(y) with non-uniform p, we have:
```
Var[∇K_h(x* - X)] = ∫||∇K_h(x* - y)||² p(y) dy
```

This integral does NOT scale as `p(x*)/h⁴` in general!

**Evidence**:
- Empirical tests show Var[∇K_h,x] * h⁴ varies with h:
  - h=0.5: Var * h⁴ = 0.00500
  - h=1.0: Var * h⁴ = 0.00281
  - h=2.0: Var * h⁴ = 0.00070
- This is NOT constant as the formula claims!

- Empirical covariance is only 0.22-0.23 times the theoretical prediction

**Why it happens**:
- For small h: The kernel ∇K_h(x* - y) is narrow, so it only samples p(y) very close to x*
- For large h: The kernel is wide, sampling p(y) over a larger region
- The convolution ∫||∇K_h||² * p does NOT factor as p(x*) * ∫||∇K_h||²!

---

### Issue 3: Peak Displacement Formula Breaks Down

**Consequence of Issue 2**:

The peak displacement formula:
```
RMSE_peak(h1, n) = sqrt(β²*h1⁴/(4(n-1)²) + σ²*(n-1)²/(n*h1⁴))
```

is **fundamentally incorrect** because σ² is not a constant independent of h.

**Observed behavior**:
- Small h1 (2.0): Theory overpredicts by factor of 2-10 (variance term too large)
- Large h1 (12.0): Theory underpredicts by factor of 2-7 (variance term too small)

This is exactly what we'd expect if the variance doesn't scale as 1/h⁴!

---

### Issue 4: Bandwidth Ratio Formula Also Affected

**Location**: Step 8 in LaTeX (bandwidth ratio)

**Formula**:
```
h1_NLL / h1_MSE = (2ρ)^(1/6) * sqrt(n_NLL / n_MSE)
```

**Observed vs Predicted**:
- Mixture 1: Observed = 0.411, Predicted = 1.122 (173% error!)
- Mixture 2: Observed = 0.576, Predicted = 1.122 (95% error!)
- Mixture 3: Observed = 0.501, Predicted = 1.122 (124% error!)

**Why it fails**: The formula depends on ρ (curvature concentration ratio), which is always computed as 1.0. This suggests:
1. The computation of ρ might be wrong, or
2. The relationship between MSE and NLL optimal bandwidths is more complex

---

## Root Cause Analysis

The fundamental problem is in **Lemma 12.3** of the LaTeX document.

The derivation states:
```
Cov[∇δp(x*)] = (1/n) * Var[∇K_h(x* - X)]
               = (1/n) * ∫||∇K_h(x* - y)||² p(y) dy
```

Then it claims:
```
               ≈ (1/n) * p(x*) * ∫||∇K_h(u)||² du
               = (C_∇K * p(x*))/(n * h⁴) * I
```

**This approximation is WRONG!**

The factorization `∫||∇K_h(x* - y)||² p(y) dy ≈ p(x*) * ∫||∇K_h(u)||² du` is only valid if:
1. p(y) is constant in the support of ∇K_h, OR
2. We're taking h → 0 limit and p is continuous

But with adaptive bandwidth h = h1/sqrt(n-1):
- As n increases, h decreases
- But we're not taking h → 0 for fixed n!
- Instead, we're asking how error scales with BOTH n and h1

The correct approach would be to derive how the variance integral depends on BOTH h and the local geometry of p near x*.

---

## Recommendations

### Short-term Fix (Empirical):
1. Fit the actual variance behavior from simulations
2. Replace σ²/(nh⁴) with a more complex function of h and n
3. This would be purely empirical, not theoretical

### Long-term Fix (Theoretical):
1. Redo the derivation in Lemma 12.3 more carefully
2. Account for the non-uniform density in the variance calculation
3. Derive how ∫||∇K_h(x* - y)||² p(y) dy depends on h when p is Gaussian or Gaussian mixture
4. This will likely give additional terms involving second derivatives of p

### Mathematical Direction:
The correct variance should probably involve terms like:
```
Var[∇K_h] = (1/(nh⁴)) * [A(x*) + B(x*) * h² + O(h⁴)]
```

where:
- A(x*) involves p(x*) and kernel moments
- B(x*) involves ∇²p(x*) (curvature of density)

This would explain why the simple 1/h⁴ scaling fails!

---

## Conclusion

**The theory in the LaTeX document has fundamental errors in the variance derivation.**

The formula for peak location error is not correct as stated. It needs to be rederived accounting for:
1. The correct constant (1/(4π) not 1/(2π))
2. The non-uniform density in the variance calculation
3. Higher-order terms in the h expansion

The observed discrepancies are NOT due to implementation errors, but due to **theoretical errors** in the LaTeX document.
