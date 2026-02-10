# Debug: Peak Error Formula Derivation

## Theory from LaTeX (Step 13.2)

From Corollary 13.2, using bandwidth `h`:

```
Bias² = h⁴/4 * ||Λ⁻¹ ∇Δp(x*)||²  = h⁴/4 * β²
Variance = (C_∇K * p(x*))/(n*h⁴) * tr(Λ⁻²) = σ²/(n*h⁴)
```

Where:
- β² = ||Λ⁻¹ ∇Δp(x*)||²
- σ² = C_∇K * p(x*) * tr(Λ⁻²)
- C_∇K = 1/(2π) for 2D Gaussian kernel

So:
```
RMSE²_peak(h, n) = β²*h⁴/4 + σ²/(n*h⁴)
```

## Conversion to h1 (Step 15.1)

Under adaptive bandwidth scaling: `h = h1/sqrt(n-1)`

Substituting:
```
h⁴ = h1⁴/(n-1)²

RMSE²_peak(h1, n) = β²*h1⁴/(4*(n-1)²) + σ²*(n-1)²/(n*h1⁴)
```

## Implementation in Code

In validate_peak_error_formulas.py (lines 423-426):
```python
n = n_total
bias_term = beta_sq * h1**4 / (4 * (n - 1)**2)
var_term = sigma_sq * (n - 1)**2 / (n * h1**4)
predicted_rmse = float(np.sqrt(bias_term + var_term))
```

**This matches the LaTeX formula exactly!**

## Issue Analysis

Let me check the observed vs predicted pattern more carefully:

For small h1 (2.0): Observed << Predicted
- This means the theory overpredicts the error
- The variance term dominates: σ²*(n-1)²/(n*h1⁴)
- For h1=2, n=20: var_term = 3.109*(19)²/(20*2⁴) = 3.507
- For h1=2, n=200: var_term = 3.109*(199)²/(200*2⁴) = 38.45
- But observed errors are much smaller!

For large h1 (12.0): Observed >> Predicted  
- This means the theory underpredicts the error
- The bias term is near zero (β²=0 for single Gaussian)
- The variance term is tiny: σ²*(n-1)²/(n*h1⁴) 
- For h1=12, n=20: var_term = 3.109*(19)²/(20*12⁴) = 0.00271
- But observed errors are much larger!

## Hypothesis: The formula is wrong!

Looking at the derivation in Step 12.3 (Variance Gradient):

```latex
\mathrm{Cov}[\nabla \delta p(x^*)] = \frac{C_{\nabla K}}{nh^4} p(x^*) I + O((nh^4)^{-1}),
```

**But wait!** This is only valid when we're averaging over the TRUE density p(y).
In practice, we're estimating from n samples with adaptive bandwidth h = h1/sqrt(n-1).

The issue is that the formula assumes:
1. Fixed h (not adaptive)
2. The covariance scales as 1/(n*h⁴)

But with adaptive bandwidth h = h1/sqrt(n-1):
- When n increases, h DECREASES
- So 1/(n*h⁴) = (n-1)²/(n*h1⁴) grows with n!

This explains why the variance term explodes with n for small h1!

## The Real Problem

The theory in the LaTeX document derives formulas for FIXED bandwidth h,
then naively substitutes h = h1/sqrt(n-1).

But the variance derivation in Lemma 12.3 is not valid under this substitution!

The variance of ∇δp depends on:
1. The number of samples n
2. The bandwidth h used for each kernel

When h is adaptive (shrinking with n), the variance calculation needs to account 
for the fact that we're averaging n increasingly narrow kernels.

## What Should the Formula Be?

The correct variance term should probably be:
```
Variance ∝ σ²/(n*h⁴)  [using the ACTUAL h, not h1]
```

So with h = h1/sqrt(n-1):
```
Variance = σ²/(n*(h1/sqrt(n-1))⁴) = σ²*(n-1)²/(n*h1⁴)
```

Wait, that's what we have! So the formula is algebraically correct...

## Alternative Hypothesis: σ² is Wrong!

Let me check how σ² is computed. From Step 13.2:
```
σ² = C_∇K * p(x*) * tr(Λ⁻²)
```

where C_∇K = 1/(2π).

But maybe the constant is wrong? Or the derivation of the gradient kernel variance?

Let me check the actual gradient kernel...
