#!/usr/bin/env python3
"""
Visual Peak Counter for KDE Estimates

This script loads actual KDE estimates and counts the number of distinct peaks
to validate the spurious peak theory predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label
from scipy.signal import find_peaks
from typing import List, Tuple, Dict
import json
import os


def count_peaks_2d(density_grid: np.ndarray, prominence_threshold: float = 0.01) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Count distinct peaks in a 2D density estimate.
    
    A peak is a local maximum that is significantly higher than its surroundings.
    
    Args:
        density_grid: 2D array of density values
        prominence_threshold: Relative prominence (as fraction of max density)
    
    Returns:
        (num_peaks, peak_locations)
    """
    # Normalize density to [0, 1]
    density_norm = (density_grid - density_grid.min()) / (density_grid.max() - density_grid.min() + 1e-10)
    
    # Find local maxima using maximum filter
    # A point is a local maximum if it equals the maximum in its neighborhood
    neighborhood_size = 5
    local_max = maximum_filter(density_norm, size=neighborhood_size) == density_norm
    
    # Remove edge artifacts
    local_max[:2, :] = False
    local_max[-2:, :] = False
    local_max[:, :2] = False
    local_max[:, -2:] = False
    
    # Only keep peaks above prominence threshold
    abs_threshold = prominence_threshold
    significant_peaks = local_max & (density_norm > abs_threshold)
    
    # Label connected components (handles plateaus)
    labeled_peaks, num_peaks = label(significant_peaks)
    
    # Find peak locations (centroids of labeled regions)
    peak_locations = []
    for peak_id in range(1, num_peaks + 1):
        peak_mask = labeled_peaks == peak_id
        coords = np.argwhere(peak_mask)
        if len(coords) > 0:
            centroid = coords.mean(axis=0)
            peak_locations.append((int(centroid[0]), int(centroid[1])))
    
    return num_peaks, peak_locations


def simulate_kde_from_mixture(
    mixture_idx: int,
    n_per_component: int,
    h1: float,
    grid_size: int = 100,
    seed: int = 0
) -> np.ndarray:
    """
    Simulate KDE estimate for a mixture configuration.
    
    This recreates what estimator.py does to generate KDE estimates.
    """
    # Load mixture definitions (same as validate_spurious_peaks.py)
    from validate_spurious_peaks import load_mixture_definitions
    
    mixtures = load_mixture_definitions()
    mixture = mixtures[mixture_idx]
    
    # Set random seed
    np.random.seed(seed + 1000 * mixture_idx + n_per_component)
    
    # Generate samples from each component
    all_samples = []
    for comp in mixture.components:
        # Number of samples proportional to weight
        n_comp = int(np.round(n_per_component * comp.weight))
        if n_comp == 0:
            continue
        
        # Sample from multivariate Gaussian
        samples = np.random.multivariate_normal(comp.mean, comp.covariance, n_comp)
        all_samples.append(samples)
    
    if not all_samples:
        return np.zeros((grid_size, grid_size))
    
    samples = np.vstack(all_samples)
    n_total = samples.shape[0]
    
    # Compute effective bandwidth (adaptive scaling)
    h_eff = h1 / np.sqrt(n_total - 1) if n_total > 1 else h1
    
    # Create evaluation grid
    x_range = (-5, 5)
    y_range = (-5, 5)
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Compute KDE
    density = np.zeros(grid_points.shape[0])
    
    for sample_point in samples:
        # Gaussian kernel
        diff = grid_points - sample_point
        exponent = -0.5 * np.sum(diff**2, axis=1) / (h_eff**2)
        density += np.exp(exponent)
    
    # Normalize
    density = density / (n_total * 2 * np.pi * h_eff**2)
    density_grid = density.reshape(grid_size, grid_size)
    
    return density_grid


def analyze_configuration(
    mixture_idx: int,
    n: int,
    h1: float,
    prominence_thresholds: List[float] = [0.01, 0.02, 0.05]
) -> Dict:
    """
    Analyze a specific KDE configuration and count peaks.
    """
    print(f"\nAnalyzing Mixture {mixture_idx}, n={n}, h1={h1:.2f}")
    
    # Generate KDE estimate
    density_grid = simulate_kde_from_mixture(mixture_idx, n, h1)
    
    # Count peaks at different prominence thresholds
    results = {
        'mixture_idx': mixture_idx,
        'n': n,
        'h1': h1,
        'prominence_analysis': []
    }
    
    for prom_thresh in prominence_thresholds:
        num_peaks, peak_locs = count_peaks_2d(density_grid, prom_thresh)
        
        results['prominence_analysis'].append({
            'prominence_threshold': prom_thresh,
            'num_peaks': num_peaks,
            'peak_locations': peak_locs
        })
        
        print(f"  Prominence {prom_thresh:.3f}: {num_peaks} peaks detected")
    
    return results


def visualize_peak_detection(
    density_grid: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    title: str,
    output_path: str
):
    """
    Visualize density with detected peaks marked.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot density as heatmap
    im = ax.imshow(density_grid, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Density')
    
    # Mark peaks
    for i, (y, x) in enumerate(peak_locations, start=1):
        ax.plot(x, y, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=1.5)
        ax.text(x, y, str(i), color='white', fontsize=12, ha='center', va='center',
                fontweight='bold')
    
    ax.set_title(f"{title}\n{len(peak_locations)} peaks detected")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved: {output_path}")


def compare_with_predictions():
    """
    Load theoretical predictions and compare with actual peak counts.
    """
    predictions_path = "results/spurious_peak_predictions.json"
    
    if not os.path.exists(predictions_path):
        print("Predictions not found. Run validate_spurious_peaks.py first.")
        return
    
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    print("\n" + "="*80)
    print("COMPARING THEORETICAL PREDICTIONS WITH ACTUAL PEAK COUNTS")
    print("="*80)
    
    # Test configurations that should show spurious peaks
    test_cases = [
        # (mixture_idx, n, h1, expected_severity)
        (2, 100, 2.0, "severe"),  # Mixture 2, should have S_global = 2.35
        (3, 100, 2.0, "severe"),  # Mixture 3, should have S_global = 3.24
        (2, 100, 7.0, "clean"),   # Should be clean
        (3, 100, 7.0, "clean"),   # Should be clean
    ]
    
    all_results = []
    
    for mixture_idx, n, h1, expected in test_cases:
        print(f"\n{'─'*80}")
        print(f"Test Case: Mixture {mixture_idx}, n={n}, h1={h1:.2f}")
        print(f"Expected: {expected}")
        
        # Get theoretical prediction
        mixture_pred = predictions[mixture_idx - 1]
        matching_config = None
        for config in mixture_pred['configurations']:
            if config['n'] == n and abs(config['h1'] - h1) < 0.01:
                matching_config = config
                break
        
        if matching_config:
            print(f"Theoretical S_global: {matching_config['S_global']:.3f}")
            print(f"Theoretical interpretation: {matching_config['interpretation']}")
            
            # Estimate expected number of peaks from components
            total_expected_peaks = sum(
                comp['N_peak'] for comp in matching_config['components']
            )
            print(f"Theoretical total N_peak: {total_expected_peaks:.1f}")
        
        # Count actual peaks
        result = analyze_configuration(mixture_idx, n, h1, prominence_thresholds=[0.02])
        
        actual_peaks = result['prominence_analysis'][0]['num_peaks']
        print(f"Actual peaks counted: {actual_peaks}")
        
        # Visualize
        density_grid = simulate_kde_from_mixture(mixture_idx, n, h1)
        peak_locs = result['prominence_analysis'][0]['peak_locations']
        
        vis_path = f"figures/peak_detection_mixture{mixture_idx}_n{n}_h{h1:.0f}.png"
        visualize_peak_detection(
            density_grid,
            peak_locs,
            f"Mixture {mixture_idx}, n={n}, h1={h1:.2f}",
            vis_path
        )
        
        result['theoretical'] = matching_config
        result['actual_peaks'] = actual_peaks
        all_results.append(result)
    
    # Save comparison results
    output_path = "results/peak_count_validation.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Comparison results saved: {output_path}")
    print(f"Peak detection visualizations saved in figures/")
    print(f"{'='*80}\n")


def main():
    """Main validation routine."""
    print("="*80)
    print("VISUAL PEAK COUNTING VALIDATION")
    print("="*80)
    print("\nThis script generates KDE estimates and counts actual peaks")
    print("to validate spurious peak theory predictions.\n")
    
    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Run comparison
    compare_with_predictions()
    
    print("\nVALIDATION COMPLETE!")
    print("\nNext steps:")
    print("1. Check figures/peak_detection_*.png for visual confirmation")
    print("2. Compare actual peak counts with theoretical predictions")
    print("3. If predictions are off, consider:")
    print("   - Adjusting prominence threshold")
    print("   - Using component-specific σ (max vs geometric vs mean)")
    print("   - Accounting for component overlap effects")


if __name__ == "__main__":
    main()
