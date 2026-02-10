#!/usr/bin/env python3
"""
Validate Spurious Peak Theory

This script implements the spurious peak formulas derived in Section "Spurious Peak Formation
in Undersmoothed KDE" of deep_mathematical_proof_PW_overlays.tex.

It computes predicted spurious peak scores for each mixture configuration and compares
them with observed behavior from experimental logs and results.
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class GaussianComponent:
    """Represents a single Gaussian component in a mixture."""
    mean: np.ndarray
    covariance: np.ndarray
    weight: float
    
    @property
    def sigma(self) -> float:
        """Compute effective standard deviation (geometric mean of eigenvalues)."""
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        # Geometric mean of standard deviations
        return float(np.sqrt(np.prod(np.sqrt(eigenvalues))))
    
    @property
    def sigma_max(self) -> float:
        """Maximum standard deviation (from largest eigenvalue)."""
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return float(np.sqrt(np.max(eigenvalues)))
    
    @property
    def sigma_mean(self) -> float:
        """Mean standard deviation (arithmetic mean of eigenvalues)."""
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return float(np.mean(np.sqrt(eigenvalues)))


@dataclass
class MixtureDefinition:
    """Represents a Gaussian mixture."""
    components: List[GaussianComponent]
    name: str
    
    def print_summary(self):
        """Print mixture statistics."""
        print(f"\n{self.name}:")
        print(f"  Number of components: {len(self.components)}")
        for i, comp in enumerate(self.components):
            print(f"  Component {i+1}:")
            print(f"    Weight: {comp.weight:.3f}")
            print(f"    Mean: {comp.mean}")
            print(f"    σ_geometric: {comp.sigma:.4f}")
            print(f"    σ_max: {comp.sigma_max:.4f}")
            print(f"    σ_mean: {comp.sigma_mean:.4f}")


def load_mixture_definitions() -> Dict[int, MixtureDefinition]:
    """
    Load mixture definitions from estimator.py.
    These are the exact mixtures used in experiments.
    """
    # From estimator.py lines 1905-1913
    g1 = GaussianComponent(
        mean=np.array([1, 2]),
        covariance=np.array([[1.62350208, -0.13337813], [-0.13337813, 0.63889251]]),
        weight=1.0  # Will be overridden per mixture
    )
    g2 = GaussianComponent(
        mean=np.array([-2, -1]),
        covariance=np.array([[1.14822883, 0.19240818], [0.19240818, 1.23432651]]),
        weight=1.0
    )
    g3 = GaussianComponent(
        mean=np.array([-1, 3]),
        covariance=np.array([[0.30198015, 0.13745508], [0.13745508, 1.69483031]]),
        weight=1.0
    )
    g4 = GaussianComponent(
        mean=np.array([1.5, -0.5]),
        covariance=np.array([[0.85553671, -0.19601649], [-0.19601649, 0.7507167]]),
        weight=1.0
    )
    g5 = GaussianComponent(
        mean=np.array([-3, 2]),
        covariance=np.array([[0.42437194, -0.17066673], [-0.17066673, 2.16117758]]),
        weight=1.0
    )
    
    # Mixture 1: single component
    mixture1 = MixtureDefinition(
        components=[GaussianComponent(g1.mean, g1.covariance, 1.0)],
        name="Mixture 1 (Single Gaussian)"
    )
    
    # Mixture 2: three components
    weights2 = [0.3, 0.3, 0.4]
    mixture2 = MixtureDefinition(
        components=[
            GaussianComponent(g1.mean, g1.covariance, weights2[0]),
            GaussianComponent(g2.mean, g2.covariance, weights2[1]),
            GaussianComponent(g3.mean, g3.covariance, weights2[2]),
        ],
        name="Mixture 2 (Three Gaussians)"
    )
    
    # Mixture 3: five components
    weights3 = [0.2, 0.2, 0.2, 0.2, 0.2]
    mixture3 = MixtureDefinition(
        components=[
            GaussianComponent(g1.mean, g1.covariance, weights3[0]),
            GaussianComponent(g2.mean, g2.covariance, weights3[1]),
            GaussianComponent(g3.mean, g3.covariance, weights3[2]),
            GaussianComponent(g4.mean, g4.covariance, weights3[3]),
            GaussianComponent(g5.mean, g5.covariance, weights3[4]),
        ],
        name="Mixture 3 (Five Gaussians)"
    )
    
    return {1: mixture1, 2: mixture2, 3: mixture3}


def compute_spurious_peak_count(h: float, n: int, sigma: float) -> float:
    """
    Compute expected number of distinct peaks for a single component.
    
    Formula from deep_mathematical_proof_PW_overlays.tex Section "Spurious Peak Formation":
    
    N_peak(h, n, σ) = {
        n                                           if h < h_min
        1 + (n-1) * exp(-0.5 * ((h - h_min)/h_trans)^2)   if h_min ≤ h ≤ h_max
        1                                           if h > h_max
    }
    
    where:
        h_min = σ / (4√n)
        h_trans = σ / (2√n)
        h_max = 2σ / √n
    
    Args:
        h: Bandwidth parameter
        n: Number of samples for this component
        sigma: Standard deviation of the component
    
    Returns:
        Expected number of distinct peaks
    """
    if n <= 1:
        return 1.0
    
    sqrt_n = np.sqrt(n)
    h_min = sigma / (4 * sqrt_n)
    h_trans = sigma / (2 * sqrt_n)
    h_max = 2 * sigma / sqrt_n
    
    if h < h_min:
        return float(n)
    elif h > h_max:
        return 1.0
    else:
        # Transition regime
        exponent = -0.5 * ((h - h_min) / h_trans) ** 2
        return 1.0 + (n - 1) * np.exp(exponent)


def compute_spurious_peak_score(N_peak: float) -> float:
    """
    Compute spurious peak score for a component.
    
    S_k = max{0, log(N_peak / 1.5)}
    
    A component has spurious peaks when S_k > 0 (more than 1.5 peaks on average).
    """
    return max(0.0, np.log(N_peak / 1.5))


def compute_global_spurious_score(
    mixture: MixtureDefinition,
    h: float,
    n_per_component: int
) -> Tuple[float, List[Dict]]:
    """
    Compute global spurious peak score for entire mixture.
    
    S_global = Σ_k w_k * S_k
    
    Args:
        mixture: Mixture definition
        h: Bandwidth parameter
        n_per_component: Number of samples per component
    
    Returns:
        (global_score, component_details)
    """
    component_details = []
    global_score = 0.0
    
    for i, comp in enumerate(mixture.components):
        # Use geometric mean sigma as default
        sigma = comp.sigma
        
        N_peak = compute_spurious_peak_count(h, n_per_component, sigma)
        S_k = compute_spurious_peak_score(N_peak)
        
        component_details.append({
            'component': i + 1,
            'weight': comp.weight,
            'sigma': sigma,
            'N_peak': N_peak,
            'S_k': S_k,
            'weighted_score': comp.weight * S_k
        })
        
        global_score += comp.weight * S_k
    
    return global_score, component_details


def interpret_spurious_score(S_global: float) -> str:
    """Interpret global spurious peak score."""
    if S_global == 0:
        return "No spurious peaks (clean)"
    elif S_global < 0.5:
        return "Mild spurious peaks (barely visible)"
    elif S_global < 1.5:
        return "Moderate spurious peaks (clearly visible)"
    else:
        return "Severe spurious peaks (multiple false modes)"


def load_pw_logs() -> Dict[int, List[Dict]]:
    """
    Load Parzen Window error logs if available.
    
    Returns:
        Dictionary mapping mixture_idx -> list of experiment records
    """
    logs_dir = "logs"
    results = {}
    
    for mixture_idx in [1, 2, 3]:
        csv_path = os.path.join(logs_dir, f"pw_errors_mixture{mixture_idx}.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        records = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'n': int(row['n']),
                    'h1': float(row['h1']),
                    'h_n': float(row['h_n']),
                    'grid_mse': float(row['grid_mse']),
                    'val_nll': float(row['val_avg_nll']),
                })
        
        results[mixture_idx] = records
    
    return results


def load_sweep_results() -> Dict[int, Dict]:
    """Load sweep results JSON files if available."""
    results = {}
    
    for mixture_idx in [1, 2, 3]:
        json_path = f"results/sweep_results_mixture{mixture_idx}.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results[mixture_idx] = json.load(f)
    
    return results


def analyze_mixture(
    mixture_idx: int,
    mixture: MixtureDefinition,
    pw_logs: Optional[List[Dict]] = None
) -> Dict:
    """
    Perform comprehensive spurious peak analysis for a mixture.
    
    Args:
        mixture_idx: Mixture index (1, 2, or 3)
        mixture: Mixture definition
        pw_logs: Optional Parzen Window experiment logs
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*80}")
    print(f"SPURIOUS PEAK ANALYSIS: {mixture.name}")
    print(f"{'='*80}")
    
    mixture.print_summary()
    
    # Test different configurations
    test_configs = [
        # Small bandwidth (expect many spurious peaks)
        {'n': 50, 'h1_values': [2.0, 7.0, 12.0, 16.0]},
        {'n': 100, 'h1_values': [2.0, 7.0, 12.0, 16.0]},
        {'n': 200, 'h1_values': [2.0, 7.0, 12.0, 16.0]},
    ]
    
    results = {
        'mixture_idx': mixture_idx,
        'mixture_name': mixture.name,
        'configurations': []
    }
    
    print(f"\n{'─'*80}")
    print("SPURIOUS PEAK PREDICTIONS")
    print(f"{'─'*80}")
    print(f"{'n':>6} {'h1':>8} {'h_eff':>8} {'S_global':>10} {'Interpretation':>30}")
    print(f"{'─'*80}")
    
    for config in test_configs:
        n = config['n']
        
        for h1 in config['h1_values']:
            # Compute effective bandwidth (adaptive scaling)
            n_samples = n * len(mixture.components)  # Total samples
            h_eff = h1 / np.sqrt(n_samples - 1) if n_samples > 1 else h1
            
            # Compute spurious peak score
            S_global, comp_details = compute_global_spurious_score(
                mixture, h_eff, n
            )
            
            interpretation = interpret_spurious_score(S_global)
            
            print(f"{n:6d} {h1:8.2f} {h_eff:8.4f} {S_global:10.4f} {interpretation:>30}")
            
            results['configurations'].append({
                'n': n,
                'h1': h1,
                'h_eff': h_eff,
                'S_global': S_global,
                'interpretation': interpretation,
                'components': comp_details
            })
    
    # If we have experimental logs, compare predictions with observations
    if pw_logs:
        print(f"\n{'─'*80}")
        print("COMPARISON WITH EXPERIMENTAL DATA")
        print(f"{'─'*80}")
        
        # Find best configurations from logs
        best_by_mse = min(pw_logs, key=lambda x: x['grid_mse'])
        best_by_nll = min(pw_logs, key=lambda x: x['val_nll'])
        
        print(f"\nBest by MSE:")
        print(f"  n={best_by_mse['n']}, h1={best_by_mse['h1']:.2f}, h_eff={best_by_mse['h_n']:.4f}")
        print(f"  MSE={best_by_mse['grid_mse']:.6e}")
        
        # Compute spurious score for best-by-MSE
        S_mse, comp_mse = compute_global_spurious_score(
            mixture, best_by_mse['h_n'], best_by_mse['n']
        )
        print(f"  Predicted spurious score: {S_mse:.4f} ({interpret_spurious_score(S_mse)})")
        
        print(f"\nBest by ValNLL:")
        print(f"  n={best_by_nll['n']}, h1={best_by_nll['h1']:.2f}, h_eff={best_by_nll['h_n']:.4f}")
        print(f"  ValNLL={best_by_nll['val_nll']:.6f}")
        
        # Compute spurious score for best-by-NLL
        S_nll, comp_nll = compute_global_spurious_score(
            mixture, best_by_nll['h_n'], best_by_nll['n']
        )
        print(f"  Predicted spurious score: {S_nll:.4f} ({interpret_spurious_score(S_nll)})")
        
        results['best_mse'] = {
            'n': best_by_mse['n'],
            'h1': best_by_mse['h1'],
            'h_eff': best_by_mse['h_n'],
            'mse': best_by_mse['grid_mse'],
            'S_global': S_mse,
            'interpretation': interpret_spurious_score(S_mse),
            'components': comp_mse
        }
        
        results['best_nll'] = {
            'n': best_by_nll['n'],
            'h1': best_by_nll['h1'],
            'h_eff': best_by_nll['h_n'],
            'val_nll': best_by_nll['val_nll'],
            'S_global': S_nll,
            'interpretation': interpret_spurious_score(S_nll),
            'components': comp_nll
        }
    
    return results


def create_visualization(all_results: List[Dict], output_path: str):
    """
    Create visualization of spurious peak scores across configurations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Spurious Peak Score Predictions Across Mixtures', fontsize=14, fontweight='bold')
    
    for mixture_idx, ax in enumerate(axes, start=1):
        mixture_results = next(
            (r for r in all_results if r['mixture_idx'] == mixture_idx),
            None
        )
        
        if not mixture_results:
            continue
        
        # Group by n value
        n_values = sorted(set(c['n'] for c in mixture_results['configurations']))
        
        for n in n_values:
            configs_n = [c for c in mixture_results['configurations'] if c['n'] == n]
            h1_vals = [c['h1'] for c in configs_n]
            S_vals = [c['S_global'] for c in configs_n]
            
            ax.plot(h1_vals, S_vals, marker='o', label=f'n={n}', linewidth=2)
        
        # Add interpretation regions
        ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.3, label='No spurious')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Mild threshold')
        ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.3, label='Severe threshold')
        
        ax.set_xlabel('Base Bandwidth h1', fontsize=11)
        ax.set_ylabel('Spurious Peak Score S_global', fontsize=11)
        ax.set_title(f'{mixture_results["mixture_name"]}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, max(3.0, ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def generate_latex_table(all_results: List[Dict], output_path: str):
    """
    Generate LaTeX table for inclusion in the paper.
    """
    with open(output_path, 'w') as f:
        f.write("% Spurious Peak Predictions for Best Configurations\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("Mixture & Selection & $n$ & $h_1$ & $S_{\\mathrm{global}}$ & Interpretation \\\\\n")
        f.write("\\hline\n")
        
        for result in all_results:
            mixture_name = result['mixture_name'].split('(')[0].strip()
            
            if 'best_mse' in result:
                best = result['best_mse']
                f.write(f"{mixture_name} & MSE & {best['n']} & {best['h1']:.2f} & "
                       f"{best['S_global']:.3f} & {best['interpretation']} \\\\\n")
            
            if 'best_nll' in result:
                best = result['best_nll']
                f.write(f"{mixture_name} & ValNLL & {best['n']} & {best['h1']:.2f} & "
                       f"{best['S_global']:.3f} & {best['interpretation']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Predicted spurious peak scores for best KDE configurations. ")
        f.write("Scores $> 1.5$ indicate severe spurious peaks.}\n")
        f.write("\\label{tab:spurious_predictions}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {output_path}")


def main():
    """Main validation routine."""
    print("="*80)
    print("SPURIOUS PEAK THEORY VALIDATION")
    print("="*80)
    print("\nImplementing formulas from deep_mathematical_proof_PW_overlays.tex")
    print("Section: Spurious Peak Formation in Undersmoothed KDE")
    
    # Load mixture definitions
    mixtures = load_mixture_definitions()
    
    # Load experimental data if available
    pw_logs = load_pw_logs()
    if pw_logs:
        print(f"\nLoaded experimental logs for {len(pw_logs)} mixture(s)")
    else:
        print("\nNo experimental logs found (run with --pw-write-logs to generate)")
    
    # Analyze each mixture
    all_results = []
    for mixture_idx, mixture in mixtures.items():
        logs = pw_logs.get(mixture_idx)
        result = analyze_mixture(mixture_idx, mixture, logs)
        all_results.append(result)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_json = os.path.join(output_dir, "spurious_peak_predictions.json")
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_json}")
    
    # Create visualization
    output_fig = "figures/spurious_peak_predictions.png"
    os.makedirs("figures", exist_ok=True)
    create_visualization(all_results, output_fig)
    
    # Generate LaTeX table
    output_tex = os.path.join(output_dir, "spurious_peak_table.tex")
    generate_latex_table(all_results, output_tex)
    
    print(f"{'='*80}")
    print("\nSUMMARY:")
    print("  The spurious peak formula predicts when KDE will show false modes.")
    print("  Compare these predictions with your overlay figures to validate the theory.")
    print("  Configurations with S_global > 1.5 should show clear spurious peaks.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
