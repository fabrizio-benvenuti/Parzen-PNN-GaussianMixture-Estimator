"""Generate assets (benchmarks + figures) referenced by experiment_report.tex.

This script is intentionally small and self-contained so the report can include
measured runtimes and a boundary-penalty comparison figure.

Outputs:
- figures/boundary_penalty_compare.jpeg
- report_assets_runtime.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

# Import from estimator.py (safe: main() is guarded by if __name__ == '__main__')
from estimator import Plotter, ParzenNeuralNetwork, compute_kde, sample_boundary_points_outside_convex_hull


@dataclass
class RuntimeRow:
    method: str
    n: int | None
    T: int
    seconds: float


def _time_once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return float(t1 - t0)


def benchmark_inference(*, T: int = 10_000, n_kde: int = 2_000, device: str = "cpu") -> list[RuntimeRow]:
    rng = np.random.default_rng(0)

    # Match the report's domain D roughly.
    eval_xy = rng.uniform(-5.0, 5.0, size=(T, 2)).astype(float)
    train_xy = rng.uniform(-5.0, 5.0, size=(n_kde, 2)).astype(float)

    # KDE benchmark (Gaussian KDE)
    def _kde():
        _ = compute_kde(eval_xy, train_xy, h1=7.0)

    # PNN benchmark (forward pass only). We don't need a trained network to measure inference cost.
    pnn = ParzenNeuralNetwork([30, 20], output_activation="relu", density_parameterization="log_density")
    pnn.eval()
    if device == "cuda" and torch.cuda.is_available():
        pnn.to("cuda")
        xt = torch.tensor(eval_xy, dtype=torch.float32, device="cuda")
        # Warm up
        with torch.no_grad():
            _ = pnn.forward(xt)
        torch.cuda.synchronize()

        def _pnn():
            with torch.no_grad():
                _ = pnn.forward(xt)
            torch.cuda.synchronize()

    else:
        xt = torch.tensor(eval_xy, dtype=torch.float32)

        def _pnn():
            with torch.no_grad():
                _ = pnn.forward(xt)

    # Run a couple of repetitions and keep the best (reduces noise).
    kde_t = min(_time_once(_kde) for _ in range(3))
    pnn_t = min(_time_once(_pnn) for _ in range(3))

    return [
        RuntimeRow(method="PW (Gaussian KDE)", n=n_kde, T=T, seconds=kde_t),
        RuntimeRow(method="PNN forward (fixed W)", n=None, T=T, seconds=pnn_t),
    ]


def boundary_penalty_figure(*, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)

    # Fixed domain D and a moderate grid.
    plotter = Plotter(-5, 5, -5, 5, 80)

    # Synthetic sample set (no ground-truth pdf needed for the comparison).
    n = 400
    samples_xy = rng.normal(loc=0.0, scale=1.5, size=(n, 2)).astype(float)

    # Same architecture/hyperparams, only lambda changes.
    common_kwargs = dict(
        bandwidth=7.0,
        learning_rate=5e-3,
        epochs=400,
        verbose=False,
        loss_mode="mse",
        num_uniform_points=300,
    )

    # Boundary points outside the convex hull of all sampled points.
    boundary_pts = sample_boundary_points_outside_convex_hull(samples_xy, alpha=0.1, k=max(50, int(0.3 * n)))

    def _train(lambda_boundary: float) -> np.ndarray:
        pnn = ParzenNeuralNetwork([30, 20], output_activation="relu", density_parameterization="log_density")
        pnn.train_network(
            samples_xy,
            plotter,
            boundary_points=boundary_pts,
            lambda_boundary=float(lambda_boundary),
            **common_kwargs,
        )
        return pnn.estimate_pdf(plotter)

    pdf0 = _train(lambda_boundary=0.0)
    pdf1 = _train(lambda_boundary=0.01)

    # Plot both surfaces side-by-side.
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(plotter.X, plotter.Y, pdf0, alpha=0.9, cmap="plasma")
    ax1.set_title(r"PNN (\lambda=0)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(plotter.X, plotter.Y, pdf1, alpha=0.9, cmap="plasma")
    ax2.set_title(r"PNN (\lambda=0.01)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    out_path = "figures/boundary_penalty_compare.jpeg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    runtimes = benchmark_inference(T=10_000, n_kde=2_000)
    out_json = {
        "runtimes": [row.__dict__ for row in runtimes],
    }
    with open("report_assets_runtime.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    fig_path = boundary_penalty_figure(seed=0)

    print("Wrote report_assets_runtime.json")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
