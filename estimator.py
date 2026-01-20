import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import sys
import time
import argparse
import csv

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)


def benchmark_loocv_target_generation(
    *,
    h1: float = 7.0,
    n_values: list[int] | None = None,
    repeats: int = 3,
    seed: int = 0,
) -> tuple[str, str]:
    """Empirically benchmark leave-one-out KDE target generation cost vs n.

    This isolates the O(n^2) step of constructing Parzen targets (pairwise kernel evaluation).

    Outputs:
      - results/training_complexity_loocv.csv
      - figures/training_complexity_loocv.jpeg
    """
    if n_values is None:
        n_values = [200, 400, 800, 1200, 1600, 2000]
    repeats = int(repeats)
    if repeats < 1:
        repeats = 1

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float]] = []

    for n in n_values:
        n = int(n)
        # Generate a synthetic point cloud roughly centered in the domain.
        pts = rng.normal(loc=0.0, scale=1.0, size=(n, 2)).astype(float)
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = compute_leave_one_out_kde_targets(pts, float(h1))
            t1 = time.perf_counter()
            times.append(float(t1 - t0))
        mean_t = float(np.mean(times))
        std_t = float(np.std(times))
        rows.append({"n": float(n), "seconds_mean": mean_t, "seconds_std": std_t})
        print(f"LOO target generation: n={n:5d}  mean={mean_t:.4f}s  std={std_t:.4f}s")

    # Save CSV
    csv_path = os.path.join("results", "training_complexity_loocv.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n,seconds_mean,seconds_std\n")
        for r in rows:
            f.write(f"{int(r['n'])},{r['seconds_mean']:.8f},{r['seconds_std']:.8f}\n")

    # Plot time vs n and fitted quadratic reference
    n_arr = np.array([r["n"] for r in rows], dtype=float)
    t_arr = np.array([r["seconds_mean"] for r in rows], dtype=float)
    # Fit c in t ≈ c n^2 by least squares (through origin)
    denom = float(np.sum((n_arr**2) ** 2))
    c_hat = float(np.sum(t_arr * (n_arr**2)) / denom) if denom > 0 else 0.0
    t_ref = c_hat * (n_arr**2)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(n_arr, t_arr, marker="o", label="Measured (mean)")
    ax.plot(n_arr, t_ref, linestyle="--", label=r"Fit: $c\,n^2$")
    ax.set_title("Empirical complexity: leave-one-out KDE target generation")
    ax.set_xlabel("n (number of samples)")
    ax.set_ylabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_path = os.path.join("figures", "training_complexity_loocv.jpeg")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved complexity CSV: {csv_path}")
    print(f"Saved complexity figure: {fig_path}")

    return csv_path, fig_path


def _dx_dy_from_plotter(plotter) -> tuple[float, float]:
    dx = float(plotter.x[1] - plotter.x[0]) if len(plotter.x) > 1 else 1.0
    dy = float(plotter.y[1] - plotter.y[0]) if len(plotter.y) > 1 else 1.0
    return dx, dy


def _effective_bandwidth(base_bandwidth: float, sample_count: int) -> float:
    """Scale base bandwidth h_1 into h_n per Parzen rule h_1/sqrt(n-1)."""
    if not np.isfinite(base_bandwidth) or base_bandwidth <= 0:
        raise ValueError("Bandwidth h_1 must be finite and > 0")
    n = int(sample_count)
    if n <= 1:
        return float(base_bandwidth)
    return float(base_bandwidth) / float(np.sqrt(n - 1))


def split_train_validation(
    points_xy: np.ndarray,
    *,
    val_fraction: float = 0.2,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split 2D points into train/validation subsets.

    This is used to evaluate estimators without access to ground truth, via
    validation log-likelihood on held-out points.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 points to split")
    frac = float(val_fraction)
    if not np.isfinite(frac) or not (0.0 < frac < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(np.floor(frac * n)))
    n_train = n - n_val
    if n_train < 1:
        n_train = 1
        n_val = n - 1

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return pts[train_idx], pts[val_idx]


def average_log_likelihood_kde(
    points_eval_xy: np.ndarray,
    points_train_xy: np.ndarray,
    h1: float,
    *,
    eps: float = 1e-300,
) -> float:
    """Average log-likelihood under a Gaussian KDE fitted on points_train_xy."""
    dens = compute_kde(points_eval_xy, points_train_xy, h1)
    dens = np.maximum(dens, float(eps))
    return float(np.mean(np.log(dens)))


def average_log_likelihood_pnn_on_domain(
    pnn: "ParzenNeuralNetwork",
    points_eval_xy: np.ndarray,
    plotter,
    *,
    eps: float = 1e-12,
) -> float:
    """Average log-likelihood using the PNN normalized on the Plotter domain D.

    Note: the PNN is normalized by a Riemann-sum approximation on the finite
    rectangle D, therefore this likelihood is an approximation and depends on D.
    """
    X = np.asarray(points_eval_xy, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points_eval_xy must have shape (m, 2)")

    # Compute normalizer on the grid (same as estimate_pdf, but without returning the full grid).
    grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    dx, dy = _dx_dy_from_plotter(plotter)
    with torch.no_grad():
        raw_grid = pnn.forward(grid_tensor).cpu().numpy()
        if pnn.density_parameterization == "log_density":
            unnorm_grid = np.exp(raw_grid)
        else:
            unnorm_grid = raw_grid
    Z = float(np.sum(unnorm_grid) * (dx * dy))
    Z = max(Z, float(eps))

    # Evaluate unnormalized density at held-out points.
    eval_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        raw_eval = pnn.forward(eval_tensor).cpu().numpy()
        if pnn.density_parameterization == "log_density":
            unnorm_eval = np.exp(raw_eval)
        else:
            unnorm_eval = raw_eval
    pdf_eval = np.maximum(unnorm_eval / Z, float(eps))
    return float(np.mean(np.log(pdf_eval)))


def mean_unnormalized_density_on_points(
    pnn: "ParzenNeuralNetwork",
    points_xy: np.ndarray,
    *,
    clamp_log_max: float = 10.0,
) -> float:
    """Mean of the PNN unnormalized density on a set of points.

    Useful as a simple diagnostic for heavy tails (e.g., on boundary/shell points).
    """
    X = np.asarray(points_xy, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points_xy must have shape (m, 2)")
    xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        raw = pnn.forward(xt)
        if pnn.density_parameterization == "log_density":
            raw = torch.clamp(raw, max=float(clamp_log_max))
            dens = torch.exp(raw)
        else:
            dens = raw
    return float(torch.mean(dens).cpu().item())


def _nll_from_avg_loglik(avg_loglik: float) -> float:
    """Convert average log-likelihood to (positive) average negative log-likelihood."""
    if not np.isfinite(avg_loglik):
        return float("inf")
    return float(-avg_loglik)


def _mean_density_on_boundary_ring(pdf_grid: np.ndarray, plotter, *, ring_width: float = 1.0) -> float:
    """Mean density near the boundary of the plot domain (inside D).

    Uses the already-normalized pdf evaluated on the plotter grid.
    """
    xmin = float(np.min(plotter.x))
    xmax = float(np.max(plotter.x))
    ymin = float(np.min(plotter.y))
    ymax = float(np.max(plotter.y))
    w = float(ring_width)
    X = plotter.X
    Y = plotter.Y
    mask = (X <= xmin + w) | (X >= xmax - w) | (Y <= ymin + w) | (Y >= ymax - w)
    vals = np.asarray(pdf_grid)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def run_uniform_supervision_demo_only() -> None:
    """Generate a small ablation artifact for the report.

    Trains the same PNN twice (no-uniform vs with-uniform interior supervision), then
    saves a side-by-side heatmap and a JSON summary under results/.

    This is intentionally minimal and does not depend on the full sweep pipeline.
    """

    # Keep it deterministic.
    seed = 9103
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use the same plot domain as the report.
    plotter = Plotter(-5, 5, -5, 5, 100)

    # Construct Mixture 3 exactly as in main().
    weights3 = [0.2, 0.2, 0.2, 0.2, 0.2]
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    g2 = MultivariateGaussian([-2, -1], [[1.14822883, 0.19240818], [0.19240818, 1.23432651]])
    g3 = MultivariateGaussian([-1, 3], [[0.30198015, 0.13745508], [0.13745508, 1.69483031]])
    g4 = MultivariateGaussian([1.5, -0.5], [[0.85553671, -0.19601649], [-0.19601649, 0.7507167]])
    g5 = MultivariateGaussian([-3, 2], [[0.42437194, -0.17066673], [-0.17066673, 2.16117758]])
    mixture_idx = 2  # Mixture 3 (0-based)
    mixture = GaussianMixture([g1, g2, g3, g4, g5], weights3)

    # Match the main experiment defaults.
    h1 = 12.0
    learning_rate = 5e-3
    epochs = 2000
    use_log_density = True

    # Representative architecture (also commonly best in the sweep).
    cfg = {"hidden_layers": [30, 20], "out": "relu", "A": "auto"}
    label = "MLP_30-20_sigmoid_outReLU"

    # Samples + held-out split consistent with the main code.
    samples_xy = mixture.sample_points_weighted(100, with_pdf=False)
    train_xy, val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=mixture_idx + 1)

    def _train(num_uniform_points: int) -> ParzenNeuralNetwork:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ParzenNeuralNetwork(
            hidden_layers=cfg["hidden_layers"],
            output_activation=cfg["out"],
            output_scale=cfg.get("A", "auto"),
            density_parameterization="log_density" if use_log_density else "density",
        )
        model.train_network(
            train_xy,
            plotter,
            bandwidth=float(h1),
            mixture=mixture,
            log_file=None,
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            boundary_points=None,
            lambda_boundary=0.0,
            verbose=False,
            loss_mode="mse",
            weight_decay=0.0,
            num_uniform_points=int(num_uniform_points),
        )
        return model

    pnn_no_uniform = _train(0)
    pnn_with_uniform = _train(len(train_xy))

    pdf_no = pnn_no_uniform.estimate_pdf(plotter)
    pdf_u = pnn_with_uniform.estimate_pdf(plotter)
    ring_no = _mean_density_on_boundary_ring(pdf_no, plotter, ring_width=1.0)
    ring_u = _mean_density_on_boundary_ring(pdf_u, plotter, ring_width=1.0)
    val_nll_no = _nll_from_avg_loglik(average_log_likelihood_pnn_on_domain(pnn_no_uniform, val_xy, plotter))
    val_nll_u = _nll_from_avg_loglik(average_log_likelihood_pnn_on_domain(pnn_with_uniform, val_xy, plotter))

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Uniform supervision ablation — mixture {mixture_idx+1}\n{label}, h_1={h1:.3f}")

    ax_a = fig.add_subplot(1, 2, 1)
    im0 = ax_a.imshow(
        pdf_no,
        origin="lower",
        extent=(float(np.min(plotter.x)), float(np.max(plotter.x)), float(np.min(plotter.y)), float(np.max(plotter.y))),
        aspect="auto",
        cmap="viridis",
    )
    ax_a.scatter(train_xy[:, 0], train_xy[:, 1], s=6, c="white", alpha=0.7)
    ax_a.set_title(f"No uniform points\nValNLL={val_nll_no:.3g}, ringMean={ring_no:.3g}")
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("y")
    fig.colorbar(im0, ax=ax_a, fraction=0.046, pad=0.04)

    ax_b = fig.add_subplot(1, 2, 2)
    im1 = ax_b.imshow(
        pdf_u,
        origin="lower",
        extent=(float(np.min(plotter.x)), float(np.max(plotter.x)), float(np.min(plotter.y)), float(np.max(plotter.y))),
        aspect="auto",
        cmap="viridis",
    )
    ax_b.scatter(train_xy[:, 0], train_xy[:, 1], s=6, c="white", alpha=0.7)
    ax_b.set_title(f"With uniform points\nValNLL={val_nll_u:.3g}, ringMean={ring_u:.3g}")
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("y")
    fig.colorbar(im1, ax=ax_b, fraction=0.046, pad=0.04)

    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/uniform_supervision_comparison_mixture3.jpeg"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved uniform supervision comparison figure: {fig_path}")

    payload = {
        "mixture": 3,
        "label": str(label),
        "h1": float(h1),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "val_fraction": 0.2,
        "ring_width": 1.0,
        "no_uniform": {"val_avg_nll": float(val_nll_no), "boundary_ring_mean": float(ring_no)},
        "with_uniform": {"val_avg_nll": float(val_nll_u), "boundary_ring_mean": float(ring_u)},
        "figure": str(fig_path),
    }
    os.makedirs("results", exist_ok=True)
    out_path = "results/uniform_supervision_demo_mixture3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved uniform supervision demo JSON: {out_path}")


def _bottom_20_bounds(values: np.ndarray) -> tuple[float, float] | None:
    """Return (min, min + 20% range) bounds for finite values, with padding fallback."""
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    delta = float(ymax - ymin)
    if delta <= 0.0:
        pad = max(1e-12, abs(ymin) * 0.05)
        return ymin - pad, ymin + pad
    return ymin, ymin + 0.20 * delta


def compute_leave_one_out_kde_targets(points_xy: np.ndarray, h1: float) -> np.ndarray:
    """Compute leave-one-out Parzen targets y_i using isotropic Gaussian kernels.

    Given x_i in R^2:

        y_i = (1/(n-1)) * sum_{j != i} N(x_i | x_j, h^2 I)

    where N(·|μ, h^2 I) includes the 2D Gaussian normalization 1/(2π h^2).
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples for leave-one-out KDE targets")
    h = _effective_bandwidth(float(h1), n)

    diffs = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    k = np.exp(-dist2 / (2.0 * h * h))
    np.fill_diagonal(k, 0.0)

    norm = 1.0 / (2.0 * np.pi * h * h)
    y = (norm * np.sum(k, axis=1)) / float(n - 1)
    return y


def compute_kde(points_eval_xy: np.ndarray, points_train_xy: np.ndarray, h1: float) -> np.ndarray:
    """Compute Parzen KDE at evaluation points using isotropic Gaussian kernels.

    For x in R^2:

        KDE(x) = (1/n) * sum_j N(x | x_j, h^2 I)

    where N includes the 2D Gaussian normalization 1/(2π h^2).
    """
    X = np.asarray(points_eval_xy, dtype=float)
    T = np.asarray(points_train_xy, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points_eval_xy must have shape (m, 2)")
    if T.ndim != 2 or T.shape[1] != 2:
        raise ValueError("points_train_xy must have shape (n, 2)")
    n = int(T.shape[0])
    if n < 1:
        raise ValueError("Need at least 1 training sample")
    h = _effective_bandwidth(float(h1), n)

    diffs = X[:, None, :] - T[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    k = np.exp(-dist2 / (2.0 * h * h))
    norm = 1.0 / (2.0 * np.pi * h * h)
    y = (norm * np.sum(k, axis=1)) / float(n)
    return y


def sample_boundary_points_outside_plot(plotter, alpha: float, k: int) -> np.ndarray:
    r"""Sample k points from a padded rectangle around the plot domain, rejecting inside points.

    The plot rectangle is the support X. We sample from the shell \bar B_δ = (X padded by δ) \ X,
    with δ = alpha * diameter(X).
    """
    min_x = float(np.min(plotter.x))
    max_x = float(np.max(plotter.x))
    min_y = float(np.min(plotter.y))
    max_y = float(np.max(plotter.y))
    diameter = float(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2))
    delta = float(alpha) * diameter

    pad_min_x = min_x - delta
    pad_max_x = max_x + delta
    pad_min_y = min_y - delta
    pad_max_y = max_y + delta

    pts: list[list[float]] = []
    attempts = 0
    max_attempts = max(10000, 10 * int(k))
    while len(pts) < int(k) and attempts < max_attempts:
        x = float(np.random.uniform(pad_min_x, pad_max_x))
        y = float(np.random.uniform(pad_min_y, pad_max_y))
        inside = (min_x <= x <= max_x) and (min_y <= y <= max_y)
        if not inside:
            pts.append([x, y])
        attempts += 1
    return np.asarray(pts, dtype=float)

class ParzenNeuralNetwork(nn.Module):
    r"""Parzen Neural Network (PNN) as required by the project prompt.

    - Input: 2D point (x, y)
    - Hidden layers: 1 or 2 layers, sigmoid activation
    - Output: either ReLU, or scaled sigmoid A·sigmoid(z)

    Training targets are NOT oracle mixture pdf values; they are leave-one-out Parzen estimates
    computed from samples only.
    """

    def __init__(
        self,
        hidden_layers: list[int] | tuple[int, ...],
        *,
        output_activation: str = "relu",
        output_scale: float | str | None = 1.0,
        density_parameterization: str = "density",
        eps: float = 1e-12,
    ):
        super().__init__()
        hidden_layers = list(hidden_layers)
        if len(hidden_layers) not in (1, 2):
            raise ValueError("PNN must have 1 or 2 hidden layers")
        if not all(int(w) > 0 for w in hidden_layers):
            raise ValueError("hidden layer widths must be positive")

        self.eps = float(eps)
        self.density_parameterization = str(density_parameterization).lower()
        if self.density_parameterization not in ("density", "log_density"):
            raise ValueError("density_parameterization must be 'density' or 'log_density'")

        self.output_activation = str(output_activation).lower()
        self.output_scale: float | None
        if output_scale is None:
            self.output_scale = None
        elif isinstance(output_scale, str):
            if output_scale.strip().lower() == "auto":
                self.output_scale = None
            else:
                self.output_scale = float(output_scale)
        else:
            self.output_scale = float(output_scale)
        if self.output_activation not in ("relu", "sigmoid"):
            raise ValueError("output_activation must be 'relu' or 'sigmoid'")

        layers: list[nn.Module] = []
        prev = 2
        for width in hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.Sigmoid())
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Avoid "dead" ReLU at init: if the final pre-activation starts negative everywhere,
        # ReLU would output ~0 and gradients would be zero. A small positive bias prevents this.
        if self.output_activation == "relu":
            last = self.net[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                nn.init.constant_(last.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape (N, 2)")
        z = self.net(x).squeeze(-1)
        if self.density_parameterization == "log_density":
            return z

        if self.output_activation == "relu":
            out = torch.relu(z)
        else:
            scale = 1.0 if self.output_scale is None else float(self.output_scale)
            out = scale * torch.sigmoid(z)
        return out

    def estimate_pdf(self, plotter) -> np.ndarray:
        grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        dx, dy = _dx_dy_from_plotter(plotter)
        with torch.no_grad():
            raw = self.forward(grid_tensor).cpu().numpy().reshape(plotter.X.shape)
            if self.density_parameterization == "log_density":
                unnorm = np.exp(raw)
            else:
                unnorm = raw
        Z = float(np.sum(unnorm) * (dx * dy))
        return unnorm / (Z + 1e-12)

    def train_network(
        self,
        points,
        plotter,
        *,
        bandwidth: float,
        mixture=None,
        log_file=None,
        learning_rate: float = 1e-3,
        epochs: int = 1500,
        boundary_points=None,
        lambda_boundary: float = 0.0,
        verbose: bool = True,
        loss_mode: str = "relative",
        weight_decay: float = 0.0,
        num_uniform_points: int = 0,
    ):
        """Train by regression on leave-one-out Parzen targets.

        loss_mode:
          - "mse": (f(x_i) - y_i)^2
          - "relative": ((f - y)/(y+eps))^2
          - "log": (log(f+eps) - log(y+eps))^2
        """
        points_xy = np.asarray(points)[:, :2]
        n = int(points_xy.shape[0])
        if n < 2:
            raise ValueError("Need at least 2 samples to train")

        h1 = float(bandwidth)
        h_n = _effective_bandwidth(h1, n)
        targets = compute_leave_one_out_kde_targets(points_xy, h1)
        targets_mean = float(np.mean(targets))
        if not np.isfinite(targets_mean) or targets_mean <= 0:
            raise ValueError("KDE targets have non-finite or non-positive mean; cannot normalize")
        targets = targets / targets_mean

        # Optional interior supervision: add uniform points in the plot domain and target them
        # with full KDE (still sample-only, still Parzen-based).
        m_uniform = int(num_uniform_points)
        uniform_xy = None
        uniform_targets = None
        if m_uniform > 0:
            min_x = float(np.min(plotter.x))
            max_x = float(np.max(plotter.x))
            min_y = float(np.min(plotter.y))
            max_y = float(np.max(plotter.y))
            ux = np.random.uniform(min_x, max_x, size=m_uniform)
            uy = np.random.uniform(min_y, max_y, size=m_uniform)
            uniform_xy = np.column_stack([ux, uy]).astype(float)
            uniform_targets = compute_kde(uniform_xy, points_xy, h1) / targets_mean

        # If using scaled-sigmoid output and scale wasn't provided, pick it from targets.
        if self.density_parameterization == "density":
            if self.output_activation == "sigmoid" and self.output_scale is None:
                tmax = float(np.max(targets))
                if np.isfinite(tmax) and tmax > 0:
                    self.output_scale = 1.5 * tmax
                else:
                    self.output_scale = 1.0

        if self.density_parameterization == "log_density":
            targets = np.log(targets + self.eps)
            if uniform_targets is not None:
                uniform_targets = np.log(uniform_targets + self.eps)

        if uniform_xy is not None and uniform_targets is not None:
            train_xy = np.vstack([points_xy, uniform_xy])
            train_targets = np.concatenate([targets, uniform_targets])
        else:
            train_xy = points_xy
            train_targets = targets

        points_tensor = torch.tensor(train_xy, dtype=torch.float32)
        targets_tensor = torch.tensor(train_targets, dtype=torch.float32)

        if verbose:
            if self.density_parameterization == "log_density":
                # Here `targets` are already log(KDE/mean).
                approx_min = float(np.exp(np.min(targets)))
                approx_max = float(np.exp(np.max(targets)))
                print(
                    f"LOG-KDE target stats (h1={h1:.6g}, h_n={h_n:.6g}): "
                    f"min={targets.min():.3e}, median={np.median(targets):.3e}, mean={targets.mean():.3e}, max={targets.max():.3e} | "
                    f"approx pdf range: [{approx_min:.3e}, {approx_max:.3e}]"
                )
            else:
                frac_tiny = float(np.mean(targets < 1e-12))
                print(
                    f"KDE target stats (h1={h1:.6g}, h_n={h_n:.6g}): "
                    f"min={targets.min():.3e}, median={np.median(targets):.3e}, mean={targets.mean():.3e}, max={targets.max():.3e}, "
                    f"frac<1e-12={100.0*frac_tiny:.2f}%"
                )

        loss_mode = str(loss_mode).lower()
        if loss_mode not in ("mse", "relative", "log"):
            raise ValueError("loss_mode must be one of: 'mse', 'relative', 'log'")

        if self.density_parameterization == "log_density" and loss_mode != "mse":
            raise ValueError("In log_density mode, set loss_mode='mse' (MSE on log-targets)")

        optimizer = optim.Adam(
            self.parameters(),
            lr=float(learning_rate),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(weight_decay),
        )

        boundary_tensor = None
        if boundary_points is not None and len(boundary_points) > 0:
            boundary_tensor = torch.tensor(np.asarray(boundary_points), dtype=torch.float32)

        eval_mse_history: list[float] = []
        train_loss_history: list[float] = []

        # Track best weights during training.
        best_state_dict = None
        best_metric = float("inf")
        for epoch in range(int(epochs)):
            optimizer.zero_grad()

            if self.density_parameterization == "log_density":
                pred_log = self.forward(points_tensor)
                loss_kde = torch.mean((pred_log - targets_tensor) ** 2)
            else:
                pred = self.forward(points_tensor)
                if loss_mode == "mse":
                    loss_kde = torch.mean((pred - targets_tensor) ** 2)
                elif loss_mode == "relative":
                    loss_kde = torch.mean(((pred - targets_tensor) / (targets_tensor + self.eps)) ** 2)
                else:
                    log_pred = torch.log(pred + self.eps)
                    log_tgt = torch.log(targets_tensor + self.eps)
                    loss_kde = torch.mean((log_pred - log_tgt) ** 2)

            penalty = torch.tensor(0.0)
            if boundary_tensor is not None and float(lambda_boundary) != 0.0:
                if self.density_parameterization == "log_density":
                    logp = self.forward(boundary_tensor)
                    # Clamp to avoid rare numeric blowups from exp(logp) during long runs.
                    logp = torch.clamp(logp, max=10.0)
                    penalty = torch.mean(torch.exp(logp) ** 2)
                else:
                    penalty = torch.mean(self.forward(boundary_tensor) ** 2)

            loss = loss_kde + float(lambda_boundary) * penalty

            train_loss_history.append(float(loss_kde.detach().item()))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch}: loss is NaN/Inf. Try reducing learning rate or adjusting bandwidth.")
                return loss.detach(), eval_mse_history, train_loss_history

            loss.backward()
            optimizer.step()

            # Optional: track evaluation MSE on grid against ground truth.
            if mixture is not None:
                est_pdf = self.estimate_pdf(plotter)
                true_pdf = mixture.get_mesh(plotter.pos)
                mse = float(np.mean((est_pdf - true_pdf) ** 2))
                eval_mse_history.append(mse)

                if mse < best_metric:
                    best_metric = mse
                    best_state_dict = copy.deepcopy(self.state_dict())
            else:
                # No mixture available: fall back to tracking best training objective.
                metric = float(loss_kde.detach().item())
                if metric < best_metric:
                    best_metric = metric
                    best_state_dict = copy.deepcopy(self.state_dict())

            if verbose and epoch % 100 == 0:
                msg = (
                    f"Epoch {epoch}: TrainLoss={loss_kde.detach().item():.6e}"
                    + (f", Pen={penalty.detach().item():.6e}" if boundary_tensor is not None and float(lambda_boundary) != 0.0 else "")
                    + (f", EvalMSE={eval_mse_history[-1]:.6e}" if eval_mse_history else "")
                    + f" (h1={h1:.6g}, h_n={h_n:.6g}, lr={float(learning_rate):.6g}, loss={loss_mode})\n"
                )
                if log_file:
                    log_file.write(msg)
                print(msg, end="")

        # Restore best parameters before returning.
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return loss.detach(), eval_mse_history, train_loss_history

class ParzenWindowEstimator:
    def __init__(self, points, window_size=0.1):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Points must be a numpy array with shape (n, 2) or (n, 3).")
        self.points = points[:, :2]
        self.h1 = float(window_size)
        if not np.isfinite(self.h1) or self.h1 <= 0:
            raise ValueError("window_size (h_1) must be finite and > 0")
        self.sample_count = int(self.points.shape[0])
        self.window_size = _effective_bandwidth(self.h1, self.sample_count)
        # For a 2D isotropic Gaussian kernel, normalization is 1/(2π h^2)
        self.kernel_norm = 1.0 / (2.0 * np.pi * self.window_size * self.window_size)

    def estimate_pdf(self, plotter):
        self.pdf = np.zeros(plotter.pos.shape[:2])
        num_points = self.points.shape[0]

        for point in self.points:
            diff = plotter.pos - point
            norm = np.sum((diff / self.window_size) ** 2, axis=2)
            self.pdf += np.exp(-0.5 * norm)

        self.pdf *= self.kernel_norm / num_points
        return self.pdf

    def calculate_error_metrics(self, original_pdf):
        if self.pdf.shape != original_pdf.shape:
            raise ValueError("The shapes of estimated_pdf and original_pdf must match.")
        mse = np.mean((self.pdf - original_pdf) ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(self.pdf - original_pdf))
        mean_abs_error = np.mean(np.abs(self.pdf - original_pdf))
        return {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Max Absolute Error': max_error,
            'Mean Absolute Error': mean_abs_error
        }

class MultivariateGaussian:
    def __init__(self, mu, covariance):
        if len(mu) != 2 or np.array(covariance).shape != (2, 2):
            raise ValueError("Mean must be a list of 2 values and covariance must be a 2x2 matrix.")
        if not np.all(np.linalg.eigvals(covariance) >= 0):
            raise ValueError("Covariance matrix must be positive semidefinite.")
        self._mu = np.array(mu)
        self._covariance = np.array(covariance)
        self._rv = multivariate_normal(self._mu, self._covariance)

    def get_mu(self):
        return self._mu

    def get_covariance(self):
        return self._covariance

    def get_distribution(self):
        return self._rv

class GaussianMixture:
    def __init__(self, gaussians, weights):
        if len(gaussians) != len(weights):
            raise ValueError("The number of Gaussian distributions must match the number of weights.")
        if not all(isinstance(g, MultivariateGaussian) for g in gaussians):
            raise TypeError("All elements in 'gaussians' must be instances of MultivariateGaussian.")
        dimensions = [g.get_mu().shape[0] for g in gaussians]
        if len(set(dimensions)) != 1:
            raise ValueError("All Gaussian distributions must have the same dimensionality.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("The sum of all weights must be 1.")
        self._gaussians = gaussians
        self._weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def get_gaussian_and_weight(self, index):
        if index < 0 or index >= len(self._gaussians):
            raise IndexError("Index out of range.")
        return self._gaussians[index], self._weights[index]

    def get_mesh(self, pos):
        total_pdf = np.zeros(pos.shape[:2])
        for gaussian, weight in zip(self._gaussians, self._weights):
            total_pdf += weight * gaussian.get_distribution().pdf(pos)
        return total_pdf

    def sample_points(self, num_points: int, with_pdf: bool = True):
        points = []
        for _ in range(num_points):
            idx = np.random.choice(len(self._gaussians), p=self._weights)
            sample = self._gaussians[idx].get_distribution().rvs()
            if with_pdf:
                pdf_value = sum(weight * gaussian.get_distribution().pdf(sample)
                                for gaussian, weight in zip(self._gaussians, self._weights))
                points.append([sample[0], sample[1], pdf_value])
            else:
                points.append([sample[0], sample[1]])
        return np.array(points, dtype=float)

    def sample_points_weighted(self, base_per_gaussian: int, with_pdf: bool = True) -> np.ndarray:
        """Sample approximately `base_per_gaussian` points per component.

        Total samples = base_per_gaussian * K, distributed proportionally to mixture weights.
        This matches the project text: "~100 points per Gaussian, increasing with its weight".
        """
        base = int(base_per_gaussian)
        if base <= 0:
            raise ValueError("base_per_gaussian must be > 0")
        K = len(self._gaussians)
        total = base * K

        w = np.asarray(self._weights, dtype=float)
        raw = w * float(total)
        counts = np.floor(raw).astype(int)
        remainder = int(total - int(np.sum(counts)))
        if remainder > 0:
            frac = raw - counts
            order = np.argsort(frac)[::-1]
            for idx in order[:remainder]:
                counts[int(idx)] += 1

        # Ensure every component contributes at least 1 sample if possible.
        for k in range(K):
            if counts[k] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[k] += 1

        points: list[list[float]] = []
        for gaussian, count in zip(self._gaussians, counts):
            if count <= 0:
                continue
            samples = gaussian.get_distribution().rvs(size=int(count))
            samples = np.atleast_2d(samples)
            for s in samples:
                if with_pdf:
                    pdf_value = float(
                        sum(weight * g.get_distribution().pdf(s) for g, weight in zip(self._gaussians, self._weights))
                    )
                    points.append([float(s[0]), float(s[1]), pdf_value])
                else:
                    points.append([float(s[0]), float(s[1])])
        pts = np.asarray(points, dtype=float)
        # Shuffle so batches aren't grouped by component.
        rng = np.random.default_rng()
        rng.shuffle(pts, axis=0)
        return pts

class Plotter:
    def __init__(self, min_x, max_x, min_y, max_y, num_points):
        self.x = np.linspace(min_x, max_x, num_points)
        self.y = np.linspace(min_y, max_y, num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((self.X, self.Y))
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def add_surface(self, mesh, color='plasma', alpha=0.7):
        # mesh: a 2D array of pdf values that matches self.X and self.Y
        self.ax.plot_surface(self.X, self.Y, mesh, alpha=alpha, cmap=color)

    def add_points(self, points, color='red'):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Points must be a numpy array with shape (n, 2) or (n, 3).")
        if points.shape[1] == 2:
            self.ax.scatter(points[:, 0], points[:, 1], np.zeros(points.shape[0]), color=color)
        else:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)

    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(self.fig)

def gen_covariance_with_relative_widths(x_width, y_width, correlation=0.0):
    sigma_x = x_width
    sigma_y = y_width
    rho = correlation
    return np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2]
    ])

def create_and_display_mixtures(avgs):
	while True:
		gaussians = [MultivariateGaussian(mu, gen_covariance_with_relative_widths(np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), correlation=np.random.uniform(-0.5, 0.5))) for mu in avgs]
		weights1 = [1.0]
		weights2 = [0.3,0.3,0.4]
		weights3 = [0.2,0.2,0.2,0.2,0.2]

		mixture1 = GaussianMixture([gaussians[0]], weights1)
		mixture2 = GaussianMixture(gaussians[:3], weights2)
		mixture3 = GaussianMixture(gaussians, weights3)

		plotter = Plotter(-5,5,-5,5,100)
		plotter2 = Plotter(-5,5,-5,5,100)
		plotter3 = Plotter(-5,5,-5,5,100)
		mesh1 = mixture1.get_mesh(plotter.pos)
		mesh2 = mixture2.get_mesh(plotter.pos)
		mesh3 = mixture3.get_mesh(plotter.pos)

		plotter.ax.plot_surface(plotter.X, plotter.Y, mesh1, alpha=0.7, cmap='plasma')
		plotter2.ax.plot_surface(plotter.X, plotter.Y, mesh2, alpha=0.7,  cmap='plasma')
		plotter3.ax.plot_surface(plotter.X, plotter.Y, mesh3, alpha=0.7,  cmap='plasma')
		plotter.show()

		user_input = input("Do you approve the mixtures? (yes/no): ").strip().lower()
		if user_input == 'yes':
			print("Covariance Matrices and Weights:")
			print(f"Mixture 1 Weights: {weights1}")
			print(f"Gaussian 1 Covariance Matrix:\n{gaussians[0].get_covariance()}\n")

			print(f"Mixture 2 Weights: {weights2}")
			for i, gaussian in enumerate(gaussians[:3], start=1):
				print(f"Gaussian {i} Covariance Matrix:\n{gaussian.get_covariance()}\n")

			print(f"Mixture 3 Weights: {weights3}")
			for i, gaussian in enumerate(gaussians, start=1):
				print(f"Gaussian {i} Covariance Matrix:\n{gaussian.get_covariance()}\n")
			break
		else:
			print("Regenerating mixtures...")

def powspace(start, stop, power, num, allow_floats: bool):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    power_array = np.power(np.linspace(start, stop, num=num), power)
    if allow_floats:
        return power_array
    else:
        return np.array([int(i) for i in power_array])


def compute_support_bounds_for_mixture(mixture: GaussianMixture, k_sigma: float = 3.0) -> tuple:
    """Axis-aligned rectangular support bounds X from mixture means and covariances.
    Uses k_sigma standard deviations around each mean, then takes the union across components.
    Returns ((min_x, max_x), (min_y, max_y)).
    """
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for gaussian, _w in zip(mixture._gaussians, mixture._weights):
        mu = gaussian.get_mu()
        cov = gaussian.get_covariance()
        sx = float(np.sqrt(cov[0, 0]))
        sy = float(np.sqrt(cov[1, 1]))
        min_x = min(min_x, mu[0] - k_sigma * sx)
        max_x = max(max_x, mu[0] + k_sigma * sx)
        min_y = min(min_y, mu[1] - k_sigma * sy)
        max_y = max(max_y, mu[1] + k_sigma * sy)
    return (min_x, max_x), (min_y, max_y)


def sample_boundary_points_outside_support(bounds: tuple, alpha: float, k: int) -> np.ndarray:
    """Sample k points uniformly in the padded rectangle B_δ outside support X.
    δ = α * diameter(X). We sample in the outer rectangle and reject points inside X.
    """
    (min_x, max_x), (min_y, max_y) = bounds
    diameter = float(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2))
    delta = alpha * diameter
    pad_min_x = min_x - delta
    pad_max_x = max_x + delta
    pad_min_y = min_y - delta
    pad_max_y = max_y + delta
    pts = []
    attempts = 0
    max_attempts = max(10000, 10 * k)
    while len(pts) < k and attempts < max_attempts:
        x = np.random.uniform(pad_min_x, pad_max_x)
        y = np.random.uniform(pad_min_y, pad_max_y)
        inside = (min_x <= x <= max_x) and (min_y <= y <= max_y)
        if not inside:
            pts.append([x, y])
        attempts += 1
    return np.array(pts, dtype=float)


def main():
    torch.autograd.set_detect_anomaly(True)

    # CLI flags (kept minimal and backward-compatible).
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pw-only", action="store_true", help="Run only Parzen Window (KDE) sweep")
    parser.add_argument("--pw-seed", type=int, default=0, help="Seed for PW-only sweep reproducibility")
    parser.add_argument(
        "--pw-write-logs",
        action="store_true",
        help="Write PW error logs (CSV + SUMMARY lines) for each (n, h1)",
    )
    args, _unknown = parser.parse_known_args()

    weights1 = [1.0]
    weights2 = [0.3, 0.3, 0.4]
    weights3 = [0.2, 0.2, 0.2, 0.2, 0.2]
    g1 = MultivariateGaussian([1, 2], [[1.62350208, -0.13337813], [-0.13337813, 0.63889251]])
    g2 = MultivariateGaussian([-2, -1], [[1.14822883, 0.19240818], [0.19240818, 1.23432651]])
    g3 = MultivariateGaussian([-1, 3], [[0.30198015, 0.13745508], [0.13745508, 1.69483031]])
    g4 = MultivariateGaussian([1.5, -0.5], [[0.85553671, -0.19601649], [-0.19601649, 0.7507167]])
    g5 = MultivariateGaussian([-3, 2], [[0.42437194, -0.17066673], [-0.17066673, 2.16117758]])
    mixture1 = GaussianMixture([g1], weights1)
    mixture2 = GaussianMixture([g1, g2, g3], weights2)
    mixture3 = GaussianMixture([g1, g2, g3, g4, g5], weights3)
    plotter = Plotter(-5, 5, -5, 5, 100)

    # Define parameter ranges
    num_samples_per_gaussian = powspace(50, 200, 10, 40, False)  # 50 to 200 samples
    window_sizes = powspace(1.15, 23, 10, 40, True)  # 0.05 to 1.0 window sizes
    # PNN architectures (prompt-compliant): 1 or 2 hidden layers, sigmoid activation.
    # Output is either ReLU or scaled sigmoid A·sigmoid(z).
    pnn_architectures = [
        {"hidden_layers": [20], "out": "sigmoid", "A": "auto"},
        {"hidden_layers": [30, 20], "out": "sigmoid", "A": "auto"},
        {"hidden_layers": [30, 20], "out": "relu", "A": "auto"},
        {"hidden_layers": [20], "out": "relu", "A": "auto"},
    ]

    def arch_label(cfg: dict) -> str:
        layers = "-".join(str(x) for x in cfg["hidden_layers"])
        if cfg["out"] == "relu":
            return f"MLP_{layers}_sigmoid_outReLU"
        A = cfg.get("A", "auto")
        if isinstance(A, str):
            return f"MLP_{layers}_sigmoid_outSigmoid_Aauto"
        return f"MLP_{layers}_sigmoid_outSigmoid_A{float(A):.2g}"

    mixtures = [mixture1, mixture2, mixture3]
    # --------------------------
    # Parzen Window Evaluation (Gaussian KDE)
    for mixture_idx, mixture in enumerate(mixtures):
        print(f"Processing Gaussian Mixture {mixture_idx + 1} with Parzen Window")
        errors = []
        sampled_points = []
        sampled_window = []

        pw_csv_path = os.path.join("logs", f"pw_errors_mixture{mixture_idx + 1}.csv")
        pw_txt_path = os.path.join("logs", f"pw_errors_mixture{mixture_idx + 1}.txt")
        csv_fh = None
        txt_fh = None
        csv_writer = None
        if bool(args.pw_write_logs) or bool(args.pw_only):
            csv_fh = open(pw_csv_path, "w", encoding="utf-8", newline="")
            txt_fh = open(pw_txt_path, "w", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_fh,
                fieldnames=[
                    "mixture",
                    "n",
                    "h1",
                    "h_n",
                    "grid_mse",
                    "grid_rmse",
                    "grid_max_abs_err",
                    "grid_mean_abs_err",
                    "val_avg_nll",
                ],
            )
            csv_writer.writeheader()

        best_by_val_nll = (float("inf"), None, None)  # (nll, n, h1)
        best_by_grid_mse = (float("inf"), None, None)

        for num_samples in num_samples_per_gaussian:
            num_samples = int(num_samples)
            # Reproducible sampling per mixture and n so h1 sweeps are comparable.
            if bool(args.pw_only):
                np.random.seed(int(args.pw_seed) + 1000 * int(mixture_idx + 1) + int(num_samples))

            samples_xy = mixture.sample_points(num_samples, with_pdf=False)
            train_xy, val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=(mixture_idx + 1))

            for window_size in window_sizes:
                h1 = float(window_size)
                parzen_estimator = ParzenWindowEstimator(train_xy, h1)
                _ = parzen_estimator.estimate_pdf(plotter)
                true_grid = mixture.get_mesh(plotter.pos)
                err = parzen_estimator.calculate_error_metrics(true_grid)
                grid_mse = float(err["Mean Squared Error"])

                ll_val = average_log_likelihood_kde(val_xy, train_xy, h1)
                val_nll = _nll_from_avg_loglik(ll_val)

                errors.append(grid_mse)
                sampled_points.append(num_samples)
                sampled_window.append(h1)

                # Track best configurations.
                if np.isfinite(val_nll) and val_nll < best_by_val_nll[0]:
                    best_by_val_nll = (float(val_nll), int(num_samples), float(h1))
                if np.isfinite(grid_mse) and grid_mse < best_by_grid_mse[0]:
                    best_by_grid_mse = (float(grid_mse), int(num_samples), float(h1))

                # Optional log export.
                if csv_writer is not None and csv_fh is not None and txt_fh is not None:
                    h_n = _effective_bandwidth(h1, int(train_xy.shape[0]))
                    csv_writer.writerow(
                        {
                            "mixture": int(mixture_idx + 1),
                            "n": int(num_samples),
                            "h1": float(h1),
                            "h_n": float(h_n),
                            "grid_mse": float(err["Mean Squared Error"]),
                            "grid_rmse": float(err["Root Mean Squared Error"]),
                            "grid_max_abs_err": float(err["Max Absolute Error"]),
                            "grid_mean_abs_err": float(err["Mean Absolute Error"]),
                            "val_avg_nll": float(val_nll),
                        }
                    )
                    txt_fh.write(
                        "SUMMARY: "
                        f"mixture={int(mixture_idx + 1)}, "
                        f"n={int(num_samples)}, "
                        f"h1={float(h1):.6g}, "
                        f"grid_mse={float(err['Mean Squared Error']):.6e}, "
                        f"val_avg_nll={float(val_nll):.6e}\n"
                    )

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np.array(sampled_points), np.array(sampled_window), np.array(errors), c='r')
        ax.set_title(f"Parzen Window Errors (Mixture {mixture_idx + 1})")
        ax.set_xlabel("Sampled Points")
        ax.set_ylabel("Window Size")
        ax.set_zlabel("Mean Squared Error")
        error_fig_filename = f"figures/Parzen_errors_mixture{mixture_idx + 1}.jpeg"
        plt.savefig(error_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved error figure: {error_fig_filename}")

        # Use data-only selection for the overlay (best by validation NLL), but also report best-by-MSE.
        print(
            f"Best Parzen (by Val NLL) for mixture {mixture_idx+1}: "
            f"samples = {best_by_val_nll[1]}, window = {best_by_val_nll[2]:.6g}, ValNLL = {best_by_val_nll[0]:.6g}"
        )
        print(
            f"Best Parzen (by grid MSE) for mixture {mixture_idx+1}: "
            f"samples = {best_by_grid_mse[1]}, window = {best_by_grid_mse[2]:.6g}, MSE = {best_by_grid_mse[0]:.6e}"
        )

        best_num_samples = int(best_by_val_nll[1]) if best_by_val_nll[1] is not None else int(sampled_points[int(np.argmin(errors))])
        best_window_size = float(best_by_val_nll[2]) if best_by_val_nll[2] is not None else float(sampled_window[int(np.argmin(errors))])

        if bool(args.pw_only):
            np.random.seed(int(args.pw_seed) + 1000 * int(mixture_idx + 1) + int(best_num_samples))
        samples_best = mixture.sample_points(best_num_samples, with_pdf=False)
        train_best, _val_best = split_train_validation(samples_best, val_fraction=0.2, seed=(mixture_idx + 1))
        parzen_best = ParzenWindowEstimator(train_best, best_window_size)
        estimated_pdf_best = parzen_best.estimate_pdf(plotter)
        real_pdf = mixture.get_mesh(plotter.pos)

        fig_overlay = plt.figure(figsize=(16, 9))
        ax_overlay = fig_overlay.add_subplot(projection='3d')
        ax_overlay.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.5, cmap='viridis')
        ax_overlay.plot_surface(plotter.X, plotter.Y, estimated_pdf_best, alpha=0.5, cmap='plasma')
        ax_overlay.set_title(
            f"Parzen Overlay (Mixture {mixture_idx+1})\nSamples = {best_num_samples}, Window = {best_window_size:.3f}"
        )
        overlay_fig_filename = f"figures/Parzen_overlay_mixture{mixture_idx + 1}.jpeg"
        plt.savefig(overlay_fig_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_overlay)
        print(f"Saved overlay figure: {overlay_fig_filename}")

        if csv_fh is not None:
            csv_fh.close()
        if txt_fh is not None:
            txt_fh.close()
        if csv_writer is not None:
            print(f"Saved PW error log CSV: {pw_csv_path}")
            print(f"Saved PW error log TXT: {pw_txt_path}")

    if bool(args.pw_only):
        # Explicitly stop here so PNN artifacts are not modified.
        return
    # --------------------------
    # Parzen Neural Network (PNN) Evaluation with learning-rate sweep and training surfaces
    for mixture_idx, mixture in enumerate(mixtures):
        print(f"Processing Gaussian Mixture {mixture_idx + 1} with PNN and LR sweep")

        # Define hyperparameters
        learning_rates = [5e-3]
        bandwidths = [2, 7, 12, 16]
        use_log_density = True
        # Prepare samples (~100 per Gaussian, scaled by weight)
        samples_xy = mixture.sample_points_weighted(100, with_pdf=False)
        # Split train/validation so we can evaluate without ground truth (data-only metric).
        train_xy, val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=mixture_idx + 1)
        # Optional boundary points set (shell outside plot rectangle)
        boundary_pts = sample_boundary_points_outside_plot(plotter, alpha=0.1, k=max(20, int(0.3 * len(train_xy))))

        # We will generate an explicit figure that compares lambda_boundary=0 vs >0 at the end
        # of the sweep, selecting the configuration that is best by validation NLL.

        def _h_tag(h: float) -> str:
            # Safe for filenames
            return f"{float(h):.2f}".replace(".", "p")

        arch_labels_all = [arch_label(cfg) for cfg in pnn_architectures]

        # Collect results across bandwidths so we can make one learning-results figure per mixture.
        n_arch_total = len(pnn_architectures)
        per_arch_eval_mse_by_h: list[list[list[float]]] = [[] for _ in range(n_arch_total)]
        per_arch_final_mse_by_h: list[list[float]] = [[] for _ in range(n_arch_total)]
        kde_mse_by_h: list[float] = []

        # Data-only metrics (validation average log-likelihood / NLL).
        kde_val_ll_by_h: list[float] = []
        kde_val_nll_by_h: list[float] = []
        per_arch_val_ll_by_h: list[list[float]] = [[] for _ in range(n_arch_total)]
        per_arch_val_nll_by_h: list[list[float]] = [[] for _ in range(n_arch_total)]

        for bandwidth in bandwidths:
            # Precompute KDE on grid once per bandwidth (same across architectures).
            kde_estimator = ParzenWindowEstimator(train_xy, window_size=float(bandwidth))
            estimated_pdf_kde = kde_estimator.estimate_pdf(plotter)
            real_pdf = mixture.get_mesh(plotter.pos)
            mse_kde = float(np.mean((estimated_pdf_kde - real_pdf) ** 2))
            kde_mse_by_h.append(mse_kde)

            # Data-only validation metric (no oracle): average log-likelihood of held-out points.
            ll_kde_val = average_log_likelihood_kde(val_xy, train_xy, float(bandwidth))
            nll_kde_val = _nll_from_avg_loglik(ll_kde_val)
            kde_val_ll_by_h.append(ll_kde_val)
            kde_val_nll_by_h.append(nll_kde_val)
            print(
                f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}: "
                f"KDE EvalMSE={mse_kde:.6e}, ValAvgLogLik={ll_kde_val:.6e} (ValAvgNLL={nll_kde_val:.6e})"
            )

            # Track per-architecture results for this bandwidth
            arch_labels: list[str] = []
            per_arch_eval_mse_hist: list[list[float]] = []
            per_arch_train_loss_hist: list[list[float]] = []
            per_arch_est_pdf: list[np.ndarray] = []
            per_arch_final_mse: list[float] = []

            for cfg_idx, cfg in enumerate(pnn_architectures):
                label = arch_label(cfg)
                arch_labels.append(label)
                log_filename = f"logs/mixture{mixture_idx+1}_{label}_h1_{_h_tag(bandwidth)}.txt"
                best_model = None
                best_final_mse = float("inf")
                best_hist: list[float] = []
                best_train_hist: list[float] = []

                for lr in learning_rates:
                    with open(log_filename, "a") as log_file:
                        pnn = ParzenNeuralNetwork(
                            hidden_layers=cfg["hidden_layers"],
                            output_activation=cfg["out"],
                            output_scale=cfg.get("A", "auto"),
                            density_parameterization="log_density" if use_log_density else "density",
                        )
                        _final_loss, mse_hist, train_hist = pnn.train_network(
                            train_xy,
                            plotter,
                            bandwidth=float(bandwidth),
                            mixture=mixture,
                            log_file=log_file,
                            learning_rate=float(lr),
                            epochs=3500,
                            boundary_points=boundary_pts,
                            lambda_boundary=0.0,
                            verbose=True,
                            loss_mode="mse" if use_log_density else "relative",
                            weight_decay=0.0,
                            num_uniform_points=len(train_xy) if use_log_density else 0,
                        )

                    # Prefer using the final tracked eval MSE when available.
                    if len(mse_hist) > 0:
                        final_mse = float(mse_hist[-1])
                    else:
                        est_pdf_tmp = pnn.estimate_pdf(plotter)
                        final_mse = float(np.mean((est_pdf_tmp - real_pdf) ** 2))

                    print(
                        f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}, {label}, LR={lr}: Final MSE = {final_mse:.6e}"
                    )

                    if final_mse < best_final_mse:
                        best_final_mse = final_mse
                        best_model = pnn
                        best_hist = list(mse_hist)
                        best_train_hist = list(train_hist)

                assert best_model is not None
                est_pdf = best_model.estimate_pdf(plotter)
                per_arch_est_pdf.append(est_pdf)
                per_arch_eval_mse_hist.append(best_hist)
                per_arch_train_loss_hist.append(best_train_hist)
                per_arch_final_mse.append(float(np.mean((est_pdf - real_pdf) ** 2)))

                # Data-only validation log-likelihood for this trained PNN.
                ll_pnn_val = average_log_likelihood_pnn_on_domain(best_model, val_xy, plotter)
                nll_pnn_val = _nll_from_avg_loglik(ll_pnn_val)
                per_arch_val_ll_by_h[cfg_idx].append(ll_pnn_val)
                per_arch_val_nll_by_h[cfg_idx].append(nll_pnn_val)
                print(
                    f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}, {label}: "
                    f"ValAvgLogLik={ll_pnn_val:.6e} (ValAvgNLL={nll_pnn_val:.6e})"
                )

                # Append a compact, machine-parseable summary line to the same log file.
                try:
                    with open(log_filename, "a", encoding="utf-8") as log_file:
                        log_file.write(
                            "SUMMARY: "
                            f"h1={float(bandwidth):.6g}, "
                            f"label={label}, "
                            f"final_grid_mse={float(per_arch_final_mse[-1]):.6e}, "
                            f"val_avg_nll={float(nll_pnn_val):.6e}\n"
                        )
                except OSError:
                    pass

                # Persist results across bandwidths for consolidated learning plot
                per_arch_eval_mse_by_h[cfg_idx].append(best_hist)
                per_arch_final_mse_by_h[cfg_idx].append(float(np.mean((est_pdf - real_pdf) ** 2)))

            # --------------------------
            # Figure 1: Overlays for this bandwidth (columns = architectures, 2 rows)
            n_arch = len(pnn_architectures)
            fig_ov = plt.figure(figsize=(5 * n_arch, 10))
            fig_ov.suptitle(f"overlays with h_1 = {bandwidth}")

            for j, label in enumerate(arch_labels):
                # Row 1: KDE vs True
                ax1 = fig_ov.add_subplot(2, n_arch, j + 1, projection='3d')
                ax1.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
                ax1.plot_surface(plotter.X, plotter.Y, estimated_pdf_kde, alpha=0.45, cmap='cividis')
                ax1.set_title(f"{label}\nKDE vs True")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("pdf")

                # Row 2: PNN vs True
                ax2 = fig_ov.add_subplot(2, n_arch, n_arch + j + 1, projection='3d')
                ax2.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
                ax2.plot_surface(plotter.X, plotter.Y, per_arch_est_pdf[j], alpha=0.45, cmap='plasma')
                ax2.set_title(f"{label}\nPNN vs True")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_zlabel("pdf")

            overlays_fig_filename = f"figures/overlays_h1_{_h_tag(bandwidth)}_mixture{mixture_idx+1}.jpeg"
            plt.tight_layout()
            plt.savefig(overlays_fig_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_ov)
            print(f"Saved overlays figure: {overlays_fig_filename}")

            # --------------------------
            # Figure 2: Learning results for this bandwidth (columns = architectures, 2 rows)
            fig_lr = plt.figure(figsize=(5 * n_arch, 10))
            fig_lr.suptitle(f"learning results h_1 = {bandwidth}")

            for j, label in enumerate(arch_labels):
                # Row 1: training results (zoom to bottom 20% of MSE range to make minima visible)
                ax_tr = fig_lr.add_subplot(2, n_arch, j + 1)
                ax_tr_right = ax_tr.twinx()
                eval_hist = np.asarray(per_arch_eval_mse_hist[j], dtype=float)
                train_hist = np.asarray(per_arch_train_loss_hist[j], dtype=float)

                eval_line = None
                if eval_hist.size > 0:
                    eval_epochs = np.arange(eval_hist.size)
                    eval_line = ax_tr.plot(
                        eval_epochs,
                        eval_hist,
                        color='tab:blue',
                        linewidth=2.2,
                        label='Eval MSE',
                    )[0]
                    bounds = _bottom_20_bounds(eval_hist)
                    if bounds is not None:
                        ax_tr.set_ylim(*bounds)

                train_line = None
                if train_hist.size > 0:
                    train_epochs = np.arange(train_hist.size)
                    train_line = ax_tr_right.plot(
                        train_epochs,
                        train_hist,
                        color='tab:orange',
                        linewidth=2.2,
                        label='Train MSE',
                    )[0]
                    bounds = _bottom_20_bounds(train_hist)
                    if bounds is not None:
                        ax_tr_right.set_ylim(*bounds)

                ax_tr.set_title(f"{label}\nTraining results (zoomed)")
                ax_tr.set_xlabel("Epoch")
                ax_tr.set_ylabel("Eval MSE", color='tab:blue')
                ax_tr_right.set_ylabel("Train MSE", color='tab:orange')
                ax_tr.tick_params(axis='y', colors='tab:blue')
                ax_tr_right.tick_params(axis='y', colors='tab:orange')
                ax_tr.grid(True, alpha=0.3)

                legend_lines = [line for line in (eval_line, train_line) if line is not None]
                if legend_lines:
                    legend_labels = [line.get_label() for line in legend_lines]
                    ax_tr.legend(legend_lines, legend_labels, loc='upper right', fontsize='small')

                # Row 2: compare final MSEs (PNN vs KDE)
                ax_cmp = fig_lr.add_subplot(2, n_arch, n_arch + j + 1)
                mse_pnn = float(per_arch_final_mse[j])
                ax_cmp.bar([0, 1], [mse_pnn, mse_kde], color=['tab:purple', 'tab:gray'])
                ax_cmp.set_xticks([0, 1])
                ax_cmp.set_xticklabels(["PNN", "KDE"])
                ax_cmp.set_title(f"{label}\nFinal grid MSE")
                ax_cmp.set_ylabel("MSE")
                ax_cmp.grid(True, axis='y', alpha=0.3)

            learning_fig_filename = f"figures/learning_results_h1_{_h_tag(bandwidth)}_mixture{mixture_idx+1}.jpeg"
            plt.tight_layout()
            plt.savefig(learning_fig_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_lr)
            print(f"Saved learning results figure: {learning_fig_filename}")

        # --------------------------
        # Consolidated learning results across all bandwidths for this mixture.
        # Layout: columns = architectures, 2 rows
        n_arch = len(pnn_architectures)
        fig_lr_all = plt.figure(figsize=(5 * n_arch, 10))
        fig_lr_all.suptitle(f"learning results (bandwidth sweep) mixture {mixture_idx + 1}")

        # Build surfaces: bandwidth x epochs
        max_epochs_tracked = 0
        for j in range(n_arch):
            for hist in per_arch_eval_mse_by_h[j]:
                max_epochs_tracked = max(max_epochs_tracked, len(hist))

        # Determine zoomed z-range (bottom 20% of overall range)
        all_vals = []
        for j in range(n_arch):
            for hist in per_arch_eval_mse_by_h[j]:
                if len(hist) > 0:
                    all_vals.extend([float(x) for x in hist if np.isfinite(x)])
        if len(all_vals) == 0:
            global_zmin, global_zmax = 0.0, 1.0
        else:
            global_zmin = float(np.min(all_vals))
            global_zmax = float(np.max(all_vals))
        zoom_zmax = global_zmin + 0.20 * max(1e-18, (global_zmax - global_zmin))

        E = np.arange(max_epochs_tracked) if max_epochs_tracked > 0 else np.arange(1)
        H = np.array([float(h) for h in bandwidths], dtype=float)
        EE, HH = np.meshgrid(E, H)

        for j, label in enumerate([arch_label(cfg) for cfg in pnn_architectures]):
            # Row 1: surface over (epochs, bandwidth)
            ax_surf = fig_lr_all.add_subplot(2, n_arch, j + 1, projection='3d')

            surface = np.zeros((len(bandwidths), len(E)), dtype=float)
            for i_h in range(len(bandwidths)):
                hist = per_arch_eval_mse_by_h[j][i_h] if i_h < len(per_arch_eval_mse_by_h[j]) else []
                if len(hist) > 0:
                    vals = np.asarray(hist, dtype=float)
                    surface[i_h, : len(vals)] = vals
                    if len(vals) < len(E):
                        surface[i_h, len(vals) :] = float(vals[-1])
                else:
                    surface[i_h, :] = np.nan

            ax_surf.plot_surface(EE, HH, surface, cmap='viridis', linewidth=0, antialiased=False, alpha=0.8)
            ax_surf.set_title(f"{label}\nTraining results")
            ax_surf.set_xlabel("Epoch")
            ax_surf.set_ylabel("h_1")
            ax_surf.set_zlabel("Eval MSE")
            ax_surf.set_zlim(global_zmin, zoom_zmax)
            ax_surf.grid(True)
            ax_surf.view_init(elev=30, azim=-135)

            # Row 2: MSE vs bandwidth (PNN vs KDE)
            ax_line = fig_lr_all.add_subplot(2, n_arch, n_arch + j + 1)
            pnn_mse = np.asarray(per_arch_final_mse_by_h[j], dtype=float)
            kde_mse = np.asarray(kde_mse_by_h, dtype=float)
            ax_line.plot(H, pnn_mse, marker='o', color='tab:purple', label='PNN')
            ax_line.plot(H, kde_mse, marker='o', color='tab:gray', label='KDE')
            ax_line.set_title(f"{label}\nFinal grid MSE vs h_1")
            ax_line.set_xlabel("h_1")
            ax_line.set_ylabel("MSE")
            ax_line.grid(True, alpha=0.3)
            ax_line.legend()

        lr_all_filename = f"figures/learning_results_bandwidth_sweep_mixture{mixture_idx+1}.jpeg"
        plt.tight_layout()
        plt.savefig(lr_all_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_lr_all)
        print(f"Saved consolidated learning results figure: {lr_all_filename}")

        # --------------------------
        # NEW: Data-only cross-validation plot (validation NLL) across bandwidths.
        fig_cv = plt.figure(figsize=(12, 6))
        ax_cv = fig_cv.add_subplot(111)
        H = np.array([float(h) for h in bandwidths], dtype=float)
        ax_cv.plot(H, np.asarray(kde_val_nll_by_h, dtype=float), marker='o', color='tab:gray', label='KDE Val NLL')
        for cfg_idx, cfg in enumerate(pnn_architectures):
            label = arch_label(cfg)
            ax_cv.plot(
                H,
                np.asarray(per_arch_val_nll_by_h[cfg_idx], dtype=float),
                marker='o',
                linewidth=2.0,
                label=f"PNN Val NLL: {label}",
            )
        ax_cv.set_title(f"Validation NLL (held-out split) vs h_1 — mixture {mixture_idx+1}")
        ax_cv.set_xlabel("h_1")
        ax_cv.set_ylabel("Avg NLL on held-out points (lower is better)")
        ax_cv.grid(True, alpha=0.3)
        ax_cv.legend(fontsize='small')
        cv_filename = f"figures/validation_nll_bandwidth_sweep_mixture{mixture_idx+1}.jpeg"
        plt.tight_layout()
        plt.savefig(cv_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_cv)
        print(f"Saved validation NLL figure: {cv_filename}")

        # --------------------------
        # NEW: Boundary penalty comparison figure (lambda=0 vs lambda>0) for the best-by-NLL config.
        # Select best (min) validation NLL across all (arch, bandwidth).
        best_cfg_idx = None
        best_h1 = None
        best_nll = float("inf")
        for cfg_idx in range(n_arch_total):
            nlls = per_arch_val_nll_by_h[cfg_idx]
            for i_h, nll in enumerate(nlls):
                if np.isfinite(nll) and float(nll) < best_nll:
                    best_nll = float(nll)
                    best_cfg_idx = int(cfg_idx)
                    best_h1 = float(bandwidths[i_h])

        if best_cfg_idx is not None and best_h1 is not None:
            cfg = pnn_architectures[best_cfg_idx]
            label = arch_label(cfg)
            lam0 = 0.0
            lam1 = 1e-2
            demo_epochs = 2000
            demo_lr = float(learning_rates[0])
            print(
                f"Boundary penalty comparison (mixture {mixture_idx+1}): "
                f"best-by-NLL: h1={best_h1:.3f}, arch={label}; training lambda={lam0} vs {lam1}"
            )

            models: dict[float, ParzenNeuralNetwork] = {}
            metrics: dict[float, dict[str, float]] = {}
            for lam in (lam0, lam1):
                pnn_demo = ParzenNeuralNetwork(
                    hidden_layers=cfg["hidden_layers"],
                    output_activation=cfg["out"],
                    output_scale=cfg.get("A", "auto"),
                    density_parameterization="log_density" if use_log_density else "density",
                )
                _final_loss, _mse_hist, _train_hist = pnn_demo.train_network(
                    train_xy,
                    plotter,
                    bandwidth=float(best_h1),
                    mixture=mixture,
                    log_file=None,
                    learning_rate=demo_lr,
                    epochs=demo_epochs,
                    boundary_points=boundary_pts,
                    lambda_boundary=float(lam),
                    verbose=False,
                    loss_mode="mse" if use_log_density else "relative",
                    weight_decay=0.0,
                    num_uniform_points=len(train_xy) if use_log_density else 0,
                )
                models[float(lam)] = pnn_demo

                boundary_mean = mean_unnormalized_density_on_points(pnn_demo, boundary_pts)
                val_ll = average_log_likelihood_pnn_on_domain(pnn_demo, val_xy, plotter)
                val_nll = _nll_from_avg_loglik(val_ll)
                metrics[float(lam)] = {
                    "boundary_mean": float(boundary_mean),
                    "val_ll": float(val_ll),
                    "val_nll": float(val_nll),
                }

            # Plot: KDE vs True, and PNN lambda=0 vs True, PNN lambda>0 vs True.
            kde_for_demo = ParzenWindowEstimator(train_xy, window_size=float(best_h1))
            kde_pdf = kde_for_demo.estimate_pdf(plotter)
            true_pdf = mixture.get_mesh(plotter.pos)
            pnn0_pdf = models[lam0].estimate_pdf(plotter)
            pnn1_pdf = models[lam1].estimate_pdf(plotter)

            fig_b = plt.figure(figsize=(16, 9))
            fig_b.suptitle(f"Boundary penalty comparison — mixture {mixture_idx+1}\n{label}, h_1={best_h1:.3f}")

            ax_kde = fig_b.add_subplot(1, 3, 1, projection='3d')
            ax_kde.plot_surface(plotter.X, plotter.Y, true_pdf, alpha=0.45, cmap='viridis')
            ax_kde.plot_surface(plotter.X, plotter.Y, kde_pdf, alpha=0.45, cmap='cividis')
            ax_kde.set_title("KDE vs True")

            ax0 = fig_b.add_subplot(1, 3, 2, projection='3d')
            ax0.plot_surface(plotter.X, plotter.Y, true_pdf, alpha=0.45, cmap='viridis')
            ax0.plot_surface(plotter.X, plotter.Y, pnn0_pdf, alpha=0.45, cmap='plasma')
            m0 = metrics[lam0]
            ax0.set_title(
                f"PNN vs True (lambda={lam0:g})\nValNLL={m0['val_nll']:.3g}, boundaryMean={m0['boundary_mean']:.3g}"
            )

            ax1 = fig_b.add_subplot(1, 3, 3, projection='3d')
            ax1.plot_surface(plotter.X, plotter.Y, true_pdf, alpha=0.45, cmap='viridis')
            ax1.plot_surface(plotter.X, plotter.Y, pnn1_pdf, alpha=0.45, cmap='plasma')
            m1 = metrics[lam1]
            ax1.set_title(
                f"PNN vs True (lambda={lam1:g})\nValNLL={m1['val_nll']:.3g}, boundaryMean={m1['boundary_mean']:.3g}"
            )

            for ax in (ax_kde, ax0, ax1):
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("pdf")

            b_filename = f"figures/boundary_penalty_comparison_mixture{mixture_idx+1}.jpeg"
            plt.tight_layout()
            plt.savefig(b_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_b)
            print(f"Saved boundary-penalty comparison figure: {b_filename}")

            # --------------------------
            # NEW (optional): Uniform supervision ablation (no-uniform vs uniform) for the same best-by-NLL config.
            # Goal: show that adding interior Parzen targets reduces extrapolation artifacts far from samples.
            # Disabled by default; enable with: --uniform-supervision-demo
            if "--uniform-supervision-demo" in sys.argv:
                try:
                    seed_base = 9000 + int(mixture_idx + 1)
                    np.random.seed(seed_base)
                    torch.manual_seed(seed_base)

                    pnn_no_uniform = ParzenNeuralNetwork(
                        hidden_layers=cfg["hidden_layers"],
                        output_activation=cfg["out"],
                        output_scale=cfg.get("A", "auto"),
                        density_parameterization="log_density" if use_log_density else "density",
                    )
                    _final_loss, _mse_hist, _train_hist = pnn_no_uniform.train_network(
                        train_xy,
                        plotter,
                        bandwidth=float(best_h1),
                        mixture=mixture,
                        log_file=None,
                        learning_rate=demo_lr,
                        epochs=demo_epochs,
                        boundary_points=None,
                        lambda_boundary=0.0,
                        verbose=False,
                        loss_mode="mse" if use_log_density else "relative",
                        weight_decay=0.0,
                        num_uniform_points=0,
                    )

                    np.random.seed(seed_base)
                    torch.manual_seed(seed_base)
                    pnn_with_uniform = ParzenNeuralNetwork(
                        hidden_layers=cfg["hidden_layers"],
                        output_activation=cfg["out"],
                        output_scale=cfg.get("A", "auto"),
                        density_parameterization="log_density" if use_log_density else "density",
                    )
                    _final_loss, _mse_hist, _train_hist = pnn_with_uniform.train_network(
                        train_xy,
                        plotter,
                        bandwidth=float(best_h1),
                        mixture=mixture,
                        log_file=None,
                        learning_rate=demo_lr,
                        epochs=demo_epochs,
                        boundary_points=None,
                        lambda_boundary=0.0,
                        verbose=False,
                        loss_mode="mse" if use_log_density else "relative",
                        weight_decay=0.0,
                        num_uniform_points=len(train_xy) if use_log_density else 0,
                    )

                    # Compare both models on the same grid and on a boundary-ring statistic.
                    pdf_no = pnn_no_uniform.estimate_pdf(plotter)
                    pdf_u = pnn_with_uniform.estimate_pdf(plotter)
                    ring_no = _mean_density_on_boundary_ring(pdf_no, plotter, ring_width=1.0)
                    ring_u = _mean_density_on_boundary_ring(pdf_u, plotter, ring_width=1.0)
                    val_ll_no = average_log_likelihood_pnn_on_domain(pnn_no_uniform, val_xy, plotter)
                    val_ll_u = average_log_likelihood_pnn_on_domain(pnn_with_uniform, val_xy, plotter)
                    val_nll_no = _nll_from_avg_loglik(val_ll_no)
                    val_nll_u = _nll_from_avg_loglik(val_ll_u)

                    fig_u = plt.figure(figsize=(14, 6))
                    fig_u.suptitle(
                        f"Uniform supervision ablation — mixture {mixture_idx+1}\n{label}, h_1={best_h1:.3f}"
                    )
                    # Use 2D heatmaps to make far-field artifacts easy to spot.
                    ax_a = fig_u.add_subplot(1, 2, 1)
                    im0 = ax_a.imshow(
                        pdf_no,
                        origin="lower",
                        extent=(
                            float(np.min(plotter.x)),
                            float(np.max(plotter.x)),
                            float(np.min(plotter.y)),
                            float(np.max(plotter.y)),
                        ),
                        aspect="auto",
                        cmap="viridis",
                    )
                    ax_a.scatter(train_xy[:, 0], train_xy[:, 1], s=6, c="white", alpha=0.7)
                    ax_a.set_title(f"No uniform points\nValNLL={val_nll_no:.3g}, ringMean={ring_no:.3g}")
                    ax_a.set_xlabel("x")
                    ax_a.set_ylabel("y")
                    fig_u.colorbar(im0, ax=ax_a, fraction=0.046, pad=0.04)

                    ax_b = fig_u.add_subplot(1, 2, 2)
                    im1 = ax_b.imshow(
                        pdf_u,
                        origin="lower",
                        extent=(
                            float(np.min(plotter.x)),
                            float(np.max(plotter.x)),
                            float(np.min(plotter.y)),
                            float(np.max(plotter.y)),
                        ),
                        aspect="auto",
                        cmap="viridis",
                    )
                    ax_b.scatter(train_xy[:, 0], train_xy[:, 1], s=6, c="white", alpha=0.7)
                    ax_b.set_title(f"With uniform points\nValNLL={val_nll_u:.3g}, ringMean={ring_u:.3g}")
                    ax_b.set_xlabel("x")
                    ax_b.set_ylabel("y")
                    fig_u.colorbar(im1, ax=ax_b, fraction=0.046, pad=0.04)

                    u_filename = f"figures/uniform_supervision_comparison_mixture{mixture_idx+1}.jpeg"
                    plt.tight_layout()
                    plt.savefig(u_filename, dpi=300, bbox_inches="tight")
                    plt.close(fig_u)
                    print(f"Saved uniform supervision comparison figure: {u_filename}")

                    # Machine-readable summary (kept separate from sweep_results_*.json to avoid breaking any consumers).
                    u_payload = {
                        "mixture": int(mixture_idx + 1),
                        "label": str(label),
                        "h1": float(best_h1),
                        "epochs": int(demo_epochs),
                        "learning_rate": float(demo_lr),
                        "ring_width": 1.0,
                        "no_uniform": {"val_avg_nll": float(val_nll_no), "boundary_ring_mean": float(ring_no)},
                        "with_uniform": {"val_avg_nll": float(val_nll_u), "boundary_ring_mean": float(ring_u)},
                        "figure": str(u_filename),
                    }
                    u_out = f"results/uniform_supervision_demo_mixture{mixture_idx+1}.json"
                    os.makedirs("results", exist_ok=True)
                    with open(u_out, "w", encoding="utf-8") as f:
                        json.dump(u_payload, f, indent=2)
                    print(f"Saved uniform supervision demo JSON: {u_out}")
                except Exception as e:
                    print(f"WARN: uniform supervision ablation skipped due to: {e}")

            # Export a JSON summary for the whole sweep (report-ready, machine-readable).
            try:
                sweep_payload = {
                    "mixture": int(mixture_idx + 1),
                    "bandwidths_h1": [float(h) for h in bandwidths],
                    "learning_rates": [float(lr) for lr in learning_rates],
                    "architectures": [
                        {
                            "label": arch_labels_all[i],
                            "hidden_layers": list(pnn_architectures[i]["hidden_layers"]),
                            "hidden_activation": "sigmoid",
                            "output": str(pnn_architectures[i]["out"]),
                            "output_scale": pnn_architectures[i].get("A", "auto"),
                            "density_parameterization": "log_density" if use_log_density else "density",
                        }
                        for i in range(len(pnn_architectures))
                    ],
                    "kde": {
                        "grid_mse": [float(v) for v in kde_mse_by_h],
                        "val_avg_nll": [float(v) for v in kde_val_nll_by_h],
                        "val_avg_loglik": [float(v) for v in kde_val_ll_by_h],
                    },
                    "pnn": {
                        "final_grid_mse": [[float(v) for v in per_arch_final_mse_by_h[i]] for i in range(n_arch_total)],
                        "val_avg_nll": [[float(v) for v in per_arch_val_nll_by_h[i]] for i in range(n_arch_total)],
                        "val_avg_loglik": [[float(v) for v in per_arch_val_ll_by_h[i]] for i in range(n_arch_total)],
                    },
                    "best_by_val_nll": {
                        "label": arch_label(pnn_architectures[int(best_cfg_idx)]),
                        "h1": float(best_h1),
                        "val_avg_nll": float(best_nll),
                    },
                    "boundary_penalty_demo": {
                        "h1": float(best_h1),
                        "label": label,
                        "lambda_0": {"lambda": float(lam0), **metrics[float(lam0)]},
                        "lambda_1": {"lambda": float(lam1), **metrics[float(lam1)]},
                    },
                }
                out_path = f"results/sweep_results_mixture{mixture_idx+1}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(sweep_payload, f, indent=2)
                print(f"Saved sweep results JSON: {out_path}")
            except OSError:
                pass


if __name__ == "__main__":
    # Lightweight CLI flags (kept minimal to avoid adding dependencies).
    #   --benchmark-complexity-only : only run the O(n^2) target-generation benchmark
    #   --benchmark-complexity      : run the benchmark after the full experiment sweep
    if "--benchmark-complexity-only" in sys.argv:
        benchmark_loocv_target_generation()
        raise SystemExit(0)

    if "--uniform-supervision-demo-only" in sys.argv:
        run_uniform_supervision_demo_only()
        raise SystemExit(0)

    main()

    if "--benchmark-complexity" in sys.argv:
        benchmark_loocv_target_generation()
