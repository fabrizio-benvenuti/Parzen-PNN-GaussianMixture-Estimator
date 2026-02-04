import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
import torch
import torch.nn as nn
import torch.optim as optim

def _pnn_cache_dir() -> str:
    return os.path.join("results", "pnn_cache")


def _pnn_sweep_cache_path(*, mixture_idx: int) -> str:
    # One cache per mixture index; contains all architectures/bandwidths/runs.
    return os.path.join(_pnn_cache_dir(), f"pnn_sweep_mixture{int(mixture_idx)+1}.pt")


def _tensor_to_cpu_state_dict(state_dict: dict) -> dict:
    out: dict = {}
    for k, v in state_dict.items():
        try:
            out[k] = v.detach().cpu()
        except Exception:
            out[k] = v
    return out


def _pnn_instantiate_from_cache(meta: dict) -> "ParzenNeuralNetwork":
    pnn = ParzenNeuralNetwork(
        hidden_layers=list(meta["hidden_layers"]),
        output_activation=str(meta.get("output_activation", meta.get("out", "relu"))),
        output_scale=meta.get("init_output_scale", meta.get("A", "auto")),
        density_parameterization=str(meta.get("density_parameterization", "density")),
    )
    # Output scale may be auto-inferred during training and is not in state_dict.
    if "trained_output_scale" in meta:
        pnn.output_scale = meta["trained_output_scale"]
    return pnn


def _safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def _torch_load_cache_trusted(path: str) -> dict:
    """Load a locally-produced cache file.

    PyTorch 2.6 changed torch.load default to weights_only=True, which rejects
    some pickled non-tensor objects (e.g., numpy arrays) used by older caches.
    We prefer safe loading, but fall back to weights_only=False for backward
    compatibility when the file is trusted (it is created locally by this script).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch without weights_only.
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _to_numpy_array(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

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


def average_log_likelihood_kde_on_domain(
    points_eval_xy: np.ndarray,
    points_train_xy: np.ndarray,
    plotter,
    h1: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Average log-likelihood for KDE renormalized on the plotter domain D.

    This makes the likelihood model comparable to the PNN likelihood that is normalized
    by a Riemann sum on the same finite grid domain.
    """
    X = np.asarray(points_eval_xy, dtype=float)
    T = np.asarray(points_train_xy, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points_eval_xy must have shape (m, 2)")
    if T.ndim != 2 or T.shape[1] != 2:
        raise ValueError("points_train_xy must have shape (n, 2)")

    # Compute normalizer on the grid for the same KDE.
    grid_points = np.c_[plotter.X.ravel(), plotter.Y.ravel()]
    grid_dens = compute_kde(grid_points, T, float(h1))
    dx, dy = _dx_dy_from_plotter(plotter)
    Z = float(np.sum(grid_dens) * (dx * dy))
    Z = max(Z, float(eps))

    dens = compute_kde(X, T, float(h1))
    dens = np.maximum(dens / Z, float(eps))
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


def benchmark_loocv_target_dynamic_range(
    *,
    mixture_index: int = 2,
    h1: float = 12.0,
    n_values: list[int] | None = None,
    seeds: list[int] | None = None,
    plot_grid: int = 100,
) -> tuple[str, str]:
    """Quantify how leave-one-out KDE targets change with n under h_n = h1/sqrt(n-1).

    This addresses the review point that the PNN regression target is not fixed: as n increases,
    h_n shrinks and the targets typically become sharper and more ill-conditioned.

    Outputs:
      - results/loocv_target_dynamic_range.csv
      - figures/loocv_target_dynamic_range.jpeg
    """
    if n_values is None:
        n_values = [50, 100, 200, 400, 800]
    if seeds is None:
        seeds = [0, 1, 2]

    mixtures = _make_default_mixtures()
    mixture = mixtures[int(mixture_index)]

    rows: list[dict[str, float]] = []
    for n in n_values:
        n = int(n)
        for seed in seeds:
            np.random.seed(int(seed) + 10_000 * int(mixture_index + 1) + 100 * int(n))
            samples_xy = mixture.sample_points_weighted(int(max(10, n // 5)), with_pdf=False)
            # Ensure we have exactly n points (weighted sampling may not give exact n).
            if samples_xy.shape[0] < n:
                extra = mixture.sample_points(n - int(samples_xy.shape[0]), with_pdf=False)
                samples_xy = np.vstack([samples_xy, extra])
            elif samples_xy.shape[0] > n:
                samples_xy = samples_xy[:n]

            train_xy, _val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=int(seed))
            y = compute_leave_one_out_kde_targets(train_xy, float(h1))
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            y = y[y > 0]
            if y.size == 0:
                continue

            rows.append(
                {
                    "mixture": float(mixture_index + 1),
                    "seed": float(seed),
                    "n_train": float(train_xy.shape[0]),
                    "h1": float(h1),
                    "h_n": float(_effective_bandwidth(float(h1), int(train_xy.shape[0]))),
                    "y_min": float(np.min(y)),
                    "y_med": float(np.median(y)),
                    "y_max": float(np.max(y)),
                    "log10_range": float(np.log10(np.max(y)) - np.log10(np.min(y))),
                }
            )

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    csv_path = os.path.join("results", "loocv_target_dynamic_range.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mixture", "seed", "n_train", "h1", "h_n", "y_min", "y_med", "y_max", "log10_range"],
        )
        w.writeheader()
        w.writerows(rows)

    # Plot mean±std log10 dynamic range vs n_train.
    import matplotlib.pyplot as plt

    n_arr = np.array([r["n_train"] for r in rows], dtype=float)
    r_arr = np.array([r["log10_range"] for r in rows], dtype=float)
    unique_n = np.unique(n_arr)
    means = []
    stds = []
    for nn in unique_n:
        vals = r_arr[n_arr == nn]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(unique_n, means, yerr=stds, marker="o", capsize=4)
    ax.set_title("LOO KDE target dynamic range vs n (log10 max/min)")
    ax.set_xlabel("n_train")
    ax.set_ylabel("log10 range")
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join("figures", "loocv_target_dynamic_range.jpeg")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved target dynamic-range CSV: {csv_path}")
    print(f"Saved target dynamic-range figure: {fig_path}")
    return csv_path, fig_path


def run_uniform_supervision_ablation_multiseed(
    *,
    seeds: list[int] | None = None,
    h1_by_mixture: dict[int, float] | None = None,
    epochs: int = 800,
    learning_rate: float = 5e-3,
    plot_grid: int = 100,
) -> tuple[str, str]:
    """Run a multi-seed uniform-supervision ablation across mixtures 1..3.

    Outputs:
      - results/uniform_supervision_ablation_multiseed.csv
      - figures/uniform_supervision_ablation_multiseed.jpeg
    """
    if seeds is None:
        seeds = [0, 1, 2]
    # Default: use the report's best-by-NLL bandwidths as representative.
    if h1_by_mixture is None:
        h1_by_mixture = {1: 7.0, 2: 7.0, 3: 12.0}

    mixtures = _make_default_mixtures()
    plotter = Plotter(-5, 5, -5, 5, int(plot_grid))

    rows: list[dict[str, float]] = []
    for mixture_idx, mixture in enumerate(mixtures, start=1):
        h1 = float(h1_by_mixture[int(mixture_idx)])
        for seed in seeds:
            # Deterministic sampling + deterministic split.
            np.random.seed(int(seed) + 10_000 * int(mixture_idx))
            torch.manual_seed(int(seed) + 10_000 * int(mixture_idx))

            samples_xy = mixture.sample_points_weighted(100, with_pdf=False)
            train_xy, val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=int(seed))

            boundary_pts = sample_boundary_points_outside_convex_hull(
                samples_xy, alpha=0.1, k=max(20, int(0.3 * len(train_xy)))
            )

            def _train(num_uniform_points: int) -> ParzenNeuralNetwork:
                model = ParzenNeuralNetwork(
                    hidden_layers=[30, 20],
                    output_activation="relu",
                    density_parameterization="log_density",
                )
                model.train_network(
                    train_xy,
                    plotter,
                    bandwidth=float(h1),
                    learning_rate=float(learning_rate),
                    epochs=int(epochs),
                    verbose=False,
                    loss_mode="mse",
                    num_uniform_points=int(num_uniform_points),
                    boundary_points=None,
                    lambda_boundary=0.0,
                )
                return model

            # Train the two variants.
            pnn_no = _train(0)
            pnn_u = _train(len(train_xy))

            # Metrics: PNN NLL on D, plus simple tail proxies.
            val_nll_no = _nll_from_avg_loglik(average_log_likelihood_pnn_on_domain(pnn_no, val_xy, plotter))
            val_nll_u = _nll_from_avg_loglik(average_log_likelihood_pnn_on_domain(pnn_u, val_xy, plotter))
            pdf_no = pnn_no.estimate_pdf(plotter)
            pdf_u = pnn_u.estimate_pdf(plotter)
            ring_no = _mean_density_on_boundary_ring(pdf_no, plotter, ring_width=1.0)
            ring_u = _mean_density_on_boundary_ring(pdf_u, plotter, ring_width=1.0)

            # Shell diagnostics: unnormalized mean density on outside boundary points.
            shell_no = mean_unnormalized_density_on_points(pnn_no, boundary_pts)
            shell_u = mean_unnormalized_density_on_points(pnn_u, boundary_pts)

            rows.append(
                {
                    "mixture": float(mixture_idx),
                    "seed": float(seed),
                    "h1": float(h1),
                    "epochs": float(epochs),
                    "val_nll_no_uniform": float(val_nll_no),
                    "val_nll_with_uniform": float(val_nll_u),
                    "delta_val_nll": float(val_nll_u - val_nll_no),
                    "boundary_ring_mean_no_uniform": float(ring_no),
                    "boundary_ring_mean_with_uniform": float(ring_u),
                    "shell_mean_unnorm_no_uniform": float(shell_no),
                    "shell_mean_unnorm_with_uniform": float(shell_u),
                }
            )

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    csv_path = os.path.join("results", "uniform_supervision_ablation_multiseed.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mixture",
                "seed",
                "h1",
                "epochs",
                "val_nll_no_uniform",
                "val_nll_with_uniform",
                "delta_val_nll",
                "boundary_ring_mean_no_uniform",
                "boundary_ring_mean_with_uniform",
                "shell_mean_unnorm_no_uniform",
                "shell_mean_unnorm_with_uniform",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    # Plot delta NLL per mixture with mean±std across seeds.
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for mixture_idx in [1, 2, 3]:
        vals = np.array([r["delta_val_nll"] for r in rows if int(r["mixture"]) == mixture_idx], dtype=float)
        if vals.size == 0:
            continue
        ax.errorbar(
            [mixture_idx],
            [float(np.mean(vals))],
            yerr=[float(np.std(vals))],
            fmt="o",
            capsize=5,
            label=f"Mixture {mixture_idx}",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Uniform supervision ablation (multi-seed): Δ ValNLL (with − without)")
    ax.set_xlabel("Mixture")
    ax.set_ylabel("Δ ValNLL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_path = os.path.join("figures", "uniform_supervision_ablation_multiseed.jpeg")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved uniform supervision ablation CSV: {csv_path}")
    print(f"Saved uniform supervision ablation figure: {fig_path}")
    return csv_path, fig_path


def run_boundary_penalty_lambda_sweep_multiseed(
    *,
    seeds: list[int] | None = None,
    lambdas: list[float] | None = None,
    h1_by_mixture: dict[int, float] | None = None,
    epochs: int = 800,
    learning_rate: float = 5e-3,
    num_uniform_points_ratio: float = 1.0,
    plot_grid: int = 100,
) -> tuple[str, str]:
    """Visualize boundary penalty lambda sweep from cached PNN models.

    Creates per-mixture figures showing:
    - Top row: 3D surface plots (h1 vs lambda vs ValNLL) for each architecture
    - Bottom row: 3D overlays comparing best λ=0 vs best λ≠0 PNN estimates

    Outputs:
      - results/boundary_penalty_lambda_sweep_cache.csv
      - figures/boundary_penalty_lambda_sweep_mixture{1,2,3}.jpeg
    """
    mixtures = _make_default_mixtures()
    plotter = Plotter(-5, 5, -5, 5, int(plot_grid))
    
    # Expected cache structure from main sweep
    bandwidths = [2, 7, 12, 16]
    lambda_values = [0.0, 0.01, 0.1]
    
    pnn_architectures = [
        {"hidden_layers": [20], "out": "sigmoid", "A": "auto"},
        {"hidden_layers": [30, 20], "out": "sigmoid", "A": "auto"},
        {"hidden_layers": [30, 20], "out": "relu", "A": "auto"},
        {"hidden_layers": [20], "out": "relu", "A": "auto"},
    ]
    
    def arch_label_fn(cfg: dict) -> str:
        layers = "-".join(str(x) for x in cfg["hidden_layers"])
        if cfg["out"] == "relu":
            return f"MLP_{layers}_sigmoid_outReLU"
        A = cfg.get("A", "auto")
        if isinstance(A, str):
            return f"MLP_{layers}_sigmoid_outSigmoid_Aauto"
        return f"MLP_{layers}_sigmoid_outSigmoid_A{float(A):.2g}"
    
    arch_labels = [arch_label_fn(cfg) for cfg in pnn_architectures]
    n_arch = len(pnn_architectures)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # CSV rows for summary
    csv_rows: list[dict] = []
    
    for mixture_idx, mixture in enumerate(mixtures):
        print(f"Generating boundary penalty lambda sweep visualization for mixture {mixture_idx + 1}")
        
        # Load cache
        cache_path = _pnn_sweep_cache_path(mixture_idx=mixture_idx)
        if not os.path.exists(cache_path):
            print(f"Warning: Cache not found for mixture {mixture_idx + 1}: {cache_path}")
            continue
        
        cache = _torch_load_cache_trusted(cache_path)
        pnn_block = cache.get("pnn", {})
        
        # Extract data
        train_xy = _to_numpy_array(cache.get("data", {}).get("train_xy"))
        val_xy = _to_numpy_array(cache.get("data", {}).get("val_xy"))
        
        # Aggregate ValNLL data: mean across iterations for each (arch, h1, lambda)
        # Structure: valnll_grid[cfg_idx][i_h][i_lam] = mean ValNLL
        valnll_grid = np.full((n_arch, len(bandwidths), len(lambda_values)), np.nan)
        
        for cfg_idx in range(n_arch):
            for i_h, h1 in enumerate(bandwidths):
                for i_lam, lam in enumerate(lambda_values):
                    lam_key = str(float(lam))
                    runs_by_lambda = pnn_block.get("runs_by_lambda", {})
                    runs_here = runs_by_lambda.get(lam_key)
                    
                    if runs_here is not None and cfg_idx < len(runs_here) and i_h < len(runs_here[cfg_idx]):
                        runs_list = runs_here[cfg_idx][i_h]
                        if len(runs_list) > 0:
                            nlls = [float(r.get("val_nll", np.nan)) for r in runs_list]
                            nlls = [x for x in nlls if np.isfinite(x)]
                            if len(nlls) > 0:
                                valnll_grid[cfg_idx, i_h, i_lam] = float(np.mean(nlls))
                                
                                # Add to CSV
                                csv_rows.append({
                                    "mixture": int(mixture_idx + 1),
                                    "architecture": str(arch_labels[cfg_idx]),
                                    "h1": float(h1),
                                    "lambda": float(lam),
                                    "mean_val_nll": float(np.mean(nlls)),
                                    "std_val_nll": float(np.std(nlls)),
                                    "n_runs": int(len(nlls)),
                                })
        
        # Create figure: top row = surface plots, bottom row = overlays
        fig = plt.figure(figsize=(6 * n_arch, 12))
        fig.suptitle(
            f"Boundary penalty λ sweep — Mixture {mixture_idx + 1}\n"
            "Top: ValNLL surface (h₁ vs λ, mean across iterations) | "
            "Bottom: Best λ=0 vs best λ PNN overlays (selected by ValNLL)",
            fontsize=12
        )
        # Add a bit more vertical spacing between rows to avoid collisions with titles
        fig.subplots_adjust(hspace=0.35)
        
        # Top row: 3D surface plots (h1 vs lambda vs ValNLL)
        for cfg_idx in range(n_arch):
            ax_surf = fig.add_subplot(2, n_arch, cfg_idx + 1, projection='3d')            
            # Prepare meshgrid
            H1, LAM = np.meshgrid(bandwidths, lambda_values)
            Z = valnll_grid[cfg_idx, :, :].T  # Transpose to match meshgrid layout
            
            # Plot surface
            surf = ax_surf.plot_surface(H1, LAM, Z, cmap='coolwarm', alpha=0.8, edgecolor='k', linewidth=0.5)
            
            ax_surf.set_xlabel("h₁", fontsize=9, labelpad=8)
            ax_surf.set_ylabel("λ", fontsize=9, labelpad=8)
            ax_surf.set_zlabel("ValNLL\n(mean across seeds)", fontsize=8, labelpad=6)
            ax_surf.set_title(
                f"{arch_labels[cfg_idx]}\nValNLL surface over (h₁, λ)\nmean across 10 training runs per config",
                fontsize=9,
                pad=1
            )
            ax_surf.view_init(elev=35, azim=60)
            fig.colorbar(surf, ax=ax_surf, shrink=0.5, aspect=10)
        
        # Bottom row: Overlays (best λ=0 vs best λ≠0)
        for cfg_idx in range(n_arch):
            ax_overlay = fig.add_subplot(2, n_arch, n_arch + cfg_idx + 1, projection='3d')
            
            # Find best (h1, λ=0) by minimum ValNLL
            lam0_idx = lambda_values.index(0.0)
            lam0_nlls = valnll_grid[cfg_idx, :, lam0_idx]
            valid_mask_lam0 = np.isfinite(lam0_nlls)
            
            if not np.any(valid_mask_lam0):
                ax_overlay.text(0.5, 0.5, 0.5, "No valid λ=0 data", ha='center', va='center')
                ax_overlay.set_title(f"{arch_labels[cfg_idx]}\nNo data available", fontsize=9)
                continue
            
            best_i_h_lam0 = int(np.nanargmin(lam0_nlls))
            best_h1_lam0 = float(bandwidths[best_i_h_lam0])
            best_nll_lam0 = float(lam0_nlls[best_i_h_lam0])
            
            # Find best (h1, λ) overall by minimum ValNLL
            all_nlls = valnll_grid[cfg_idx, :, :].ravel()
            valid_mask_all = np.isfinite(all_nlls)
            
            if not np.any(valid_mask_all):
                ax_overlay.text(0.5, 0.5, 0.5, "No valid data", ha='center', va='center')
                ax_overlay.set_title(f"{arch_labels[cfg_idx]}\nNo data available", fontsize=9)
                continue
            
            best_flat_idx = int(np.nanargmin(all_nlls))
            best_i_h_any, best_i_lam_any = np.unravel_index(best_flat_idx, valnll_grid[cfg_idx, :, :].shape)
            best_h1_any = float(bandwidths[best_i_h_any])
            best_lam_any = float(lambda_values[best_i_lam_any])
            best_nll_any = float(valnll_grid[cfg_idx, best_i_h_any, best_i_lam_any])
            
            # Reconstruct PNN models from cache
            try:
                # Best λ=0 model
                run_lam0 = _pnn_best_run_from_cache_for(
                    cache, cfg_idx=cfg_idx, i_h=best_i_h_lam0, lambda_boundary=0.0
                )
                pnn_lam0 = _pnn_instantiate_from_cache(run_lam0.get("meta", {}))
                pnn_lam0.load_state_dict(run_lam0.get("state_dict"))
                pdf_lam0 = pnn_lam0.estimate_pdf(plotter)
                
                # Best λ≠0 model (could be λ=0 if that's overall best)
                run_any = _pnn_best_run_from_cache_for(
                    cache, cfg_idx=cfg_idx, i_h=best_i_h_any, lambda_boundary=best_lam_any
                )
                pnn_any = _pnn_instantiate_from_cache(run_any.get("meta", {}))
                pnn_any.load_state_dict(run_any.get("state_dict"))
                pdf_any = pnn_any.estimate_pdf(plotter)
                
                # Plot overlays with low opacity
                ax_overlay.plot_surface(
                    plotter.X, plotter.Y, pdf_lam0,
                    cmap='plasma', alpha=0.5, label=f'λ=0'
                )
                ax_overlay.plot_surface(
                    plotter.X, plotter.Y, pdf_any,
                    cmap='viridis', alpha=0.5, label=f'λ={best_lam_any:g}'
                )
                
                ax_overlay.set_xlabel("x", fontsize=8)
                ax_overlay.set_ylabel("y", fontsize=8)
                ax_overlay.set_zlabel("pdf", fontsize=8, labelpad=8)
                ax_overlay.set_title(
                    f"{arch_labels[cfg_idx]}\n"
                    f"Plasma: λ=0 at h₁={best_h1_lam0:g} (ValNLL={best_nll_lam0:.3g})\n"
                    f"Viridis: λ={best_lam_any:g} at h₁={best_h1_any:g} (ValNLL={best_nll_any:.3g})",
                    fontsize=9,
                    pad=2
                )
                ax_overlay.view_init(elev=25, azim=215)
                
            except Exception as e:
                ax_overlay.text(0.5, 0.5, 0.5, f"Error: {str(e)[:50]}", ha='center', va='center', fontsize=8)
                ax_overlay.set_title(f"{arch_labels[cfg_idx]}\nReconstruction failed", fontsize=9)
        
        # Save figure
        fig_path = os.path.join("figures", f"boundary_penalty_lambda_sweep_mixture{mixture_idx + 1}.jpeg")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved boundary penalty lambda sweep figure: {fig_path}")
    
    # Save CSV
    csv_path = os.path.join("results", "boundary_penalty_lambda_sweep_cache.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if len(csv_rows) > 0:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        else:
            f.write("mixture,architecture,h1,lambda,mean_val_nll,std_val_nll,n_runs\n")
    
    print(f"Saved boundary penalty lambda sweep CSV: {csv_path}")
    return csv_path, fig_path


def _make_default_mixtures() -> list["GaussianMixture"]:
    """Construct Mixtures 1..3 exactly as in main()."""
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
    return [mixture1, mixture2, mixture3]


def _pnn_best_run_from_cache_for(
    cache: dict,
    *,
    cfg_idx: int,
    i_h: int,
    lambda_boundary: float | None = None,
) -> dict:
    """Return the cached run entry for the best run at (architecture, bandwidth index).

    Prefers the explicit 'selected' index (fast). Falls back to scanning runs.
    """
    pnn_block = cache.get("pnn", {})
    lam = 0.0 if lambda_boundary is None else float(lambda_boundary)

    runs = None
    selected = None
    if "runs_by_lambda" in pnn_block:
        runs_by_lambda = pnn_block.get("runs_by_lambda", {})
        selected_by_lambda = pnn_block.get("selected_by_lambda", {})
        runs = runs_by_lambda.get(str(lam))
        selected = selected_by_lambda.get(str(lam))

    if runs is None:
        # Backward compatibility: lambda=0 stored in the legacy fields.
        runs = pnn_block.get("runs", [])
        selected = pnn_block.get("selected")
    if cfg_idx >= len(runs) or i_h >= len(runs[cfg_idx]):
        raise KeyError("Cache missing PNN runs for requested indices")
    runs_list = runs[cfg_idx][i_h]
    if len(runs_list) == 0:
        raise KeyError("Cache has empty runs list for requested indices")

    if isinstance(selected, list) and cfg_idx < len(selected) and i_h < len(selected[cfg_idx]):
        sel_entry = selected[cfg_idx][i_h]
        if isinstance(sel_entry, dict) and "best_run_index" in sel_entry:
            best_idx = int(sel_entry["best_run_index"])
            if 0 <= best_idx < len(runs_list):
                return runs_list[best_idx]

    # Fallback: choose by minimum val_nll.
    vals = [float(r.get("val_nll", float("inf"))) for r in runs_list]
    best_idx = int(np.nanargmin(np.asarray(vals, dtype=float)))
    return runs_list[best_idx]


def _plot_best_over_h1_overlays_for_mixture(
    *,
    mixture_idx: int,
    mixture: "GaussianMixture",
    plotter: "Plotter",
    train_xy: np.ndarray,
    val_xy: np.ndarray,
    bandwidths: list[float] | list[int],
    pnn_architectures: list[dict],
    arch_label_fn,
    per_arch_val_nll_by_h: list[list[float]],
    cache_for_reconstruct: dict,
) -> str:
    """Create one overlay figure per mixture, independent of h1.

    Each column (architecture) uses its own best h1 (min PNN ValNLL over h1), and the
    two rows show KDE-vs-truth and PNN-vs-truth at that same h1.
    """
    n_arch = len(pnn_architectures)
    fig_best = plt.figure(figsize=(5 * n_arch, 10))
    fig_best.suptitle(
        "Best surface overlays\n"
        f"Mixture {mixture_idx + 1} — each column uses its own PNN architecture and h_1 that minimizes ValNLL",
        fontsize=11,
    )

    arch_labels_all = [arch_label_fn(cfg) for cfg in pnn_architectures]
    real_pdf = mixture.get_mesh(plotter.pos)

    for j, label in enumerate(arch_labels_all):
        nlls = per_arch_val_nll_by_h[j] if j < len(per_arch_val_nll_by_h) else []
        if not isinstance(nlls, list) or len(nlls) != len(bandwidths):
            best_i_h = 0
        else:
            best_i_h = int(np.nanargmin(np.asarray(nlls, dtype=float)))
        best_h1 = float(bandwidths[best_i_h])

        # KDE at the same best_h1 (within-column comparability).
        kde_estimator = ParzenWindowEstimator(train_xy, window_size=float(best_h1))
        estimated_pdf_kde = kde_estimator.estimate_pdf(plotter)
        kde_val_ll = average_log_likelihood_kde(val_xy, train_xy, float(best_h1))
        kde_val_nll = _nll_from_avg_loglik(kde_val_ll)

        # PNN surface reconstructed from cached state_dict (no retraining).
        run_entry = _pnn_best_run_from_cache_for(cache_for_reconstruct, cfg_idx=j, i_h=best_i_h)
        meta = run_entry.get("meta", {})
        pnn = _pnn_instantiate_from_cache(meta)
        sd = run_entry.get("state_dict")
        if sd is None:
            raise RuntimeError("Cache entry missing state_dict")
        pnn.load_state_dict(sd)
        pnn_surface = pnn.estimate_pdf(plotter)
        pnn_val_nll = float(run_entry.get("val_nll", float("nan")))

        # Row 1: KDE vs truth
        ax1 = fig_best.add_subplot(2, n_arch, j + 1, projection='3d')
        ax1.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
        ax1.plot_surface(plotter.X, plotter.Y, estimated_pdf_kde, alpha=0.45, cmap='cividis')
        ax1.set_title(f"{label}\nKDE vs truth @ h_1={best_h1:.3g} (ValNLL={kde_val_nll:.3g})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("pdf")

        # Row 2: PNN vs truth
        ax2 = fig_best.add_subplot(2, n_arch, n_arch + j + 1, projection='3d')
        ax2.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
        ax2.plot_surface(plotter.X, plotter.Y, pnn_surface, alpha=0.45, cmap='plasma')
        ax2.set_title(
            f"{label}\n"
            f"PNN vs truth @ h_1={best_h1:.3g} (best-by-ValNLL over h_1;\nValNLL={pnn_val_nll:.3g})"
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("pdf")

    out_path = f"figures/overlays_best_mixture{mixture_idx + 1}.jpeg"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig_best)
    return out_path


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


def _points_inside_convex_hull(points_xy: np.ndarray, hull: ConvexHull, *, tol: float = 1e-12) -> np.ndarray:
    """Return boolean mask for points inside (or on) the convex hull.

    Uses the half-space representation from QHull: for each facet, the hull stores
    an equation a*x + b*y + c = 0 with the *inside* satisfying <= 0.
    """
    P = np.asarray(points_xy, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_xy must have shape (m, 2)")
    A = np.asarray(hull.equations[:, :2], dtype=float)
    b = np.asarray(hull.equations[:, 2], dtype=float)
    vals = P @ A.T + b[None, :]
    return np.all(vals <= float(tol), axis=1)


def sample_boundary_points_outside_convex_hull(points_xy: np.ndarray, alpha: float, k: int) -> np.ndarray:
    r"""Sample k points outside the convex hull of a 2D point set.

    We sample uniformly from a padded axis-aligned bounding box around the sampled
    points, then reject candidate points that lie inside the convex hull.

    If the hull is degenerate (too few points / collinear), we fall back to sampling
    outside the axis-aligned bounding box of the points (shell outside the box).
    """
    pts_xy = np.asarray(points_xy, dtype=float)
    if pts_xy.ndim != 2 or pts_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    n = int(pts_xy.shape[0])
    if n < 1:
        return np.zeros((0, 2), dtype=float)

    min_x = float(np.min(pts_xy[:, 0]))
    max_x = float(np.max(pts_xy[:, 0]))
    min_y = float(np.min(pts_xy[:, 1]))
    max_y = float(np.max(pts_xy[:, 1]))
    diameter = float(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2))
    delta = float(alpha) * (diameter if diameter > 0 else 1.0)

    pad_min_x = min_x - delta
    pad_max_x = max_x + delta
    pad_min_y = min_y - delta
    pad_max_y = max_y + delta

    hull = None
    if n >= 3:
        try:
            hull = ConvexHull(pts_xy)
        except QhullError:
            hull = None
        except Exception:
            hull = None

    out: list[list[float]] = []
    attempts = 0
    max_attempts = max(20000, 20 * int(k))

    # Degenerate hull: sample from the shell outside the tight bounding box.
    if hull is None:
        while len(out) < int(k) and attempts < max_attempts:
            x = float(np.random.uniform(pad_min_x, pad_max_x))
            y = float(np.random.uniform(pad_min_y, pad_max_y))
            inside_box = (min_x <= x <= max_x) and (min_y <= y <= max_y)
            if not inside_box:
                out.append([x, y])
            attempts += 1
        return np.asarray(out, dtype=float)

    # Proper hull: reject points inside the convex hull.
    while len(out) < int(k) and attempts < max_attempts:
        x = float(np.random.uniform(pad_min_x, pad_max_x))
        y = float(np.random.uniform(pad_min_y, pad_max_y))
        cand = np.asarray([[x, y]], dtype=float)
        inside = bool(_points_inside_convex_hull(cand, hull, tol=1e-12)[0])
        if not inside:
            out.append([x, y])
        attempts += 1
    return np.asarray(out, dtype=float)

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
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
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
        checkpoint_metric: str = "train_loss",
    ):
        """Train by regression on leave-one-out Parzen targets.

        loss_mode:
          - "mse": (f(x_i) - y_i)^2
          - "relative": ((f - y)/(y+eps))^2
          - "log": (log(f+eps) - log(y+eps))^2

                checkpoint_metric:
                    - "train_loss": selects best checkpoint by the training objective (data-only)
                    - "eval_mse": selects best checkpoint by grid MSE vs ground-truth mixture (oracle; diagnostics only)
        """
        points_xy = np.asarray(points)[:, :2].astype(float)
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

        checkpoint_metric = str(checkpoint_metric).lower().strip()
        if checkpoint_metric not in ("train_loss", "eval_mse"):
            raise ValueError("checkpoint_metric must be 'train_loss' or 'eval_mse'")

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

            # Optional: track evaluation MSE on grid vs ground truth (oracle) for plotting.
            eval_mse = None
            if mixture is not None:
                est_pdf = self.estimate_pdf(plotter)
                true_pdf = mixture.get_mesh(plotter.pos)
                eval_mse = float(np.mean((est_pdf - true_pdf) ** 2))
                eval_mse_history.append(eval_mse)

            # Checkpoint selection (can be data-only or oracle, depending on flag).
            if checkpoint_metric == "eval_mse" and eval_mse is not None:
                metric = float(eval_mse)
            else:
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
    parser.add_argument("--pnn-only", action="store_true", help="Run only Parzen Neural Network (skip Parzen Window)")
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="PNN: skip training and load cached sweep models/results (re-generate figures/JSON only)",
    )
    parser.add_argument(
        "--best-overlays-only",
        action="store_true",
        help="PNN: generate per-mixture overlays_best_mixture*.jpeg from cached sweep and exit (fast)",
    )
    parser.add_argument("--pw-seed", type=int, default=0, help="Seed for PW-only sweep reproducibility")
    parser.add_argument(
        "--pw-write-logs",
        action="store_true",
        help="Write PW error logs (CSV + SUMMARY lines) for each (n, h1)",
    )
    # Professor-review helpers (lightweight experiments / diagnostics)
    parser.add_argument(
        "--review-assets",
        action="store_true",
        help="Generate extra review assets (target range + uniform ablation + boundary sweep) and exit",
    )
    parser.add_argument(
        "--review-epochs",
        type=int,
        default=800,
        help="Epochs for review ablation trainings (kept small by default)",
    )
    parser.add_argument(
        "--review-grid",
        type=int,
        default=100,
        help="Grid resolution for review assets (matches report default at 100)",
    )
    args, _unknown = parser.parse_known_args()

    if bool(args.pw_only) and bool(args.pnn_only):
        raise SystemExit("Choose at most one of --pw-only or --pnn-only")

    if bool(args.review_assets):
        benchmark_loocv_target_dynamic_range(mixture_index=2, h1=12.0)
        run_uniform_supervision_ablation_multiseed(epochs=int(args.review_epochs), plot_grid=int(args.review_grid))
        run_boundary_penalty_lambda_sweep_multiseed(epochs=int(args.review_epochs), plot_grid=int(args.review_grid))
        return

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
        if bool(args.pnn_only):
            break

        print(f"Processing Gaussian Mixture {mixture_idx + 1} with Parzen Window")
        grid_errors = []   # grid MSE
        nll_errors = []       # validation NLL
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

        # Store the *actual* grid estimates corresponding to the selected best configs.
        # Otherwise, overlays could be generated from a fresh re-sample and not match the
        # reported MSE/NLL values from the sweep.
        best_pdf_by_val_nll: np.ndarray | None = None
        best_pdf_by_grid_mse: np.ndarray | None = None

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
                est_grid = parzen_estimator.estimate_pdf(plotter)
                true_grid = mixture.get_mesh(plotter.pos)
                err = parzen_estimator.calculate_error_metrics(true_grid)
                grid_mse = float(err["Mean Squared Error"])

                ll_val = average_log_likelihood_kde(val_xy, train_xy, h1)
                val_nll = _nll_from_avg_loglik(ll_val)

                grid_errors.append(grid_mse)
                nll_errors.append(val_nll)
                sampled_points.append(num_samples)
                sampled_window.append(h1)

                # Track best configurations.
                if np.isfinite(val_nll) and val_nll < best_by_val_nll[0]:
                    best_by_val_nll = (float(val_nll), int(num_samples), float(h1))
                    best_pdf_by_val_nll = np.asarray(est_grid, dtype=float).copy()
                if np.isfinite(grid_mse) and grid_mse < best_by_grid_mse[0]:
                    best_by_grid_mse = (float(grid_mse), int(num_samples), float(h1))
                    best_pdf_by_grid_mse = np.asarray(est_grid, dtype=float).copy()

                # Optional log export.
                if csv_writer is not None and csv_fh is not None and txt_fh is not None:
                    csv_writer.writerow(
                        {
                            "mixture": float(mixture_idx + 1),
                            "n": int(num_samples),
                            "h1": float(h1),
                            "h_n": float(_effective_bandwidth(float(h1), int(train_xy.shape[0]))),
                            "grid_mse": float(grid_mse),
                            "grid_rmse": float(np.sqrt(grid_mse)),
                            "grid_max_abs_err": float(err["Max Absolute Error"]),
                            "grid_mean_abs_err": float(err["Mean Absolute Error"]),
                            "val_avg_nll": float(val_nll),
                        }
                    )
                    txt_fh.write(
                        f"SUMMARY: mixture={mixture_idx+1}, n={int(num_samples)}, h1={h1}, "
                        f"h_n={_effective_bandwidth(float(h1), int(train_xy.shape[0]))}, "
                        f"grid_mse={grid_mse:.6e}, val_avg_nll={val_nll:.6e}\n"
                    )
                    csv_fh.flush()
                    txt_fh.flush()

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(projection='3d')
        sampled_points_arr = np.array(sampled_points)
        sampled_window_arr = np.array(sampled_window)
        grid_errors_arr = np.array(grid_errors, dtype=float)
        nll_errors_arr = np.array(nll_errors, dtype=float)

        # Normalize both metrics independently to [0, 1] so they share
        # the same visual z-range while preserving their internal ordering.
        def _normalize(arr: np.ndarray) -> np.ndarray:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.zeros_like(arr, dtype=float)
            amin = float(np.min(finite))
            amax = float(np.max(finite))
            if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
                return np.zeros_like(arr, dtype=float)
            return (arr - amin) / (amax - amin)

        grid_errors_norm = _normalize(grid_errors_arr)
        nll_errors_norm = _normalize(nll_errors_arr)

        # Plot normalized values on a single shared z-axis
        sc_mse = ax.scatter(
            sampled_points_arr,
            sampled_window_arr,
            grid_errors_norm,
            c='r',
            alpha=0.2,
            s=30,
            label='Grid MSE (normalized)',
        )
        sc_nll = ax.scatter(
            sampled_points_arr,
            sampled_window_arr,
            nll_errors_norm,
            c='b',
            alpha=0.2,
            s=30,
            label='Val NLL (normalized)',
        )

        # Annotate and highlight un-normalized min/max values at the corresponding plotted points.
        def _highlight_and_label(orig_arr, norm_arr, color, label_prefix):
            try:
                finite_mask = np.isfinite(orig_arr)
                if not np.any(finite_mask):
                    return
                idx_min = int(np.nanargmin(orig_arr))
                idx_max = int(np.nanargmax(orig_arr))

                for idx, which in ((idx_min, 'min'), (idx_max, 'max')):
                    x_pt = float(sampled_points_arr[idx])
                    y_pt = float(sampled_window_arr[idx])
                    z_pt = float(norm_arr[idx])

                    # Make highlighted point larger with a thick contrasting border
                    marker_size = 220 if which == 'min' else 180
                    ax.scatter(
                        [x_pt], [y_pt], [z_pt],
                        s=marker_size,
                        facecolors=color,
                        edgecolors='k',
                        linewidths=2.2,
                        marker='o',
                        zorder=20,
                    )

                    # Place the label above the point in normalized z coordinates and offset in x for visibility
                    text_z = z_pt + 0.18
                    text_z = max(-0.06, min(1.12, text_z))
                    text_x = x_pt + max(1.0, 0.02 * max(1.0, abs(x_pt)))

                    # Draw a connector line from label to point
                    ax.plot([x_pt, text_x], [y_pt, y_pt], [z_pt, text_z], color=color, linewidth=1.0, zorder=19)

                    txt = (
                        f"{label_prefix} {which}={orig_arr[idx]:.3e}\n"
                        f"(n={int(x_pt)}, h1={y_pt:.3g})"
                    )
                    ax.text(
                        text_x,
                        y_pt,
                        text_z,
                        txt,
                        color=color,
                        fontsize=9,
                        bbox={'facecolor': 'white', 'alpha': 0.95, 'edgecolor': color, 'pad': 0.4},
                        zorder=21,
                    )
            except Exception:
                pass

        _highlight_and_label(grid_errors_arr, grid_errors_norm, 'r', 'GridMSE')
        _highlight_and_label(nll_errors_arr, nll_errors_norm, 'b', 'ValNLL')

        # Reset view to keep the data centered and readable
        ax.view_init(elev=30, azim=105)

        ax.set_title(f"Parzen Window Errors (Mixture {mixture_idx + 1}) — normalized z")
        ax.set_xlabel("Samples per Gaussian (n)")
        ax.set_ylabel("Base bandwidth h1")
        ax.set_zlabel("Normalized error (per-metric)")

        # Slightly zoom out on normalized z-axis so labels remain fully visible
        zpad = 0.3
        ax.set_zlim(-zpad, 1.0 + zpad)

        # Keep legend close to the plot
        ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(0.98, 0.95))

        # Increase margins to avoid cut labels.
        fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08)
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

        # Build TWO overlays:
        #  - best by validation NLL (data-only selection)
        #  - best by grid MSE (oracle selection)
        idx_fallback = int(np.argmin(grid_errors)) if len(grid_errors) else 0
        n_nll = int(best_by_val_nll[1]) if best_by_val_nll[1] is not None else int(sampled_points[idx_fallback])
        h_nll = float(best_by_val_nll[2]) if best_by_val_nll[2] is not None else float(sampled_window[idx_fallback])
        n_mse = int(best_by_grid_mse[1]) if best_by_grid_mse[1] is not None else int(sampled_points[idx_fallback])
        h_mse = float(best_by_grid_mse[2]) if best_by_grid_mse[2] is not None else float(sampled_window[idx_fallback])

        # Prefer the exact grids from the sweep; fall back only if they are missing.
        if best_pdf_by_val_nll is None or best_pdf_by_grid_mse is None:
            def _estimate_kde_pdf_for(n_samples: int, h1: float) -> np.ndarray:
                # Keep deterministic behavior in --pw-only mode.
                if bool(args.pw_only):
                    # Mix in both n and h1 so the two overlays don't accidentally reuse identical seeds.
                    h_key = int(round(1000.0 * float(h1)))
                    np.random.seed(int(args.pw_seed) + 1000 * int(mixture_idx + 1) + int(n_samples) + 10_000 * int(h_key))
                samples_xy = mixture.sample_points(int(n_samples), with_pdf=False)
                train_xy, _val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=(mixture_idx + 1))
                kde = ParzenWindowEstimator(train_xy, float(h1))
                return kde.estimate_pdf(plotter)

            estimated_pdf_nll = best_pdf_by_val_nll if best_pdf_by_val_nll is not None else _estimate_kde_pdf_for(n_nll, h_nll)
            estimated_pdf_mse = best_pdf_by_grid_mse if best_pdf_by_grid_mse is not None else _estimate_kde_pdf_for(n_mse, h_mse)
        else:
            estimated_pdf_nll = best_pdf_by_val_nll
            estimated_pdf_mse = best_pdf_by_grid_mse
        real_pdf = mixture.get_mesh(plotter.pos)

        cmap_true = 'viridis'
        cmap_overlay = 'plasma'  # keep the existing overlay cmap
        alpha_true_only = 0.8
        alpha_true_overlay = 0.10
        alpha_overlay = 0.80  # lower opacity so the real PDF below remains visible

        fig_overlay = plt.figure(figsize=(18, 12))
        fig_overlay.suptitle(f"Parzen Window overlays — mixture {mixture_idx+1}")

        # Layout:
        #   Columns: left=NLL-selected, right=MSE-selected
        #   Rows:    top=real mixture only, bottom=overlay (real + KDE estimate)
        ax_true_nll = fig_overlay.add_subplot(2, 2, 1, projection='3d')
        ax_true_mse = fig_overlay.add_subplot(2, 2, 2, projection='3d')
        ax_overlay_nll = fig_overlay.add_subplot(2, 2, 3, projection='3d')
        ax_overlay_mse = fig_overlay.add_subplot(2, 2, 4, projection='3d')

        # Top row: ground truth (same surface, but titles reflect the selection column).
        ax_true_nll.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=alpha_true_only, cmap=cmap_true)
        ax_true_nll.set_title(
            "Real mixture PDF (ground truth)\n"
            f"NLL-selected setup: Samples/gaussian = {n_nll}, Window size = {h_nll:.6g}"
        )
        ax_true_mse.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=alpha_true_only, cmap=cmap_true)
        ax_true_mse.set_title(
            "Real mixture PDF (ground truth)\n"
            f"MSE-selected setup: Samples/gaussian = {n_mse}, Window size = {h_mse:.6g}"
        )

        # Bottom row: overlays.
        ax_overlay_nll.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=alpha_true_overlay, cmap=cmap_true)
        ax_overlay_nll.plot_surface(plotter.X, plotter.Y, estimated_pdf_nll, alpha=alpha_overlay, cmap=cmap_overlay)
        ax_overlay_nll.set_title(
            "Overlay: real PDF + KDE estimate\n"
            f"Selected by validation NLL (data-only): Samples/gaussian = {n_nll}, Window size = {h_nll:.6g}"
        )

        ax_overlay_mse.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=alpha_true_overlay, cmap=cmap_true)
        ax_overlay_mse.plot_surface(plotter.X, plotter.Y, estimated_pdf_mse, alpha=alpha_overlay, cmap=cmap_overlay)
        ax_overlay_mse.set_title(
            "Overlay: real PDF + KDE estimate\n"
            f"Selected by grid MSE (oracle): Samples/gaussian = {n_mse}, Window size = {h_mse:.6g}"
        )

        # Common axis labels + view/zoom adjustments.
        all_axes = [ax_true_nll, ax_true_mse, ax_overlay_nll, ax_overlay_mse]
        for ax_i, ax in enumerate(all_axes):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # Extra padding on the right column to avoid clipped z-labels.
            zpad = 18 if (ax_i % 2 == 1) else 10
            ax.set_zlabel("pdf", labelpad=zpad)
            ax.view_init(elev=28, azim=215)
            # Zoom out slightly so labels fit better (matplotlib may deprecate this, so guard it).
            try:
                ax.dist = 12
            except Exception:
                pass

        # Tight-layout is unreliable for 3D; use manual margins so right-column labels stay visible.
        fig_overlay.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.04, wspace=0.08, hspace=0.18)

        overlay_fig_filename = f"figures/Parzen_overlay_mixture{mixture_idx + 1}.jpeg"
        plt.savefig(overlay_fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.2)
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

        cache_path = _pnn_sweep_cache_path(mixture_idx=mixture_idx)
        cache_loaded: dict | None = None

        if os.path.exists(cache_path):
            # Cache exists: either reuse it (and optionally extend it by training missing lambdas),
            # or delete it if the user explicitly requests a full retrain.
            if (not bool(args.no_learn)) and (sys.stdin is not None) and sys.stdin.isatty():
                resp = input(
                    f"PNN cache already exists for mixture {mixture_idx+1}: {cache_path}\n"
                    "Type YES to delete it and retrain everything from scratch.\n"
                    "Type anything else to reuse it (and train only missing lambdas, if any): "
                ).strip()
                if resp == "YES":
                    try:
                        os.remove(cache_path)
                        print(f"Deleted cache: {cache_path}")
                    except OSError as e:
                        raise SystemExit(f"Failed to delete cache {cache_path}: {e}")

            if os.path.exists(cache_path):
                cache_loaded = _torch_load_cache_trusted(cache_path)

        if bool(args.no_learn) and cache_loaded is None:
            raise SystemExit(
                f"Cache not found: {cache_path}. "
                "Run once without --no-learn to generate it."
            )

        if bool(args.best_overlays_only):
            if cache_loaded is None:
                raise SystemExit("--best-overlays-only requires --no-learn (cached sweep)")

        # Define hyperparameters
        learning_rates = [5e-3]
        bandwidths = [2, 7, 12, 16]
        use_log_density = True
        lambda_sweep = [0.0, 0.01, 0.1]
        boundary_support_mode = "convex_hull"

        if cache_loaded is not None:
            try:
                ver = int(cache_loaded.get("version", 0)) if isinstance(cache_loaded, dict) else 0
                cfg0 = cache_loaded.get("config", {})
                cached_bandwidths = list(cfg0.get("bandwidths", []))
                cached_lrs = list(cfg0.get("learning_rates", []))
                cached_use_log = bool(cfg0.get("use_log_density", True))
                cached_support_mode = str(cfg0.get("boundary_support_mode", "")).strip().lower()
                if cached_support_mode != str(boundary_support_mode).lower():
                    raise SystemExit(
                        "Cache/config mismatch: boundary support mode differs. "
                        "Delete the cache file and re-run to regenerate boundary points."
                    )
                if [float(x) for x in cached_bandwidths] != [float(x) for x in bandwidths]:
                    raise SystemExit(
                        "Cache/config mismatch: bandwidths differ. "
                        "Delete the cache file or re-run training with matching settings."
                    )
                if [float(x) for x in cached_lrs] != [float(x) for x in learning_rates]:
                    raise SystemExit(
                        "Cache/config mismatch: learning_rates differ. "
                        "Delete the cache file or re-run training with matching settings."
                    )
                if bool(cached_use_log) != bool(use_log_density):
                    raise SystemExit(
                        "Cache/config mismatch: use_log_density differs. "
                        "Delete the cache file or re-run training with matching settings."
                    )

                # Ensure per-lambda containers exist (newer caches); otherwise, upgrade in-memory.
                pnn_block = cache_loaded.get("pnn", {})
                if isinstance(pnn_block, dict):
                    if "runs_by_lambda" not in pnn_block:
                        pnn_block["runs_by_lambda"] = {"0.0": pnn_block.get("runs", [])}
                    if "selected_by_lambda" not in pnn_block:
                        pnn_block["selected_by_lambda"] = {"0.0": pnn_block.get("selected", [])}
                    if "lambdas" not in pnn_block:
                        pnn_block["lambdas"] = [0.0]
                    cache_loaded["pnn"] = pnn_block
            except Exception as e:
                raise SystemExit(f"Failed to validate cache file {cache_path}: {e}")

        # Prepare samples (~100 per Gaussian, scaled by weight)
        if cache_loaded is not None:
            # Reuse exact cached split for reproducibility of selection/metrics.
            d = cache_loaded.get("data", {})
            train_xy = _to_numpy_array(d.get("train_xy"))
            val_xy = _to_numpy_array(d.get("val_xy"))
            boundary_pts = _to_numpy_array(d.get("boundary_pts"))
        else:
            samples_xy = mixture.sample_points_weighted(100, with_pdf=False)
            # Split train/validation so we can evaluate without ground truth (data-only metric).
            train_xy, val_xy = split_train_validation(samples_xy, val_fraction=0.2, seed=mixture_idx + 1)
            # Optional boundary points set: sample outside convex hull of all sampled points.
            boundary_pts = sample_boundary_points_outside_convex_hull(
                np.vstack([np.asarray(train_xy, dtype=float), np.asarray(val_xy, dtype=float)]),
                alpha=0.1,
                k=max(20, int(0.3 * len(train_xy))),
            )

        # We will generate an explicit figure that compares lambda_boundary=0 vs >0 at the end
        # of the sweep, selecting the configuration that is best by validation NLL.

        def _h_tag(h: float) -> str:
            # Safe for filenames
            return f"{float(h):.2f}".replace(".", "p")

        def _lam_tag(lam: float) -> str:
            return f"{float(lam):g}".replace("-", "m").replace(".", "p")

        arch_labels_all = [arch_label(cfg) for cfg in pnn_architectures]

        # Collect results across bandwidths so we can make one learning-results figure per mixture.
        n_arch_total = len(pnn_architectures)
        # per_arch_eval_mse_by_h[cfg_idx][i_h] -> list of iteration mse_hist arrays
        per_arch_eval_mse_by_h: list[list[list[list[float]]]] = [[] for _ in range(n_arch_total)]
        # per_arch_final_mse_by_h[cfg_idx][i_h] -> list of final-truth-mse per iteration
        per_arch_final_mse_by_h: list[list[list[float]]] = [[] for _ in range(n_arch_total)]
        kde_mse_by_h: list[float] = []

        # Data-only metrics (validation average log-likelihood / NLL).
        kde_val_ll_by_h: list[float] = []
        kde_val_nll_by_h: list[float] = []
        per_arch_val_ll_by_h: list[list[float]] = [[] for _ in range(n_arch_total)]
        per_arch_val_nll_by_h: list[list[float]] = [[] for _ in range(n_arch_total)]
        # NEW: keep *all* 10 ValNLLs per (arch, h1) so we can plot mean±std.
        # per_arch_val_nlls_by_h[cfg_idx][i_h] -> list of iteration ValNLL values
        per_arch_val_nlls_by_h: list[list[list[float]]] = [[[] for _ in range(len(bandwidths))] for _ in range(n_arch_total)]

        # Cache container (written at end of mixture).
        pnn_cache_payload = None
        if cache_loaded is None:
            _safe_mkdir(_pnn_cache_dir())
            pnn_cache_payload = {
                "version": 2,
                "mixture": int(mixture_idx + 1),
                "config": {
                    "bandwidths": [float(h) for h in bandwidths],
                    "learning_rates": [float(lr) for lr in learning_rates],
                    "use_log_density": bool(use_log_density),
                    "boundary_support_mode": str(boundary_support_mode),
                    "architectures": copy.deepcopy(pnn_architectures),
                },
                "data": {
                    "train_xy": torch.tensor(np.asarray(train_xy, dtype=float), dtype=torch.float32),
                    "val_xy": torch.tensor(np.asarray(val_xy, dtype=float), dtype=torch.float32),
                    "boundary_pts": torch.tensor(np.asarray(boundary_pts, dtype=float), dtype=torch.float32),
                },
                "kde": {},
                "pnn": {
                    # Legacy lambda=0 fields (still used by most plotting code).
                    "runs": [[[] for _ in range(len(bandwidths))] for _ in range(n_arch_total)],
                    "selected": [[None for _ in range(len(bandwidths))] for _ in range(n_arch_total)],
                    # New per-lambda containers for full sweep.
                    "lambdas": [float(l) for l in lambda_sweep],
                    "runs_by_lambda": {
                        str(float(lam)): [[[] for _ in range(len(bandwidths))] for _ in range(n_arch_total)]
                        for lam in lambda_sweep
                    },
                    "selected_by_lambda": {
                        str(float(lam)): [[None for _ in range(len(bandwidths))] for _ in range(n_arch_total)]
                        for lam in lambda_sweep
                    },
                },
                "boundary_penalty_demo": None,
            }
        else:
            # Restore cached aggregated metrics for consolidated plots/JSON.
            try:
                per_arch_eval_mse_by_h = cache_loaded["pnn"]["per_arch_eval_mse_by_h"]
                per_arch_final_mse_by_h = cache_loaded["pnn"]["per_arch_final_mse_by_h"]
                kde_mse_by_h = list(cache_loaded.get("kde", {}).get("grid_mse", []))
                kde_val_ll_by_h = list(cache_loaded.get("kde", {}).get("val_avg_loglik", []))
                kde_val_nll_by_h = list(cache_loaded.get("kde", {}).get("val_avg_nll", []))
                per_arch_val_ll_by_h = cache_loaded["pnn"]["per_arch_val_ll_by_h"]
                per_arch_val_nll_by_h = cache_loaded["pnn"]["per_arch_val_nll_by_h"]
                # Optional: newer caches include the full ValNLL per-iteration lists.
                per_arch_val_nlls_by_h = cache_loaded["pnn"].get("per_arch_val_nlls_by_h", per_arch_val_nlls_by_h)
            except Exception:
                # Backward compatibility: cache might not have these fields yet.
                pass

            # If we are allowed to learn, we will extend this cache in-place (train missing lambdas)
            # and then re-save it at the end.
            if not bool(args.no_learn):
                pnn_cache_payload = cache_loaded
                try:
                    pnn_block = pnn_cache_payload.get("pnn", {})
                    if "runs_by_lambda" not in pnn_block:
                        pnn_block["runs_by_lambda"] = {"0.0": pnn_block.get("runs", [])}
                    if "selected_by_lambda" not in pnn_block:
                        pnn_block["selected_by_lambda"] = {"0.0": pnn_block.get("selected", [])}
                    if "lambdas" not in pnn_block:
                        pnn_block["lambdas"] = [0.0]
                    for lam in lambda_sweep:
                        k = str(float(lam))
                        if k not in pnn_block["runs_by_lambda"]:
                            pnn_block["runs_by_lambda"][k] = [[[] for _ in range(len(bandwidths))] for _ in range(n_arch_total)]
                        if k not in pnn_block["selected_by_lambda"]:
                            pnn_block["selected_by_lambda"][k] = [[None for _ in range(len(bandwidths))] for _ in range(n_arch_total)]
                    # Keep legacy config fields coherent.
                    cfg = pnn_cache_payload.get("config", {})
                    cfg["boundary_support_mode"] = str(boundary_support_mode)
                    pnn_cache_payload["config"] = cfg
                    pnn_cache_payload["pnn"] = pnn_block
                except Exception:
                    pass

        # Fast path: generate only the best-over-h1 overlays and skip all other figures.
        if bool(args.best_overlays_only):
            try:
                per_arch_val_nll_by_h = cache_loaded["pnn"]["per_arch_val_nll_by_h"]
            except Exception as e:
                raise SystemExit(f"Cache missing per_arch_val_nll_by_h required for --best-overlays-only: {e}")
            out_path = _plot_best_over_h1_overlays_for_mixture(
                mixture_idx=mixture_idx,
                mixture=mixture,
                plotter=plotter,
                train_xy=train_xy,
                val_xy=val_xy,
                bandwidths=[float(h) for h in bandwidths],
                pnn_architectures=pnn_architectures,
                arch_label_fn=arch_label,
                per_arch_val_nll_by_h=per_arch_val_nll_by_h,
                cache_for_reconstruct=cache_loaded,
            )
            print(f"Saved best-over-h1 overlays figure: {out_path}")
            continue

        lambdas_to_process = [float(l) for l in lambda_sweep] if (not bool(args.no_learn)) else [0.0]

        for i_h, bandwidth in enumerate(bandwidths):
            # Precompute KDE on grid once per bandwidth (same across architectures).
            kde_estimator = ParzenWindowEstimator(train_xy, window_size=float(bandwidth))
            estimated_pdf_kde = kde_estimator.estimate_pdf(plotter)
            real_pdf = mixture.get_mesh(plotter.pos)
            mse_kde = float(np.mean((estimated_pdf_kde - real_pdf) ** 2))
            if cache_loaded is None:
                kde_mse_by_h.append(mse_kde)

            # Data-only validation metric (no oracle): average log-likelihood of held-out points.
            ll_kde_val = average_log_likelihood_kde(val_xy, train_xy, float(bandwidth))
            nll_kde_val = _nll_from_avg_loglik(ll_kde_val)
            if cache_loaded is None:
                kde_val_ll_by_h.append(ll_kde_val)
                kde_val_nll_by_h.append(nll_kde_val)
            print(
                f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}: "
                f"KDE EvalMSE={mse_kde:.6e}, ValAvgLogLik={ll_kde_val:.6e} (ValAvgNLL={nll_kde_val:.6e})"
            )

            # Track per-architecture results for this bandwidth (λ=0 only, for existing figures).
            arch_labels: list[str] = []
            per_arch_eval_mse_hist: list[list[float]] = []
            per_arch_train_loss_hist: list[list[float]] = []
            per_arch_est_pdf: list[np.ndarray] = []
            per_arch_final_mse: list[float] = []
            per_arch_best_lr: list[float] = []
            per_arch_best_val_nll: list[float] = []

            for cfg_idx, cfg in enumerate(pnn_architectures):
                label = arch_label(cfg)
                arch_labels.append(label)

                # Train/cache a full lambda sweep, but keep plotting/summary based on lambda=0.
                for lam in lambdas_to_process:
                    lam_key = str(float(lam))
                    log_filename = f"logs/mixture{mixture_idx+1}_{label}_lam_{_lam_tag(lam)}_h1_{_h_tag(bandwidth)}.txt"

                    # Repeat training for statistical stability: run multiple iterations per (h1,arch,lambda).
                    num_iterations = 10
                    iter_mse_hists: list[list[float]] = []
                    iter_final_kde_mses: list[float] = []
                    iter_final_truth_mses: list[float] = []
                    iter_val_nlls: list[float] = []
                    iter_val_lls: list[float] = []
                    iter_models: list[ParzenNeuralNetwork] = []
                    iter_train_hists: list[list[float]] = []

                    # Load cached runs if available; otherwise train and append to cache.
                    runs_list = []
                    if cache_loaded is not None:
                        pnn_block = cache_loaded.get("pnn", {})
                        runs_by_lambda = pnn_block.get("runs_by_lambda", {}) if isinstance(pnn_block, dict) else {}
                        runs_here = runs_by_lambda.get(lam_key)
                        if runs_here is None and float(lam) == 0.0:
                            runs_here = pnn_block.get("runs") if isinstance(pnn_block, dict) else None
                        if runs_here is not None:
                            try:
                                runs_list = runs_here[cfg_idx][i_h]
                            except Exception:
                                runs_list = []

                    if len(runs_list) > 0:
                        # Reconstruct models from cached state_dicts.
                        for run in runs_list:
                            meta = run.get("meta", {})
                            pnn = _pnn_instantiate_from_cache(meta)
                            sd = run.get("state_dict")
                            if sd is None:
                                raise SystemExit("Cache entry missing state_dict")
                            pnn.load_state_dict(sd)
                            iter_models.append(pnn)
                            iter_mse_hists.append(list(run.get("eval_mse_hist", [])))
                            iter_train_hists.append(list(run.get("train_loss_hist", [])))
                            iter_final_kde_mses.append(float(run.get("final_kde_mse", float("nan"))))
                            iter_final_truth_mses.append(float(run.get("final_truth_mse", float("nan"))))
                            iter_val_nlls.append(float(run.get("val_nll", float("inf"))))
                            iter_val_lls.append(float(run.get("val_ll", -float("inf"))))
                    else:
                        # Train new runs for this lambda (unless --no-learn, in which case lambdas_to_process=[0]).
                        for it in range(num_iterations):
                            for lr in learning_rates:
                                with open(log_filename, "a", encoding="utf-8") as log_file:
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
                                        # Keep oracle EvalMSE curves for plotting, but do NOT checkpoint/select by them.
                                        mixture=mixture,
                                        log_file=log_file,
                                        learning_rate=float(lr),
                                        epochs=3500,
                                        boundary_points=boundary_pts,
                                        lambda_boundary=float(lam),
                                        verbose=True,
                                        loss_mode="mse" if use_log_density else "relative",
                                        weight_decay=0.0,
                                        num_uniform_points=len(train_xy) if use_log_density else 0,
                                        checkpoint_metric="train_loss",
                                    )

                                # Data-only selection metric: validation NLL on held-out samples.
                                ll_pnn_val = average_log_likelihood_pnn_on_domain(pnn, val_xy, plotter)
                                nll_pnn_val = _nll_from_avg_loglik(ll_pnn_val)

                                # Data-only diagnostic: grid MSE vs the KDE surface (both from data).
                                est_pdf_tmp = pnn.estimate_pdf(plotter)
                                final_kde_mse = float(np.mean((est_pdf_tmp - estimated_pdf_kde) ** 2))

                                # Oracle diagnostic (kept for reporting/plots only; NOT used for selection).
                                final_truth_mse = float(np.mean((est_pdf_tmp - real_pdf) ** 2))

                                print(
                                    f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}, lam={lam:g}, {label}, iter={it}, LR={lr}: "
                                    f"ValAvgNLL={nll_pnn_val:.6e}, GridMSEvsKDE={final_kde_mse:.6e}, GridMSEvsTruth={final_truth_mse:.6e}"
                                )

                                iter_mse_hists.append(list(mse_hist))
                                iter_final_kde_mses.append(final_kde_mse)
                                iter_final_truth_mses.append(final_truth_mse)
                                iter_val_nlls.append(float(nll_pnn_val))
                                iter_val_lls.append(float(ll_pnn_val))
                                iter_models.append(pnn)
                                iter_train_hists.append(list(train_hist))

                                # Cache this run.
                                if pnn_cache_payload is not None:
                                    run_entry = {
                                        "meta": {
                                            "hidden_layers": list(cfg["hidden_layers"]),
                                            "output_activation": str(cfg["out"]),
                                            "init_output_scale": cfg.get("A", "auto"),
                                            "trained_output_scale": (None if pnn.output_scale is None else float(pnn.output_scale)),
                                            "density_parameterization": "log_density" if use_log_density else "density",
                                        },
                                        "iter": int(it),
                                        "lr": float(lr),
                                        "lambda_boundary": float(lam),
                                        "state_dict": _tensor_to_cpu_state_dict(pnn.state_dict()),
                                        "eval_mse_hist": list(mse_hist),
                                        "train_loss_hist": list(train_hist),
                                        "val_nll": float(nll_pnn_val),
                                        "val_ll": float(ll_pnn_val),
                                        "final_kde_mse": float(final_kde_mse),
                                        "final_truth_mse": float(final_truth_mse),
                                    }
                                    try:
                                        pnn_cache_payload["pnn"]["runs_by_lambda"][lam_key][cfg_idx][i_h].append(run_entry)
                                        if float(lam) == 0.0:
                                            # Keep legacy lambda=0 path in sync.
                                            pnn_cache_payload["pnn"]["runs"][cfg_idx][i_h].append(run_entry)
                                    except Exception:
                                        pass

                    # Choose the best iteration by minimum validation NLL.
                    if len(iter_val_nlls) == 0:
                        raise RuntimeError("No PNN iterations produced valid val NLLs")
                    best_idx = int(np.nanargmin(np.asarray(iter_val_nlls, dtype=float)))
                    best_val_nll = float(iter_val_nlls[best_idx])
                    best_val_ll = float(iter_val_lls[best_idx]) if best_idx < len(iter_val_lls) else float(-best_val_nll)
                    best_final_kde_mse = float(iter_final_kde_mses[best_idx]) if best_idx < len(iter_final_kde_mses) else float("nan")
                    best_lr = float(learning_rates[0]) if len(learning_rates) > 0 else float('nan')

                    if pnn_cache_payload is not None:
                        try:
                            pnn_cache_payload["pnn"]["selected_by_lambda"][lam_key][cfg_idx][i_h] = {
                                "best_run_index": int(best_idx),
                                "val_nll": float(best_val_nll),
                                "val_ll": float(best_val_ll),
                                "final_kde_mse": float(best_final_kde_mse),
                                "final_truth_mse": float(iter_final_truth_mses[best_idx]) if best_idx < len(iter_final_truth_mses) else float("nan"),
                                "eval_mse_hist": list(iter_mse_hists[best_idx]) if best_idx < len(iter_mse_hists) else [],
                                "train_loss_hist": list(iter_train_hists[best_idx]) if best_idx < len(iter_train_hists) else [],
                                "lr": float(best_lr),
                                "lambda_boundary": float(lam),
                            }
                            if float(lam) == 0.0:
                                pnn_cache_payload["pnn"]["selected"][cfg_idx][i_h] = dict(pnn_cache_payload["pnn"]["selected_by_lambda"][lam_key][cfg_idx][i_h])
                        except Exception:
                            pass

                    # For lambda=0 only: feed existing figures/summary arrays.
                    if float(lam) == 0.0:
                        # Keep full per-iteration ValNLL list for the final mean±std plot.
                        try:
                            per_arch_val_nlls_by_h[cfg_idx][i_h] = [float(x) for x in iter_val_nlls]
                        except Exception:
                            pass

                        best_model = iter_models[best_idx]
                        best_hist = list(iter_mse_hists[best_idx]) if best_idx < len(iter_mse_hists) else []
                        best_train_hist = list(iter_train_hists[best_idx]) if best_idx < len(iter_train_hists) else []

                        if cache_loaded is None:
                            per_arch_eval_mse_by_h[cfg_idx].append(list(iter_mse_hists))
                            per_arch_final_mse_by_h[cfg_idx].append(list(iter_final_truth_mses))

                        per_arch_best_lr.append(float(best_lr))
                        per_arch_best_val_nll.append(float(best_val_nll))

                        est_pdf = best_model.estimate_pdf(plotter)
                        per_arch_est_pdf.append(est_pdf)
                        per_arch_eval_mse_hist.append(best_hist)
                        per_arch_train_loss_hist.append(best_train_hist)
                        per_arch_final_mse.append(float(np.mean((est_pdf - real_pdf) ** 2)))

                        if cache_loaded is None:
                            per_arch_val_ll_by_h[cfg_idx].append(float(best_val_ll))
                            per_arch_val_nll_by_h[cfg_idx].append(float(best_val_nll))

                        print(
                            f"Mixture {mixture_idx+1}, h1={bandwidth:.3f}, {label}: "
                            f"ValAvgLogLik={best_val_ll:.6e} (ValAvgNLL={best_val_nll:.6e})"
                        )

                        # Append a compact summary line for lambda=0 logs only.
                        if cache_loaded is None:
                            try:
                                with open(log_filename, "a", encoding="utf-8") as log_file:
                                    log_file.write(
                                        "SUMMARY: "
                                        f"h1={float(bandwidth):.6g}, "
                                        f"label={label}, "
                                        f"lambda={float(lam):.6g}, "
                                        f"final_grid_mse_vs_truth={float(per_arch_final_mse[-1]):.6e}, "
                                        f"final_grid_mse_vs_kde={float(best_final_kde_mse):.6e}, "
                                        f"val_avg_nll={float(best_val_nll):.6e}\n"
                                    )
                            except OSError:
                                pass

            # --------------------------
            # Figure 1: Overlays for this bandwidth (columns = architectures, 2 rows)
            n_arch = len(pnn_architectures)
            fig_ov = plt.figure(figsize=(5 * n_arch, 10))
            fig_ov.suptitle(
                "Surface overlays on the plot grid (fixed train/val split)\n"
                f"Mixture {mixture_idx+1} — bandwidth h_1={float(bandwidth):.3g} (effective h_n=h_1/√(n_train−1)) | "
                "PNN panels show the best run per architecture (min ValNLL on held-out points)"
            )

            for j, label in enumerate(arch_labels):
                # Row 1: KDE vs True
                ax1 = fig_ov.add_subplot(2, n_arch, j + 1, projection='3d')
                ax1.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
                ax1.plot_surface(plotter.X, plotter.Y, estimated_pdf_kde, alpha=0.45, cmap='cividis')
                ax1.set_title(f"{label}\nKDE vs ground truth (same KDE for all columns)")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("pdf")

                # Row 2: PNN vs True
                ax2 = fig_ov.add_subplot(2, n_arch, n_arch + j + 1, projection='3d')
                ax2.plot_surface(plotter.X, plotter.Y, real_pdf, alpha=0.45, cmap='viridis')
                ax2.plot_surface(plotter.X, plotter.Y, per_arch_est_pdf[j], alpha=0.45, cmap='plasma')
                lr_txt = per_arch_best_lr[j]
                lr_note = f",\nbest LR={lr_txt:.2g}" if np.isfinite(lr_txt) else ""
                val_nll_txt = per_arch_best_val_nll[j] if j < len(per_arch_best_val_nll) else float("nan")
                nll_note = f"ValNLL={val_nll_txt:.3g}" if np.isfinite(val_nll_txt) else "ValNLL=?"
                ax2.set_title(f"{label}\nPNN vs ground truth (best-by-ValNLL) — {nll_note}{lr_note}")
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
            fig_lr.suptitle(
                "Training curves and final grid error (oracle)\n"
                f"Mixture {mixture_idx+1} — h_1={float(bandwidth):.3g} | Eval grid MSE is vs ground-truth mixture PDF"
            )

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
                        label='Train loss',
                    )[0]
                    bounds = _bottom_20_bounds(train_hist)
                    if bounds is not None:
                        ax_tr_right.set_ylim(*bounds)

                lr_txt = per_arch_best_lr[j]
                lr_note = f"\nbest LR={lr_txt:.2g}" if np.isfinite(lr_txt) else "best LR=?"
                ax_tr.set_title(f"{label}\nTraining curves (zoomed bottom 20%) — {lr_note}")
                ax_tr.set_xlabel("Epoch")

                # Eval MSE is an *oracle* metric: grid MSE between estimate and the known ground-truth mixture PDF.
                ax_tr.set_ylabel("Eval grid MSE (vs ground truth)", color='tab:blue', labelpad=18)

                # Train loss is the optimization objective. With log-density parameterization this is MSE on log-targets.
                if use_log_density:
                    ax_tr_right.set_ylabel("Train loss (MSE on log targets)", color='tab:orange')
                else:
                    ax_tr_right.set_ylabel("Train loss", color='tab:orange')
                ax_tr.tick_params(axis='y', colors='tab:blue')
                ax_tr_right.tick_params(axis='y', colors='tab:orange')
                ax_tr.grid(True, alpha=0.3)

                # Highlight minimum Eval MSE on the curve (PW-style callout).
                try:
                    if eval_hist.size > 0 and np.any(np.isfinite(eval_hist)):
                        idx_min = int(np.nanargmin(eval_hist))
                        x_min = int(idx_min)
                        y_min = float(eval_hist[idx_min])
                        ax_tr.scatter(
                            [x_min],
                            [y_min],
                            s=140,
                            facecolors='tab:blue',
                            edgecolors='k',
                            linewidths=2.2,
                            marker='o',
                            zorder=20,
                        )

                        xlim = ax_tr.get_xlim()
                        ylim = ax_tr.get_ylim()
                        x_off = 0.05 * max(1.0, (xlim[1] - xlim[0]))
                        y_off = 0.18 * max(1e-18, (ylim[1] - ylim[0]))
                        x_text = float(min(xlim[1], max(xlim[0], x_min + x_off)))
                        y_text = float(min(ylim[1], max(ylim[0], y_min + y_off)))
                        ax_tr.plot([x_min, x_text], [y_min, y_text], color='tab:blue', linewidth=1.2, zorder=19)
                        ax_tr.text(
                            x_text,
                            y_text,
                            f"EvalMSE min={y_min:.3e}\n(epoch={x_min})",
                            color='tab:blue',
                            fontsize=9,
                            bbox={'facecolor': 'white', 'alpha': 0.95, 'edgecolor': 'tab:blue', 'pad': 0.4},
                            zorder=21,
                        )
                except Exception:
                    pass

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
                ax_cmp.set_title(f"{label}\nFinal grid MSE vs ground truth (PNN vs KDE) — selection from 10 runs by Val NLL")
                ax_cmp.set_ylabel("Grid MSE")
                ax_cmp.grid(True, axis='y', alpha=0.3)

            learning_fig_filename = f"figures/learning_results_h1_{_h_tag(bandwidth)}_mixture{mixture_idx+1}.jpeg"
            plt.tight_layout()
            plt.savefig(learning_fig_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_lr)
            print(f"Saved learning results figure: {learning_fig_filename}")

        # --------------------------
        # Figure: Best-over-h1 overlays for this mixture (independent of h1 filename)
        try:
            cache_for_reconstruct = cache_loaded if cache_loaded is not None else pnn_cache_payload
            if cache_for_reconstruct is None:
                raise RuntimeError("No cache available to reconstruct best runs")
            out_path = _plot_best_over_h1_overlays_for_mixture(
                mixture_idx=mixture_idx,
                mixture=mixture,
                plotter=plotter,
                train_xy=train_xy,
                val_xy=val_xy,
                bandwidths=[float(h) for h in bandwidths],
                pnn_architectures=pnn_architectures,
                arch_label_fn=arch_label,
                per_arch_val_nll_by_h=per_arch_val_nll_by_h,
                cache_for_reconstruct=cache_for_reconstruct,
            )
            print(f"Saved best-over-h1 overlays figure: {out_path}")
        except Exception as e:
            print(f"Warning: failed to generate best-over-h1 overlays for mixture {mixture_idx+1}: {e}")

        # Persist cache after finishing this mixture.
        if pnn_cache_payload is not None:
            # Add aggregated arrays used by consolidated plots/JSON.
            pnn_cache_payload["kde"] = {
                "grid_mse": [float(v) for v in kde_mse_by_h],
                "val_avg_nll": [float(v) for v in kde_val_nll_by_h],
                "val_avg_loglik": [float(v) for v in kde_val_ll_by_h],
            }
            pnn_cache_payload["pnn"]["per_arch_eval_mse_by_h"] = per_arch_eval_mse_by_h
            pnn_cache_payload["pnn"]["per_arch_final_mse_by_h"] = per_arch_final_mse_by_h
            pnn_cache_payload["pnn"]["per_arch_val_ll_by_h"] = per_arch_val_ll_by_h
            pnn_cache_payload["pnn"]["per_arch_val_nll_by_h"] = per_arch_val_nll_by_h
            pnn_cache_payload["pnn"]["per_arch_val_nlls_by_h"] = per_arch_val_nlls_by_h
            try:
                torch.save(pnn_cache_payload, cache_path)
                print(f"Saved PNN sweep cache: {cache_path}")
            except Exception as e:
                print(f"WARN: failed to save PNN sweep cache to {cache_path}: {e}")

        # --------------------------
        # Consolidated learning results across all bandwidths for this mixture.
        # Layout: columns = architectures, 2 rows
        n_arch = len(pnn_architectures)
        fig_lr_all = plt.figure(figsize=(5 * n_arch, 12))
        fig_lr_all.suptitle(
            "Bandwidth sweep: training curves and diagnostics\n"
            f"Mixture {mixture_idx + 1} — PNN trains on KDE targets (sample-only) and selects by held-out Val NLL;\n"
            "Eval grid MSE vs ground-truth is computed only for visualization; training repeated 10 iterations per (h1,arch)",
            fontsize=11,
        )

        # Build surfaces: bandwidth x epochs (use representative iteration closest to mean)
        max_epochs_tracked = 0
        for j in range(n_arch):
            for hists_for_h in per_arch_eval_mse_by_h[j]:
                # hists_for_h is a list of iteration histories for one h
                for hist in hists_for_h:
                    max_epochs_tracked = max(max_epochs_tracked, len(hist))

        # Determine zoomed z-range (bottom 20% of overall range)
        all_vals = []
        for j in range(n_arch):
            for hists_for_h in per_arch_eval_mse_by_h[j]:
                for hist in hists_for_h:
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
                hists_for_h = per_arch_eval_mse_by_h[j][i_h] if i_h < len(per_arch_eval_mse_by_h[j]) else []
                # hists_for_h: list of iteration histories
                if len(hists_for_h) > 0:
                    # Truncate to the minimum epoch length across iterations for averaging/distance
                    min_len = min(len(h) for h in hists_for_h)
                    if min_len <= 0:
                        surface[i_h, :] = np.nan
                        continue
                    arr = np.asarray([np.asarray(h[:min_len], dtype=float) for h in hists_for_h], dtype=float)
                    mean_hist = np.nanmean(arr, axis=0)
                    # find iteration closest to mean (L2 distance)
                    dists = np.linalg.norm(arr - mean_hist[None, :], axis=1)
                    rep_idx = int(np.nanargmin(dists))
                    rep_hist = list(hists_for_h[rep_idx])
                    vals = np.asarray(rep_hist, dtype=float)
                    surface[i_h, : len(vals)] = vals
                    if len(vals) < len(E):
                        surface[i_h, len(vals) :] = float(vals[-1])
                else:
                    surface[i_h, :] = np.nan

            ax_surf.plot_surface(EE, HH, surface, cmap='viridis', linewidth=0, antialiased=False, alpha=0.8)
            title_obj = ax_surf.set_title(
                f"{label}\nBandwidth sweep: Eval grid MSE vs truth",
                fontsize=9,
                pad=12,
            )
            try:
                title_obj.set_wrap(True)
            except Exception:
                pass
            ax_surf.set_xlabel("Epoch")
            ax_surf.set_ylabel("h_1")
            base_zlabel = "Eval grid MSE"
            ax_surf.set_zlabel(base_zlabel)
            ax_surf.set_zlim(global_zmin, zoom_zmax)
            ax_surf.grid(True)

            # Use scientific notation for readability, but avoid Matplotlib's separate
            # offset text (e.g. “×10^k”) which in 3D often overlaps the z-label.
            # We instead fold the exponent into the z-label and hide the offset text.
            try:
                from matplotlib.ticker import ScalarFormatter

                zfmt = ScalarFormatter(useMathText=True)
                zfmt.set_powerlimits((0, 0))
                ax_surf.zaxis.set_major_formatter(zfmt)

                # Force a draw so ScalarFormatter computes orderOfMagnitude.
                try:
                    ax_surf.figure.canvas.draw()
                except Exception:
                    pass

                try:
                    oom = int(getattr(zfmt, "orderOfMagnitude", 0))
                except Exception:
                    oom = 0
                if oom != 0:
                    # IMPORTANT: avoid `$...$` here. In 3D, Matplotlib may decide the
                    # label is already mathtext and then choke on embedded `$`.
                    # Plain text keeps layout stable and avoids parse errors.
                    ax_surf.set_zlabel(f"{base_zlabel} (×10^{oom})")

                try:
                    ax_surf.zaxis.get_offset_text().set_visible(False)
                except Exception:
                    pass
            except Exception:
                pass

            # Rotate 270° clockwise around z (matplotlib azimuth).
            ax_surf.view_init(elev=33, azim=30)

            # Highlight points on the surface:
            #  - global minimum EvalMSE (over all epochs × h_1)
            #  - minimum Val NLL (over h_1), shown at the final tracked epoch for that h_1
            try:
                finite_mask = np.isfinite(surface)
                if np.any(finite_mask):
                    flat_idx = int(np.nanargmin(surface))
                    i_h_min, i_e_min = np.unravel_index(flat_idx, surface.shape)
                    e_min = int(E[i_e_min])
                    h_min = float(H[i_h_min])
                    mse_min = float(surface[i_h_min, i_e_min])
                    nll_at_mse_min = float(per_arch_val_nll_by_h[j][i_h_min]) if i_h_min < len(per_arch_val_nll_by_h[j]) else float('nan')

                    ax_surf.scatter([e_min], [h_min], [mse_min], s=70, c='tab:red', edgecolors='k', linewidths=1.2, zorder=30)

                    # Min ValNLL point (placed at final epoch for that bandwidth)
                    nlls = np.asarray(per_arch_val_nll_by_h[j], dtype=float)
                    red_handle = None
                    blue_handle = None
                    red_label = None
                    blue_label = None
                    if nlls.size > 0 and np.any(np.isfinite(nlls)):
                        i_h_nll = int(np.nanargmin(nlls))
                        h_nll = float(H[i_h_nll])
                        # Place at the last available epoch index for that history (or 0).
                        hist_for_h = per_arch_eval_mse_by_h[j][i_h_nll] if i_h_nll < len(per_arch_eval_mse_by_h[j]) else []
                        # hist_for_h is a list of iteration histories; place at the last epoch of the representative iteration
                        if len(hist_for_h) > 0:
                            # pick the representative iter (closest to mean) used above
                            min_len = min(len(hh) for hh in hist_for_h)
                            arr = np.asarray([np.asarray(hh[:min_len], dtype=float) for hh in hist_for_h], dtype=float)
                            mean_hist = np.nanmean(arr, axis=0)
                            rep_idx = int(np.nanargmin(np.linalg.norm(arr - mean_hist[None, :], axis=1)))
                            rep_hist = hist_for_h[rep_idx]
                            i_e_nll = max(0, int(len(rep_hist) - 1))
                        else:
                            i_e_nll = 0
                        e_nll = int(E[min(i_e_nll, len(E) - 1)])
                        mse_at_nll = float(surface[i_h_nll, min(i_e_nll, surface.shape[1] - 1)])
                        nll_min = float(nlls[i_h_nll])

                        ax_surf.scatter([e_nll], [h_nll], [mse_at_nll], s=70, c='tab:blue', edgecolors='k', linewidths=1.2, zorder=30)

                    # Build legend entries (do not draw verbose text on the surface)
                    try:
                        from matplotlib.lines import Line2D

                        # Red: Min EvalMSE
                        red_label = (
                            f"Min EvalMSE: {mse_min:.3e}, {nll_at_mse_min:.3e}\n"
                            f"(h_1={h_min:.3g}, epoch={e_min})"
                        )
                        red_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', markeredgecolor='k', markersize=7, linestyle='')

                        # Blue: Min ValNLL
                        if nlls.size > 0 and np.any(np.isfinite(nlls)):
                            blue_label = (
                                f"Min ValNLL: {mse_at_nll:.3e}, {nll_min:.3e}\n"
                                f"(h_1={h_nll:.3g}, epoch={e_nll})"
                            )
                        else:
                            blue_label = "Min ValNLL: N/A"
                        blue_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markeredgecolor='k', markersize=7, linestyle='')

                        # Place legend out of the way (upper-left inside a transparent box)
                        legend_handles = [red_handle, blue_handle]
                        legend_labels = [red_label, blue_label]
                        ax_surf.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9, framealpha=0.85)
                    except Exception:
                        pass
            except Exception:
                pass

            # Row 2: MSE vs bandwidth (PNN vs KDE) with errorbars across iterations
            ax_line = fig_lr_all.add_subplot(2, n_arch, n_arch + j + 1)
            # per_arch_final_mse_by_h[j] is a list (per h) of lists (per iteration)
            pnn_means = []
            pnn_stds = []
            for i_h in range(len(bandwidths)):
                it_vals = per_arch_final_mse_by_h[j][i_h] if i_h < len(per_arch_final_mse_by_h[j]) else []
                if len(it_vals) > 0:
                    arr = np.asarray(it_vals, dtype=float)
                    pnn_means.append(float(np.nanmean(arr)))
                    pnn_stds.append(float(np.nanstd(arr)))
                else:
                    pnn_means.append(float('nan'))
                    pnn_stds.append(0.0)
            kde_means = np.asarray(kde_mse_by_h, dtype=float)
            kde_stds = np.zeros_like(kde_means)
            Hf = H
            ax_line.errorbar(Hf, np.asarray(pnn_means, dtype=float), yerr=np.asarray(pnn_stds, dtype=float), marker='o', color='tab:purple', label='PNN (mean ± std)')
            ax_line.errorbar(Hf, kde_means, yerr=kde_stds, marker='o', color='tab:gray', label='KDE')
            title_obj = ax_line.set_title(
                f"{label}\nFinal grid MSE vs h_1 (mean ± std over runs)",
                fontsize=9,
                pad=8,
            )
            try:
                title_obj.set_wrap(True)
            except Exception:
                pass
            ax_line.set_xlabel("h_1")
            ax_line.set_ylabel("MSE")
            try:
                ax_line.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            except Exception:
                pass
            # Zoom y-range after sci notation for readability.
            try:
                pnn_arr = np.asarray(pnn_means, dtype=float)
                kde_arr = np.asarray(kde_means, dtype=float)
                finite_vals = np.concatenate([pnn_arr[np.isfinite(pnn_arr)], kde_arr[np.isfinite(kde_arr)]])
                if finite_vals.size > 0:
                    ymin = float(np.min(finite_vals))
                    ymax = float(np.max(finite_vals))
                    pad = 0.15 * max(1e-18, (ymax - ymin))
                    ax_line.set_ylim(ymin - pad, ymax + pad)
            except Exception:
                pass
            ax_line.grid(True, alpha=0.3)
            ax_line.legend()

        lr_all_filename = f"figures/learning_results_bandwidth_sweep_mixture{mixture_idx+1}.jpeg"
        # 3D subplots + long labels tend to clash with tight_layout; use manual spacing.
        try:
            fig_lr_all.subplots_adjust(top=0.90, bottom=0.06, wspace=0.30, hspace=0.42)
        except Exception:
            pass
        fig_lr_all.savefig(lr_all_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_lr_all)
        print(f"Saved consolidated learning results figure: {lr_all_filename}")

        # --------------------------
        # NEW: Data-only cross-validation plot (validation NLL) across bandwidths.
        # Use all 10 trained PNN iterations: plot grouped bars with mean±std.
        # KDE is included as a bar series too (std = 0).
        fig_cv = plt.figure(figsize=(12, 6))
        ax_cv = fig_cv.add_subplot(111)

        h1_vals = [float(h) for h in bandwidths]
        x = np.arange(len(h1_vals), dtype=float)

        # Series: KDE + each PNN architecture
        series_labels = ["KDE"] + [arch_label(cfg) for cfg in pnn_architectures]
        n_series = len(series_labels)
        width = 0.84 / max(1, n_series)
        offsets = (np.arange(n_series, dtype=float) - (n_series - 1) / 2.0) * width

        # KDE bars
        kde_means = np.asarray(kde_val_nll_by_h, dtype=float)
        kde_stds = np.zeros_like(kde_means)
        ax_cv.bar(
            x + offsets[0],
            kde_means,
            yerr=kde_stds,
            width=0.95 * width,
            capsize=4,
            alpha=0.80,
            color='tab:gray',
            label="KDE (single run)",
        )

        # PNN bars: mean±std across iterations
        n_arch = len(pnn_architectures)
        try:
            colors = [plt.get_cmap('tab10')(i % 10) for i in range(n_arch)]
        except Exception:
            colors = [None] * n_arch

        for cfg_idx, cfg in enumerate(pnn_architectures):
            label = arch_label(cfg)
            means: list[float] = []
            stds: list[float] = []
            for i_h in range(len(h1_vals)):
                vals: list[float]
                try:
                    vals = list(per_arch_val_nlls_by_h[cfg_idx][i_h])
                except Exception:
                    vals = []

                # Backward compatibility: if per-iteration list is missing, fall back to the selected best run.
                if len(vals) == 0:
                    try:
                        vals = [float(per_arch_val_nll_by_h[cfg_idx][i_h])]
                    except Exception:
                        vals = []

                arr = np.asarray(vals, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    means.append(float('nan'))
                    stds.append(0.0)
                else:
                    means.append(float(np.mean(arr)))
                    stds.append(float(np.std(arr)))

            series_pos = 1 + cfg_idx
            ax_cv.bar(
                x + offsets[series_pos],
                np.asarray(means, dtype=float),
                yerr=np.asarray(stds, dtype=float),
                width=0.95 * width,
                capsize=4,
                alpha=0.85,
                color=colors[cfg_idx] if cfg_idx < len(colors) else None,
                label=f"PNN mean±std (10 runs): {label}",
            )

        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels([f"{h:g}" for h in h1_vals])
        ax_cv.set_title(f"Validation NLL vs h_1 — mixture {mixture_idx+1} (bars: mean±std over 10 PNN runs)")
        ax_cv.set_xlabel("h_1")
        ax_cv.set_ylabel("Avg NLL on held-out points (lower is better)")
        ax_cv.grid(True, axis='y', alpha=0.3)
        ax_cv.legend(fontsize='small')
        cv_filename = f"figures/validation_nll_bandwidth_sweep_mixture{mixture_idx+1}.jpeg"
        plt.tight_layout()
        plt.savefig(cv_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_cv)
        print(f"Saved validation NLL figure: {cv_filename}")

        # --------------------------
        # Boundary penalty comparison figure (λ=0, 0.01, 0.1) for the best-by-NLL (λ=0) config.
        # IMPORTANT: do not retrain here; reconstruct from the sweep cache.
        boundary_demo_payload = None
        if cache_loaded is not None:
            boundary_demo_payload = cache_loaded.get("boundary_penalty_demo")

        best_cfg_idx = None
        best_i_h = None
        best_h1 = None
        best_nll = float("inf")
        for cfg_idx in range(n_arch_total):
            nlls = per_arch_val_nll_by_h[cfg_idx]
            for i_h, nll in enumerate(nlls):
                if np.isfinite(nll) and float(nll) < best_nll:
                    best_nll = float(nll)
                    best_cfg_idx = int(cfg_idx)
                    best_i_h = int(i_h)
                    best_h1 = float(bandwidths[i_h])

        if best_cfg_idx is not None and best_i_h is not None and best_h1 is not None:
            cfg = pnn_architectures[best_cfg_idx]
            label = arch_label(cfg)
            demo_lambdas = [0.0, 0.01, 0.1]

            if bool(args.no_learn) and (cache_loaded is None):
                print(
                    f"Boundary penalty comparison skipped in --no-learn mode (no cache loaded) "
                    f"for mixture {mixture_idx+1}."
                )
            else:
                cache_for_demo = cache_loaded if cache_loaded is not None else pnn_cache_payload
                if cache_for_demo is None:
                    cache_for_demo = cache_loaded

                print(
                    f"Boundary penalty comparison (mixture {mixture_idx+1}): "
                    f"best-by-NLL (λ=0): h1={best_h1:.3f}, arch={label}; demo lambdas={demo_lambdas}"
                )

                models: dict[float, ParzenNeuralNetwork] = {}
                metrics: dict[float, dict[str, float]] = {}

                if cache_for_demo is not None:
                    for lam in demo_lambdas:
                        try:
                            run_entry = _pnn_best_run_from_cache_for(
                                cache_for_demo,
                                cfg_idx=int(best_cfg_idx),
                                i_h=int(best_i_h),
                                lambda_boundary=float(lam),
                            )
                            meta = run_entry.get("meta", {})
                            pnn_demo = _pnn_instantiate_from_cache(meta)
                            sd = run_entry.get("state_dict")
                            if sd is None:
                                raise RuntimeError("Cache entry missing state_dict")
                            pnn_demo.load_state_dict(sd)
                            models[float(lam)] = pnn_demo

                            # Metrics computed on the current fixed split.
                            boundary_mean = mean_unnormalized_density_on_points(pnn_demo, boundary_pts)
                            val_ll = average_log_likelihood_pnn_on_domain(pnn_demo, val_xy, plotter)
                            val_nll = _nll_from_avg_loglik(val_ll)
                            metrics[float(lam)] = {
                                "boundary_mean": float(boundary_mean),
                                "val_ll": float(val_ll),
                                "val_nll": float(val_nll),
                            }
                        except Exception as e:
                            print(f"WARN: missing cached runs for mixture {mixture_idx+1}, lam={lam:g}: {e}")

                if len(models) >= 2:
                    kde_for_demo = ParzenWindowEstimator(train_xy, window_size=float(best_h1))
                    kde_pdf = kde_for_demo.estimate_pdf(plotter)
                    true_pdf = mixture.get_mesh(plotter.pos)

                    n_panels = 1 + len(demo_lambdas)
                    fig_b = plt.figure(figsize=(5.2 * n_panels, 8.0))
                    fig_b.suptitle(f"Boundary penalty comparison — mixture {mixture_idx+1}\n{label}, h_1={best_h1:.3f}")

                    ax_kde = fig_b.add_subplot(1, n_panels, 1, projection='3d')
                    ax_kde.plot_surface(plotter.X, plotter.Y, true_pdf, alpha=0.45, cmap='viridis')
                    ax_kde.plot_surface(plotter.X, plotter.Y, kde_pdf, alpha=0.45, cmap='cividis')
                    ax_kde.set_title("KDE vs True")
                    ax_kde.set_xlabel("x")
                    ax_kde.set_ylabel("y")
                    ax_kde.set_zlabel("pdf")

                    for j, lam in enumerate(demo_lambdas, start=2):
                        ax = fig_b.add_subplot(1, n_panels, j, projection='3d')
                        ax.plot_surface(plotter.X, plotter.Y, true_pdf, alpha=0.45, cmap='viridis')
                        if float(lam) in models:
                            pdf = models[float(lam)].estimate_pdf(plotter)
                            ax.plot_surface(plotter.X, plotter.Y, pdf, alpha=0.45, cmap='plasma')
                            m = metrics.get(float(lam), {})
                            ax.set_title(
                                f"PNN vs True (λ={lam:g})\nValNLL={m.get('val_nll', float('nan')):.3g}, boundaryMean={m.get('boundary_mean', float('nan')):.3g}"
                            )
                        else:
                            ax.set_title(f"PNN (λ={lam:g}) — missing")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("pdf")

                    b_filename = f"figures/boundary_penalty_comparison_mixture{mixture_idx+1}.jpeg"
                    plt.tight_layout()
                    plt.savefig(b_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig_b)
                    print(f"Saved boundary-penalty comparison figure: {b_filename}")

                    boundary_demo_payload = {
                        "h1": float(best_h1),
                        "label": str(label),
                        "figure": str(b_filename),
                    }
                    for idx, lam in enumerate(demo_lambdas):
                        k = f"lambda_{idx}"
                        if float(lam) in metrics:
                            boundary_demo_payload[k] = {"lambda": float(lam), "metrics": dict(metrics[float(lam)])}

                    if pnn_cache_payload is not None:
                        pnn_cache_payload["boundary_penalty_demo"] = boundary_demo_payload

            # --------------------------
            # NEW (optional): Uniform supervision ablation (no-uniform vs uniform) for the same best-by-NLL config.
            # Goal: show that adding interior Parzen targets reduces extrapolation artifacts far from samples.
            # Disabled by default; enable with: --uniform-supervision-demo
            if ("--uniform-supervision-demo" in sys.argv) and (not bool(args.no_learn)):
                try:
                    demo_epochs = 2000
                    demo_lr = float(learning_rates[0])
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

            # Refresh the cache at the end of the mixture sweep (may include demo payloads).
            if pnn_cache_payload is not None:
                try:
                    torch.save(pnn_cache_payload, cache_path)
                except Exception as e:
                    print(f"WARN: failed to update PNN sweep cache to {cache_path}: {e}")

            # Export a JSON summary for the whole sweep (report-ready, machine-readable).
            try:
                def _float_or_none(v: object) -> float | None:
                    try:
                        fv = float(v)
                    except Exception:
                        return None
                    if not np.isfinite(fv):
                        return None
                    return fv

                def _pad_1d(values: list[object], n: int) -> list[float | None]:
                    out = [_float_or_none(v) for v in values[:n]]
                    if len(out) < n:
                        out.extend([None] * (n - len(out)))
                    return out

                def _boundary_demo_json(payload: object) -> dict | None:
                    # The cache may contain torch Tensors (state_dict). Those must NOT go into JSON.
                    if not isinstance(payload, dict):
                        return None
                    out: dict = {}
                    for k in ("h1", "label", "figure"):
                        if k in payload:
                            out[k] = payload[k]
                    lambda_keys = sorted(
                        [k for k in payload.keys() if isinstance(k, str) and k.startswith("lambda_")],
                        key=lambda s: int(s.split("_", 1)[1]) if s.split("_", 1)[1].isdigit() else s,
                    )
                    for lk in lambda_keys:
                        if isinstance(payload.get(lk), dict):
                            entry = payload[lk]
                            lam = entry.get("lambda")
                            metrics = entry.get("metrics")
                            if isinstance(metrics, dict):
                                out[lk] = {"lambda": _float_or_none(lam), "metrics": metrics}
                            else:
                                # Backward compatibility: older payloads stored metrics flat.
                                # Keep only JSON-friendly scalars.
                                flat = {kk: _float_or_none(vv) for kk, vv in entry.items() if kk != "models"}
                                out[lk] = flat
                    if len(out) == 0:
                        return None
                    out["models_saved_in_cache"] = bool("models" in payload)
                    return out

                # per_arch_final_mse_by_h is 3-D: [arch][h][iter]. Export a stable 2-D summary
                # plus the raw per-iteration values.
                pnn_final_grid_mse_mean: list[list[float | None]] = []
                pnn_final_grid_mse_std: list[list[float | None]] = []
                pnn_final_grid_mse_runs: list[list[list[float | None]]] = []
                for cfg_idx in range(n_arch_total):
                    means_row: list[float | None] = []
                    stds_row: list[float | None] = []
                    runs_row: list[list[float | None]] = []
                    for i_h in range(len(bandwidths)):
                        it_vals = per_arch_final_mse_by_h[cfg_idx][i_h] if i_h < len(per_arch_final_mse_by_h[cfg_idx]) else []
                        runs_row.append([_float_or_none(v) for v in it_vals])
                        if len(it_vals) > 0:
                            arr = np.asarray(it_vals, dtype=float)
                            means_row.append(_float_or_none(np.nanmean(arr)))
                            stds_row.append(_float_or_none(np.nanstd(arr)))
                        else:
                            means_row.append(None)
                            stds_row.append(None)
                    pnn_final_grid_mse_mean.append(means_row)
                    pnn_final_grid_mse_std.append(stds_row)
                    pnn_final_grid_mse_runs.append(runs_row)

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
                        "final_grid_mse": pnn_final_grid_mse_mean,
                        "final_grid_mse_std": pnn_final_grid_mse_std,
                        "final_grid_mse_runs": pnn_final_grid_mse_runs,
                        "val_avg_nll": [_pad_1d(per_arch_val_nll_by_h[i], len(bandwidths)) for i in range(n_arch_total)],
                        "val_avg_loglik": [_pad_1d(per_arch_val_ll_by_h[i], len(bandwidths)) for i in range(n_arch_total)],
                    },
                    "best_by_val_nll": {
                        "label": arch_label(pnn_architectures[int(best_cfg_idx)]),
                        "h1": float(best_h1),
                        "val_avg_nll": float(best_nll),
                    },
                    "boundary_penalty_demo": _boundary_demo_json(boundary_demo_payload),
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
