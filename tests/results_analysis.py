from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepResults:
    mixture: int
    bandwidths_h1: list[float]
    kde_val_avg_nll: list[float]
    kde_grid_mse: list[float]
    architectures: list[dict[str, Any]]
    pnn_val_avg_nll: list[list[float]]
    pnn_final_grid_mse: list[list[float]]
    best_by_val_nll: dict[str, Any]
    boundary_penalty_demo: dict[str, Any]


def load_sweep_results(path: Path) -> SweepResults:
    data = json.loads(path.read_text(encoding="utf-8"))
    boundary_penalty_demo = dict(data.get("boundary_penalty_demo", {}))
    # Forward/backward compatibility: some artifacts store per-lambda metrics nested under "metrics".
    # Flatten those into the lambda entry so tests can consistently use demo["lambda_0"]["val_nll"], etc.
    for k, v in list(boundary_penalty_demo.items()):
        if isinstance(k, str) and k.startswith("lambda_") and isinstance(v, dict):
            metrics = v.get("metrics")
            if isinstance(metrics, dict):
                for mk, mv in metrics.items():
                    if mk not in v:
                        v[mk] = mv
                boundary_penalty_demo[k] = v
    return SweepResults(
        mixture=int(data["mixture"]),
        bandwidths_h1=[float(x) for x in data["bandwidths_h1"]],
        kde_val_avg_nll=[float(x) for x in data["kde"]["val_avg_nll"]],
        kde_grid_mse=[float(x) for x in data["kde"]["grid_mse"]],
        architectures=list(data["architectures"]),
        pnn_val_avg_nll=[[float(x) for x in row] for row in data["pnn"]["val_avg_nll"]],
        pnn_final_grid_mse=[[float(x) for x in row] for row in data["pnn"]["final_grid_mse"]],
        best_by_val_nll=dict(data["best_by_val_nll"]),
        boundary_penalty_demo=boundary_penalty_demo,
    )


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    return workspace_root() / "results"


def logs_dir() -> Path:
    return workspace_root() / "logs"
