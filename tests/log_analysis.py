from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_LOG_LINE_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+):\s+"
    r"TrainLoss=(?P<train_loss>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"EvalMSE=(?P<eval_mse>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)[^\(]*\("
    r"h1=(?P<h1>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"h_n=(?P<h_n>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"lr=(?P<lr>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"loss=(?P<loss_mode>[^\)]+)\)\s*$"
)

_SUMMARY_RE = re.compile(
    r"^SUMMARY:\s+h1=(?P<h1>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"label=(?P<label>[^,]+),\s+"
    r"final_grid_mse=(?P<final_grid_mse>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?),\s+"
    r"val_avg_nll=(?P<val_avg_nll>[+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class LogConfig:
    mixture: int
    arch: str
    hidden_activation: str
    output: str
    output_scale: str
    h1: float


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    train_loss: float
    eval_mse: float


@dataclass
class ParsedLog:
    path: Path
    config: LogConfig
    # Summary values (if present)
    summary_label: Optional[str]
    summary_final_grid_mse: Optional[float]
    summary_val_avg_nll: Optional[float]
    # Potentially multiple runs within the same file (epoch counter resets)
    runs: list[list[EpochPoint]]


def _parse_filename(path: Path) -> LogConfig:
    # Example:
    # mixture1_MLP_30-20_sigmoid_outSigmoid_Aauto_h1_12p00.txt
    name = path.stem
    parts = name.split("_")
    if len(parts) < 5 or not parts[0].startswith("mixture"):
        raise ValueError(f"Unexpected log filename: {path.name}")

    mixture = int(parts[0].replace("mixture", ""))
    if parts[1] != "MLP":
        raise ValueError(f"Unexpected log filename (missing MLP): {path.name}")

    arch = parts[2]
    hidden_activation = parts[3]

    # outReLU or outSigmoid or outSigmoid_Aauto
    out_part = parts[4]
    if not out_part.startswith("out"):
        raise ValueError(f"Unexpected log filename (missing out*): {path.name}")
    out_kind = out_part.replace("out", "")

    output_scale = "auto"
    if out_kind.lower() == "relu":
        output = "relu"
    elif out_kind.lower() == "sigmoid":
        output = "sigmoid"
        # optional scale token
        if len(parts) >= 6 and parts[5].startswith("A"):
            output_scale = parts[5].replace("A", "").lower()
    else:
        # tolerate unknown but keep raw
        output = out_kind.lower()

    # Extract h1 robustly (supports h1_12p00, h1_12, h1_12.0).
    m_h1 = re.search(r"(?:^|_)h1_(\d+(?:p\d+)?|\d+\.\d+)(?:_|$)", name)
    if not m_h1:
        raise ValueError(f"Unexpected log filename (missing h1_<value>): {path.name}")
    h1 = float(m_h1.group(1).replace("p", "."))

    return LogConfig(
        mixture=mixture,
        arch=arch,
        hidden_activation=hidden_activation,
        output=output,
        output_scale=output_scale,
        h1=h1,
    )


def parse_training_log(path: Path) -> ParsedLog:
    config = _parse_filename(path)

    runs: list[list[EpochPoint]] = []
    current: list[EpochPoint] = []

    summary_label: Optional[str] = None
    summary_final_grid_mse: Optional[float] = None
    summary_val_avg_nll: Optional[float] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m = _LOG_LINE_RE.match(line)
            if m:
                epoch = int(m.group("epoch"))
                train_loss = float(m.group("train_loss"))
                eval_mse = float(m.group("eval_mse"))

                # New run starts when epoch counter resets.
                if epoch == 0 and current:
                    runs.append(current)
                    current = []

                current.append(EpochPoint(epoch=epoch, train_loss=train_loss, eval_mse=eval_mse))
                continue

            s = _SUMMARY_RE.match(line)
            if s:
                summary_label = s.group("label")
                summary_final_grid_mse = float(s.group("final_grid_mse"))
                summary_val_avg_nll = float(s.group("val_avg_nll"))
                continue

    if current:
        runs.append(current)

    return ParsedLog(
        path=path,
        config=config,
        summary_label=summary_label,
        summary_final_grid_mse=summary_final_grid_mse,
        summary_val_avg_nll=summary_val_avg_nll,
        runs=runs,
    )


def iter_all_logs(logs_dir: Path) -> Iterable[ParsedLog]:
    for path in sorted(logs_dir.glob("*.txt")):
        yield parse_training_log(path)


def last_run(log: ParsedLog) -> list[EpochPoint]:
    if not log.runs:
        return []
    return log.runs[-1]


def tail_std(values: list[float], k: int) -> float:
    if not values:
        return math.nan
    tail = values[-k:] if len(values) >= k else values
    mu = sum(tail) / len(tail)
    var = sum((x - mu) ** 2 for x in tail) / len(tail)
    return math.sqrt(var)


def convergence_epoch_to_within(log_run: list[EpochPoint], target: float, frac: float = 1.05) -> Optional[int]:
    if not log_run:
        return None
    threshold = target * frac
    for pt in log_run:
        if pt.eval_mse <= threshold:
            return pt.epoch
    return None


def min_eval_mse(log_run: list[EpochPoint]) -> Optional[float]:
    if not log_run:
        return None
    return min(pt.eval_mse for pt in log_run)
