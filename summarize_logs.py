import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class LogRow:
    file: str
    mixture: int
    label: str
    h1_file: Optional[float]
    epoch: int
    train_loss: float
    eval_mse: Optional[float]
    penalty: Optional[float]
    h1: Optional[float]
    h_n: Optional[float]
    lr: Optional[float]
    loss: Optional[str]


_EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+):\s*"
    r"TrainLoss=(?P<train>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
    r"(?:,\s*Pen=(?P<pen>[-+]?\d*\.?\d+(?:e[-+]?\d+)?))?"
    r"(?:,\s*EvalMSE=(?P<eval>[-+]?\d*\.?\d+(?:e[-+]?\d+)?))?"
    r"\s*\(h1=(?P<h1>[^,]+),\s*h_n=(?P<hn>[^,]+),\s*lr=(?P<lr>[^,]+),\s*loss=(?P<loss>[^\)]+)\)\s*$"
)

_FILENAME_RE = re.compile(
    r"^mixture(?P<mixture>\d+)_(?P<label>.+)_h1_(?P<h1tag>[-+]?\d+p\d+)\.txt$"
)


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _parse_h1_tag(tag: str) -> Optional[float]:
    # e.g. 2p00 -> 2.00
    try:
        return float(tag.replace("p", "."))
    except Exception:
        return None


def iter_log_rows(log_dir: Path) -> Iterable[LogRow]:
    for path in sorted(log_dir.glob("*.txt")):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        mixture = int(m.group("mixture"))
        label = m.group("label")
        h1_file = _parse_h1_tag(m.group("h1tag"))

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mm = _EPOCH_RE.match(line)
                if not mm:
                    continue

                epoch = int(mm.group("epoch"))
                train_loss = float(mm.group("train"))
                pen = _parse_float(mm.group("pen")) if mm.group("pen") is not None else None
                eval_mse = _parse_float(mm.group("eval")) if mm.group("eval") is not None else None
                h1 = _parse_float(mm.group("h1"))
                hn = _parse_float(mm.group("hn"))
                lr = _parse_float(mm.group("lr"))
                loss = mm.group("loss")

                yield LogRow(
                    file=path.name,
                    mixture=mixture,
                    label=label,
                    h1_file=h1_file,
                    epoch=epoch,
                    train_loss=train_loss,
                    eval_mse=eval_mse,
                    penalty=pen,
                    h1=h1,
                    h_n=hn,
                    lr=lr,
                    loss=loss,
                )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    log_dir = repo_root / "logs"
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    rows = list(iter_log_rows(log_dir))

    # Write raw parsed rows
    raw_csv = out_dir / "log_rows.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "mixture",
                "label",
                "h1_file",
                "epoch",
                "train_loss",
                "penalty",
                "eval_mse",
                "h1",
                "h_n",
                "lr",
                "loss",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.file,
                    r.mixture,
                    r.label,
                    r.h1_file,
                    r.epoch,
                    r.train_loss,
                    r.penalty,
                    r.eval_mse,
                    r.h1,
                    r.h_n,
                    r.lr,
                    r.loss,
                ]
            )

    # Best-by-file (min eval_mse across all epochs/blocks), ignoring missing eval_mse
    best_by_file = {}
    for r in rows:
        if r.eval_mse is None:
            continue
        key = (r.file,)
        cur = best_by_file.get(key)
        if cur is None or r.eval_mse < cur.eval_mse:
            best_by_file[key] = r

    best_file_csv = out_dir / "best_by_log_file.csv"
    with best_file_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "mixture",
                "label",
                "h1_file",
                "best_epoch",
                "best_eval_mse",
                "h1",
                "h_n",
                "lr",
                "loss",
            ]
        )
        for (file_key,), r in sorted(best_by_file.items(), key=lambda kv: (kv[1].mixture, kv[1].label, kv[1].h1_file or 0.0)):
            w.writerow([r.file, r.mixture, r.label, r.h1_file, r.epoch, r.eval_mse, r.h1, r.h_n, r.lr, r.loss])

    # Best-by-mixture overall (across all log files)
    best_by_mixture = {}
    for r in best_by_file.values():
        cur = best_by_mixture.get(r.mixture)
        if cur is None or (r.eval_mse is not None and cur.eval_mse is not None and r.eval_mse < cur.eval_mse):
            best_by_mixture[r.mixture] = r

    best_mix_csv = out_dir / "best_by_mixture_oracle_mse.csv"
    with best_mix_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mixture", "file", "label", "h1_file", "best_epoch", "best_eval_mse", "h1", "h_n", "lr", "loss"])
        for mix in sorted(best_by_mixture.keys()):
            r = best_by_mixture[mix]
            w.writerow([r.mixture, r.file, r.label, r.h1_file, r.epoch, r.eval_mse, r.h1, r.h_n, r.lr, r.loss])

    print(f"Parsed {len(rows)} epoch rows")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {best_file_csv}")
    print(f"Wrote {best_mix_csv}")


if __name__ == "__main__":
    main()
