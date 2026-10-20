"""Aggregate every run record into ``results/experiments.csv``.

Each training / baseline run writes ``results/runs/<run>.json``; ``evaluate.py``
writes ``results/runs/<run>_test.json``. This script merges them into one tidy
comparison table (the same numbers that appear in the README) so the table can
never drift from the actual runs.

    python -m src.summarize
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR, RUNS_DIR, ensure_dirs

COLUMNS = [
    "run_name",
    "stage",
    "model",
    "params_trainable",
    "lr",
    "epochs",
    "batch_size",
    "max_length",
    "amp",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "train_time_s",
    "peak_gpu_alloc_mb",
    "checkpoint",
    "notes",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        if path.name.endswith(("_test.json", "_val.json")):
            continue  # split-level evaluation files, not training runs
        record = _load(path)
        name = record.get("run_name", path.stem)
        if name.startswith("smoke_"):
            continue  # smoke tests are debugging runs, not experiments
        cfg = record.get("config", {})
        test = record.get("test") or {}
        test_path = RUNS_DIR / f"{name}_test.json"
        if not test and test_path.exists():
            test = _load(test_path)
        val = record.get("best_val", {})
        stage = record.get("stage", "finetune")
        is_baseline = stage in {"baseline", "reference"}
        rows.append(
            {
                "run_name": name,
                "stage": stage,
                "model": record.get("model_name"),
                "params_trainable": record.get("parameters", {}).get("trainable", ""),
                "lr": "n/a" if is_baseline else cfg.get("lr"),
                "epochs": "n/a" if is_baseline else cfg.get("epochs"),
                "batch_size": "n/a" if is_baseline else cfg.get("batch_size"),
                "max_length": "n/a" if is_baseline else cfg.get("max_length"),
                "amp": "n/a" if is_baseline else record.get("amp_dtype", cfg.get("amp_dtype", "")),
                "val_accuracy": val.get("val_accuracy"),
                "val_macro_f1": val.get("val_macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "test_weighted_f1": test.get("weighted_f1"),
                "train_time_s": record.get("train_time_s") or record.get("total_time_s"),
                "peak_gpu_alloc_mb": record.get("peak_gpu_alloc_mb", ""),
                "checkpoint": record.get("checkpoint", ""),
                "notes": "" if not is_baseline else f"best C={record.get('best_C')}; n_features={record.get('n_features')}",
            }
        )

    # Trivial reference: always predict the majority class (neutral).
    baseline_path = RUNS_DIR / "baseline_tfidf_logreg.json"
    if baseline_path.exists():
        trivial = _load(baseline_path).get("trivial_majority", {})
        if trivial:
            rows.append(
                {
                    "run_name": "trivial_majority_class",
                    "stage": "reference",
                    "model": f"always predict '{trivial['label']}'",
                    "params_trainable": 0,
                    "lr": "n/a",
                    "epochs": "n/a",
                    "batch_size": "n/a",
                    "max_length": "n/a",
                    "amp": "n/a",
                    "val_accuracy": trivial.get("val_accuracy"),
                    "val_macro_f1": 0.0,
                    "test_accuracy": trivial.get("test_accuracy"),
                    "test_macro_f1": trivial.get("test_macro_f1"),
                    "test_weighted_f1": "",
                    "train_time_s": 0.0,
                    "peak_gpu_alloc_mb": "",
                    "checkpoint": "",
                    "notes": "sanity floor: predicts the most common class for every input",
                }
            )

    order = {"reference": 0, "baseline": 1, "finetune": 2}
    rows.sort(key=lambda r: (order.get(r["stage"], 3), -(r["val_macro_f1"] or 0)))
    return rows


def main() -> None:
    ensure_dirs()
    rows = build_rows()
    out = RESULTS_DIR / "experiments.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for row in rows:
        val_f1 = row["val_macro_f1"]
        test_acc = row["test_accuracy"]
        test_f1 = row["test_macro_f1"]

        def fmt(value: Any) -> str:
            return "   -  " if value in (None, "") else f"{float(value):.4f}"

        print(f"  {row['run_name']:<24} val_f1={fmt(val_f1)} test_acc={fmt(test_acc)} test_f1={fmt(test_f1)}")


if __name__ == "__main__":
    main()
