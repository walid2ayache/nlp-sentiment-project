"""Evaluate a saved checkpoint on the held-out test split - exactly once.

Produces
--------
* ``results/runs/<run>_test.json`` - all metrics (also merged into experiments.csv)
* ``results/figures/<run>_confusion_matrix.png``
* ``results/<run>_classification_report.md``
* ``results/errors_<run>_test.csv`` - every misclassified test example with
  text, true label, predicted label, confidence and per-class probabilities

Usage
-----
    python -m src.evaluate --checkpoint checkpoints/e1_lr3e-5_len96/best.pt
    python -m src.evaluate --checkpoint ... --split val   # sanity check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import (
    FIG_DIR,
    LABEL_NAMES,
    RESULTS_DIR,
    RUNS_DIR,
    ensure_dirs,
    gpu_info,
    pick_device,
    set_seed,
    write_json,
)
from .data import PadCollator, TweetDataset, load_splits
from .model import build_model, forward_batch, load_tokenizer
from .plotting import (
    classification_report_df,
    plot_confusion_matrix,
    save_classification_report_md,
)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    texts: list[str],
    tokenizer,
    max_length: int,
    device: torch.device,
    batch_size: int = 128,
    amp_dtype: str = "fp16",
) -> np.ndarray:
    """Return a ``(n_examples, n_classes)`` probability matrix for raw texts."""
    model.eval()
    ds = TweetDataset(texts, [0] * len(texts), tokenizer, max_length)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PadCollator(pad_token_id=tokenizer.pad_token_id or 0),
    )
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    probs: list[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = forward_batch(model, batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Test-set evaluation for a fine-tuned checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to best.pt (default: best val macro-F1 across results/runs/*.json)")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-name", default=None, help="Override the run name used for output files")
    parser.add_argument("--no-errors", action="store_true", help="Skip writing the misclassified-examples CSV")
    args = parser.parse_args()

    ensure_dirs()
    set_seed()

    # ---- resolve checkpoint -------------------------------------------------
    if args.checkpoint is None:
        candidates = []
        for path in sorted(RUNS_DIR.glob("*.json")):
            if path.name.endswith("_test.json") or path.name.startswith("smoke_"):
                continue
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            if "best_val" in payload and payload.get("stage") != "baseline":
                candidates.append((payload["best_val"]["val_macro_f1"], Path(payload["checkpoint"])))
        assert candidates, "no fine-tuned checkpoints found - run src.train first"
        _, best_path = max(candidates)
        checkpoint_path = best_path
    else:
        checkpoint_path = args.checkpoint
    print(f"[info] checkpoint: {checkpoint_path}")

    device = pick_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_name = args.run_name or checkpoint_path.parent.name
    tokenizer = load_tokenizer(ckpt["model_name"])
    model = build_model(ckpt["model_name"], num_labels=len(ckpt.get("label_names", LABEL_NAMES)), device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    max_length = int(ckpt["config"].get("max_length", 128))
    amp_dtype = ckpt["config"].get("amp_dtype", "fp16")
    print(f"[info] model={ckpt['model_name']} max_length={max_length} amp={amp_dtype} device={device}")

    # ---- data ---------------------------------------------------------------
    splits = load_splits()
    df = splits[args.split]
    print(f"[info] {args.split} split: {len(df):,} examples")

    t0 = time.time()
    probs = predict_loader(model, df["text"].tolist(), tokenizer, max_length, device, args.batch_size, amp_dtype)
    elapsed = time.time() - t0
    y_true = df["label"].to_numpy()
    y_pred = probs.argmax(axis=1)

    metrics = {
        "split": args.split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_examples": int(len(df)),
        "inference_time_s": round(elapsed, 1),
        "examples_per_sec": round(len(df) / elapsed, 1),
        "model_name": ckpt["model_name"],
        "checkpoint": str(checkpoint_path),
        "best_epoch": ckpt.get("epoch"),
        "val_metrics": ckpt.get("val_metrics"),
        "gpu": gpu_info(),
    }
    print(
        f"[{args.split}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} ({elapsed:.1f}s)"
    )

    # ---- artefacts ----------------------------------------------------------
    report = classification_report_df(y_true, y_pred, LABEL_NAMES)
    tag = f"{run_name}_{args.split}"
    plot_confusion_matrix(
        y_true, y_pred, LABEL_NAMES,
        title=f"{ckpt['model_name']} - {args.split} (macro-F1 {metrics['macro_f1']:.3f})",
        path=FIG_DIR / f"{run_name}_confusion_matrix.png",
    )
    save_classification_report_md(report, f"{run_name} - {args.split} classification report", RESULTS_DIR / f"{tag}_classification_report.md")
    metrics["per_class"] = {
        label: {
            "precision": float(report.loc[label, "precision"]),
            "recall": float(report.loc[label, "recall"]),
            "f1": float(report.loc[label, "f1-score"]),
            "support": int(report.loc[label, "support"]),
        }
        for label in LABEL_NAMES
    }
    write_json(RESULTS_DIR / f"{tag}_metrics.json", metrics)
    write_json(RUNS_DIR / f"{run_name}_{args.split}.json", metrics)

    # ---- misclassified examples -------------------------------------------
    if not args.no_errors:
        confidence = probs.max(axis=1)
        errors = pd.DataFrame(
            {
                "text": df["text"].to_numpy(),
                "true_label": [LABEL_NAMES[i] for i in y_true],
                "pred_label": [LABEL_NAMES[i] for i in y_pred],
                "confidence": confidence.round(4),
                "p_negative": probs[:, 0].round(4),
                "p_neutral": probs[:, 1].round(4),
                "p_positive": probs[:, 2].round(4),
                "n_words": [len(t.split()) for t in df["text"]],
            }
        )
        wrong = errors[errors["true_label"] != errors["pred_label"]].sort_values("confidence", ascending=False)
        wrong.to_csv(RESULTS_DIR / f"errors_{run_name}_{args.split}.csv", index=False, encoding="utf-8")
        print(f"[info] {len(wrong):,} misclassified examples -> results/errors_{run_name}_{args.split}.csv")

    # ---- confidence plot ---------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct_conf = probs.max(axis=1)[y_true == y_pred]
    wrong_conf = probs.max(axis=1)[y_true != y_pred]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.3, 1.0, 30)
    ax.hist(correct_conf, bins=bins, alpha=0.65, label=f"correct (n={len(correct_conf):,})", color="#2a9d8f")
    ax.hist(wrong_conf, bins=bins, alpha=0.65, label=f"wrong (n={len(wrong_conf):,})", color="#e76f51")
    ax.set_xlabel("max softmax probability (confidence)")
    ax.set_ylabel("count")
    ax.set_title(f"{run_name}: confidence of correct vs wrong predictions ({args.split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{run_name}_confidence.png", dpi=150)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    main()
