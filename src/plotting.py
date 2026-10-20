"""Shared plotting / reporting helpers (confusion matrix, classification report)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: we only ever save figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
    normalize: bool = True,
) -> pd.DataFrame:
    """Save a confusion matrix (counts + row-normalised percentages) and return raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = f"{cm_norm[i, j]:.1%}\n({cm[i, j]:,})"
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm_norm[i, j] > 0.55 else "#1a1a1a",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return pd.DataFrame(cm, index=list(labels), columns=list(labels))


def classification_report_df(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1 plus support, as a DataFrame."""
    report = classification_report(
        y_true, y_pred, labels=list(range(len(labels))), target_names=list(labels), output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T
    df = df.rename(index={"accuracy": "accuracy", "macro avg": "macro avg", "weighted avg": "weighted avg"})
    return df


def save_classification_report_md(report: pd.DataFrame, title: str, path: Path) -> None:
    """Write the sklearn classification report as a github-flavoured markdown table."""
    cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in report.columns]
    lines = [f"# {title}", "", "| class | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for index, row in report.iterrows():
        values = []
        for col in cols:
            val = row[col]
            values.append(f"{val:,}" if col == "support" else f"{val:.3f}")
        lines.append(f"| {index} | " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
