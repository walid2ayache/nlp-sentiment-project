"""Error-analysis helper: bucket the model's misclassified tweets into patterns.

This script does NOT write the final report - it prepares candidates so the
analysis in ``results/error_analysis.md`` is based on real examples instead of
made-up ones. It prints, per heuristic bucket, the highest-confidence errors
(text, true label, predicted label, confidence) plus the confusion-pair counts.

    python -m src.error_analysis --csv results/errors_<run>_test.csv
    python -m src.error_analysis --csv ... --out results/error_candidates.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RESULTS_DIR

PATTERNS: list[tuple[str, re.Pattern[str] | None]] = [
    ("Negation", re.compile(r"\b(not|n't|never|no|none|nobody|nothing|without|dont|don't|isnt|isn't|wasnt|wasn't|aint|ain't)\b", re.I)),
    ("Contrast / mixed sentiment", re.compile(r"\b(but|however|although|though|yet|except|otherwise|still)\b", re.I)),
    ("Sarcasm / irony cues", re.compile(r"(yeah right|thanks a lot|just what|oh great|great\.|wow|love that|nice one|smh|lol|/s\b|not bad|sure\.|can't wait)", re.I)),
    ("Hashtags / mentions / links", re.compile(r"(#\w+|@\w+|https?://)")),
    ("Very short (<= 5 words)", None),
]


def confusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count errors per (true, predicted) pair."""
    pairs = (
        df.groupby(["true_label", "pred_label"])
        .agg(n=("text", "size"), mean_confidence=("confidence", "mean"), median_words=("n_words", "median"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    pairs["mean_confidence"] = pairs["mean_confidence"].round(3)
    return pairs


def assign_groups(row: pd.Series) -> list[str]:
    """Heuristic bucket labels for one error example."""
    groups: list[str] = []
    for name, pattern in PATTERNS:
        if pattern is None:
            if row["n_words"] <= 5:
                groups.append(name)
        elif pattern.search(str(row["text"])):
            groups.append(name)
    return groups or ["No obvious cue"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="errors_*.csv produced by src.evaluate")
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    lines: list[str] = [
        f"# Error candidates - `{args.csv.name}`",
        "",
        f"{len(df):,} misclassified examples. Mean confidence of an error: {df['confidence'].mean():.3f}.",
        "",
        "## Confusion pairs",
        "",
        "| true -> predicted | n | mean confidence | median words |",
        "|---|---|---|---|",
    ]
    for _, row in confusion_summary(df).iterrows():
        lines.append(
            f"| {row['true_label']} -> {row['pred_label']} | {row['n']} | {row['mean_confidence']:.3f} | {row['median_words']:.0f} |"
        )

    lines += ["", "## High-confidence errors (>= 0.90)", ""]
    high = df[df["confidence"] >= 0.9].sort_values("confidence", ascending=False).head(25)
    for _, row in high.iterrows():
        lines.append(
            f"- `{row['confidence']:.2f}` **{row['true_label']} -> {row['pred_label']}** - \"{row['text']}\""
        )

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in PATTERNS}
    for _, row in df.iterrows():
        for group in assign_groups(row):
            if group in grouped and len(grouped[group]) < args.per_group * 2:
                grouped[group].append(row.to_dict())

    for name, examples in grouped.items():
        lines += ["", f"## Heuristic bucket: {name} ({len(examples)} shown)", ""]
        for row in sorted(examples, key=lambda r: -float(r["confidence"]))[: args.per_group]:
            lines.append(
                f"- `{row['confidence']:.2f}` **{row['true_label']} -> {row['pred_label']}** - \"{row['text']}\""
            )

    report = "\n".join(lines) + "\n"
    out = args.out or RESULTS_DIR / "error_candidates.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
