"""Dataset loading, splitting, EDA and PyTorch ``DataLoader`` construction.

Dataset: TweetEval sentiment (Barbieri et al., 2020)
---------------------------------------------------
* 3 classes: negative / neutral / positive
* ~45k training tweets, 2k validation, 12k test (official TweetEval splits)
* short, noisy, real-world text -> honest error analysis

We merge TweetEval's train+validation splits and re-split them 90/10 with a
fixed seed. The official *test* split is kept completely untouched and is only
touched once, in ``evaluate.py``, by the model selected on validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_NAME,
    FIG_DIR,
    LABEL_NAMES,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
    DataConfig,
    ensure_dirs,
    write_json,
)


# ---------------------------------------------------------------------------
# 1. Raw data -> pandas splits
# ---------------------------------------------------------------------------
def _load_hf_dataset():
    """Load TweetEval sentiment from the Hugging Face Hub.

    Primary path: ``load_dataset("tweet_eval", "sentiment")``.

    Fallback: ``datasets>=5`` + ``huggingface_hub>=2`` reject *namespace-less*
    dataset ids such as ``tweet_eval`` ("Repository id must be 'namespace/name'").
    In that case we download the exact same official parquet files from the Hub
    into ``data/raw`` and hand them to the local parquet loader. Same data, same
    splits - only the download path differs.
    """
    import urllib.request

    from datasets import load_dataset  # imported lazily: only needed here

    try:
        return load_dataset(DATASET_NAME, DATASET_CONFIG)
    except Exception as exc:  # pragma: no cover - depends on library versions
        print(f"[warn] load_dataset({DATASET_NAME!r}) failed ({type(exc).__name__}); "
              "falling back to the official parquet files.")
        base = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/{DATASET_CONFIG}/"
        raw_dir = DATA_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for split in ("train", "validation", "test"):
            path = raw_dir / f"{split}.parquet"
            if not path.exists():
                urllib.request.urlretrieve(base + f"{split}-00000-of-00001.parquet", path)
            files[split] = str(path)
        return load_dataset("parquet", data_files=files)


def load_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    """Return ``{"train", "val", "test"}`` DataFrames with ``text``/``label``.

    The splits are cached as parquet in ``data/splits/`` so every later phase
    is fast and deterministic even without network access.
    """
    ensure_dirs()
    paths = {name: SPLITS_DIR / f"{name}.parquet" for name in ("train", "val", "test")}
    if not force and all(p.exists() for p in paths.values()):
        return {name: pd.read_parquet(path) for name, path in paths.items()}

    raw = _load_hf_dataset()
    train_full = pd.concat([raw["train"].to_pandas(), raw["validation"].to_pandas()], ignore_index=True)
    train_full = train_full[["text", "label"]].dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stratified hold-out for validation: class proportions stay identical.
    train_df, val_df = train_test_split(
        train_full,
        test_size=DataConfig().val_size,
        random_state=SEED,
        stratify=train_full["label"],
    )
    test_df = raw["test"].to_pandas()[["text", "label"]].reset_index(drop=True)

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for name, df in splits.items():
        df.to_parquet(paths[name], index=False)
    return splits


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")


def clean_for_baseline(text: str) -> str:
    """Light normalisation used by the TF-IDF baseline only.

    The transformer sees the *raw* tweet because BERT-style models already
    handle @mentions/URLs well and their casing carries sentiment signal.
    TF-IDF, on the other hand, would happily memorise individual URLs.
    """
    text = URL_RE.sub(" urltoken ", text)
    text = MENTION_RE.sub(" usertoken ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)  # keep the hashtag word, drop the '#'
    text = text.lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def length_stats(texts: Iterable[str]) -> dict[str, float]:
    """Word-count summary for a list of raw strings."""
    words = np.array([len(t.split()) for t in texts])
    return {
        "mean_words": float(words.mean()),
        "median_words": float(np.median(words)),
        "p90_words": float(np.percentile(words, 90)),
        "p99_words": float(np.percentile(words, 99)),
        "max_words": int(words.max()),
    }


def run_eda(splits: dict[str, pd.DataFrame], tokenizer=None, max_length: int = 96) -> dict[str, Any]:
    """Compute dataset statistics, save plots and a markdown summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    summary: dict[str, Any] = {"dataset": f"{DATASET_NAME}/{DATASET_CONFIG}", "label_names": list(LABEL_NAMES)}
    for name, df in splits.items():
        counts = df["label"].value_counts().sort_index()
        summary[name] = {
            "n": int(len(df)),
            "class_counts": {LABEL_NAMES[i]: int(c) for i, c in counts.items()},
            "class_pct": {LABEL_NAMES[i]: round(100 * c / len(df), 2) for i, c in counts.items()},
            "duplicates": int(df["text"].duplicated().sum()),
            **length_stats(df["text"]),
        }

    # --- plot 1: class balance per split -----------------------------------
    rows = []
    for name, df in splits.items():
        vc = df["label"].value_counts(normalize=True).sort_index()
        for label_id, pct in vc.items():
            rows.append({"split": name, "label": LABEL_NAMES[label_id], "pct": 100 * pct})
    balance = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=balance, x="split", y="pct", hue="label", ax=ax)
    ax.set_title("Class balance per split (TweetEval sentiment)")
    ax.set_ylabel("% of split")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=9, padding=2)
    ax.set_ylim(0, 62)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_class_balance.png", dpi=150)
    plt.close(fig)

    # --- plot 2: word-length distribution + per-class boxplot --------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_lengths = pd.DataFrame(
        {
            "words": [len(t.split()) for df in splits.values() for t in df["text"]],
            "label": [LABEL_NAMES[i] for df in splits.values() for i in df["label"]],
        }
    )
    sns.histplot(all_lengths["words"].clip(upper=60), bins=40, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Tweet length in words (clipped at 60)")
    axes[0].set_xlabel("words")
    sns.boxplot(data=all_lengths, x="label", y="words", ax=axes[1], showfliers=False)
    axes[1].set_title("Word count by sentiment class")
    axes[1].set_ylabel("words")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_length_distribution.png", dpi=150)
    plt.close(fig)

    # --- tokenizer coverage / truncation -----------------------------------
    if tokenizer is not None:
        sample = pd.concat(splits.values()).sample(n=min(4000, sum(len(d) for d in splits.values())), random_state=SEED)
        token_lengths = [
            len(ids) for ids in tokenizer(list(sample["text"]), truncation=False, padding=False)["input_ids"]
        ]
        token_lengths = np.asarray(token_lengths)
        summary["token_lengths"] = {
            "max_length_used": max_length,
            "mean_tokens": round(float(token_lengths.mean()), 2),
            "median_tokens": float(np.median(token_lengths)),
            "p95_tokens": float(np.percentile(token_lengths, 95)),
            "p99_tokens": float(np.percentile(token_lengths, 99)),
            "pct_truncated": round(float((token_lengths > max_length).mean() * 100), 2),
        }

    write_json(RESULTS_DIR / "eda_summary.json", summary)
    (RESULTS_DIR / "eda_summary.md").write_text(_eda_markdown(summary), encoding="utf-8")
    return summary


def _eda_markdown(summary: dict[str, Any]) -> str:
    """Render the EDA numbers as a human-readable markdown file."""
    train_pct = summary["train"]["class_pct"]
    test_pct = summary["test"]["class_pct"]
    majority = max(train_pct, key=train_pct.get)
    shift_label = max(train_pct, key=lambda k: abs(test_pct[k] - train_pct[k]))
    shift_pp = test_pct[shift_label] - train_pct[shift_label]
    lines = [
        "# EDA - TweetEval sentiment",
        "",
        f"Dataset: `{summary['dataset']}`  |  classes: {', '.join(summary['label_names'])}",
        "",
        "| split | n | negative | neutral | positive | median words | p99 words |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        s = summary[name]
        c = s["class_counts"]
        lines.append(
            f"| {name} | {s['n']:,} | {c['negative']:,} ({s['class_pct']['negative']}%) | "
            f"{c['neutral']:,} ({s['class_pct']['neutral']}%) | "
            f"{c['positive']:,} ({s['class_pct']['positive']}%) | {s['median_words']:.0f} | {s['p99_words']:.0f} |"
        )
    lines += [
        "",
        "Plots: `results/figures/eda_class_balance.png`, `results/figures/eda_length_distribution.png`.",
        "",
        "## Observations",
        "",
        f"1. **Class imbalance** - `{majority}` is the most common training class "
        f"({train_pct[majority]}%). Accuracy alone would therefore be misleading, so "
        "**macro-F1 is the headline metric** (all three classes weigh the same) and "
        "accuracy is reported next to it.",
        f"2. **Label shift between validation and test** - the official test split is *not* "
        f"drawn from the same label distribution: `{shift_label}` is {train_pct[shift_label]}% of train "
        f"but {test_pct[shift_label]}% of test ({shift_pp:+.1f} percentage points). This is a property of "
        "TweetEval, not of our split. It means test scores are systematically lower than "
        "validation scores, and it is one of the limitations discussed in the README. "
        "We do **not** re-balance the test set: that would be tuning on test.",
        f"3. **Short texts** - the median tweet is {summary['train']['median_words']:.0f} words and 99% are at most "
        f"{summary['train']['p99_words']:.0f} words, so almost nothing is truncated at 96 tokens; tokenisation is "
        "cheap and each epoch is fast.",
        "4. **Noisy language** - @mentions, URLs, hashtags, slang and typos mean a purely "
        "lexical model (TF-IDF) will miss sarcasm and negation; that is the gap the "
        "transformer is expected to close.",
        f"5. **Duplicates** - {summary['train']['duplicates']} duplicate training texts were removed before splitting.",
        "",
    ]
    if "token_lengths" in summary:
        t = summary["token_lengths"]
        lines += [
            "## Tokenizer coverage",
            "",
            f"On a 4,000-tweet sample the tokenizer produced a median of "
            f"{t['median_tokens']:.0f} tokens (p95 = {t['p95_tokens']:.0f}, p99 = {t['p99_tokens']:.0f}). "
            f"With `max_length={t['max_length_used']}`, **{t['pct_truncated']}%** of tweets get truncated.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Torch datasets / collator / loaders
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    """Tokenised tweets. ``__getitem__`` returns one unpadded example."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int) -> None:
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        self.input_ids: list[list[int]] = enc["input_ids"]
        self.attention_mask: list[list[int]] = enc["attention_mask"]
        # Some tokenizers (BERT-style) also emit token_type_ids.
        self.token_type_ids: list[list[int]] | None = enc.get("token_type_ids")
        self.labels: list[int] = [int(label) for label in labels]
        assert len(self.input_ids) == len(self.labels), "tokenizer dropped examples"

    def __len__(self) -> int:  # noqa: D105
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


@dataclass
class PadCollator:
    """Dynamic padding: pad each batch only to its own longest sequence.

    This is meaningfully faster than padding every tweet to ``max_length``
    because most tweets are far shorter than the limit.
    """

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(np.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        batch: dict[str, torch.Tensor] = {}
        for key, pad_value in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("token_type_ids", 0)):
            if key not in features[0]:
                continue
            out = torch.full((len(features), max_len), pad_value, dtype=torch.long)
            for row, feat in enumerate(features):
                ids = torch.as_tensor(feat[key], dtype=torch.long)
                out[row, : len(ids)] = ids
            batch[key] = out
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch


def build_dataloaders(
    splits: dict[str, pd.DataFrame],
    tokenizer,
    cfg: DataConfig,
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
    subset: int | None = None,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders with dynamic padding."""
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id or 0)
    datasets: dict[str, TweetDataset] = {}
    for name, df in splits.items():
        if subset is not None and name != "test":
            df = df.sample(n=min(subset, len(df)), random_state=cfg.seed).reset_index(drop=True)
        datasets[name] = TweetDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, cfg.max_length)

    # A dedicated generator makes shuffling reproducible across runs.
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    pin = device.type == "cuda"

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            drop_last=False,
            generator=generator,
            persistent_workers=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        ),
    }


if __name__ == "__main__":  # pragma: no cover - manual EDA entry point
    from transformers import AutoTokenizer

    from .config import set_seed

    set_seed()
    splits_ = load_splits(force="--force" in __import__("sys").argv)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    stats = run_eda(splits_, tokenizer=tok)
    print(f"train={stats['train']['n']:,} val={stats['val']['n']:,} test={stats['test']['n']:,}")
    print("Wrote results/eda_summary.md and results/figures/eda_*.png")
