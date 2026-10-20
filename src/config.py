"""Central configuration: paths, dataset choice, hyper-parameters and seeds.

Everything that another module might want to tweak lives here so that
``train.py``, ``evaluate.py`` and ``app.py`` cannot drift apart.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root so the project is portable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"
CKPT_DIR = ROOT / "checkpoints"
MLRUNS_DIR = ROOT / "mlruns"
SPLITS_DIR = DATA_DIR / "splits"

# ---------------------------------------------------------------------------
# Dataset choice: TweetEval sentiment (3 classes, short noisy tweets)
# ---------------------------------------------------------------------------
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"
LABEL_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")
NUM_LABELS = len(LABEL_NAMES)

SEED = 42
MLFLOW_EXPERIMENT = "tweet_eval_sentiment"


@dataclass
class DataConfig:
    """How the raw dataset is turned into train/val/test splits."""

    max_length: int = 96          # tokens kept per example (truncate the rest)
    val_size: float = 0.10        # fraction of train+validation used for val
    seed: int = SEED
    num_workers: int = 0          # Windows: 0 avoids DataLoader spawn issues


@dataclass
class TrainConfig:
    """Hyper-parameters for one fine-tuning run."""

    run_name: str = "distilbert_lr3e-5"
    model_name: str = "distilbert-base-uncased"
    lr: float = 3e-5
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    max_length: int = 96
    amp_dtype: str = "fp16"        # "fp16" (GradScaler) or "bf16" (no scaler)
    patience: int = 2
    seed: int = SEED
    num_workers: int = 0
    subset: int | None = None      # tiny subset for smoke tests / debugging
    log_every: int = 50
    freeze_encoder: bool = False    # ablation: train only the classification head


@dataclass
class BaselineConfig:
    """TF-IDF + Logistic Regression baseline hyper-parameters."""

    run_name: str = "baseline_tfidf_logreg"
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    max_features_word: int = 200_000
    max_features_char: int = 200_000
    C_values: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    max_iter: int = 2000
    seed: int = SEED


# Ready-made experiment presets. ``python -m src.train --experiment e2_...``
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "e1_lr3e-5": dict(model_name="distilbert-base-uncased", lr=3e-5, epochs=3, max_length=96, batch_size=32),
    "e2_lr2e-5_4ep": dict(model_name="distilbert-base-uncased", lr=2e-5, epochs=4, max_length=96, batch_size=32),
    "e3_lr5e-5": dict(model_name="distilbert-base-uncased", lr=5e-5, epochs=3, max_length=96, batch_size=32),
    "e4_roberta_lr2e-5": dict(model_name="roberta-base", lr=2e-5, epochs=3, max_length=96, batch_size=32),
    "e5_frozen_encoder": dict(model_name="distilbert-base-uncased", lr=1e-3, epochs=3, max_length=96,
                              batch_size=32, freeze_encoder=True),
}


def ensure_dirs() -> None:
    """Create every output directory the project writes to."""
    for path in (DATA_DIR, RESULTS_DIR, FIG_DIR, RUNS_DIR, CKPT_DIR, MLRUNS_DIR, SPLITS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + GPU) for reproducibility.

    ``cudnn.deterministic`` makes cuDNN pick deterministic kernels. It costs a
    little speed, which is a fine trade-off for a portfolio project where
    results must be reproducible run-to-run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """Return CUDA if available, otherwise CPU (with a loud warning)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[WARN] CUDA is not available - falling back to CPU. Training will be slow.")
    return torch.device("cpu")


def gpu_info() -> dict[str, Any]:
    """Snapshot of the GPU/environment, logged to MLflow and saved to disk."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_gb": round(props.total_memory / 1024**3, 2),
                "gpu_capability": f"{props.major}.{props.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parents if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dataclass_dict(obj: Any) -> dict[str, Any]:
    """``asdict`` that keeps tuples as lists (JSON friendly)."""
    return json.loads(json.dumps(asdict(obj), default=str))
