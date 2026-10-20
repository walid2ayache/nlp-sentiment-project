"""Inference helper used by ``app.py`` and the CLI.

    python -m src.predict --text "I love this so much" --text "worst day ever"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import CKPT_DIR, LABEL_NAMES, RUNS_DIR, pick_device, read_json
from .evaluate import predict_loader
from .model import build_model, load_tokenizer


class SentimentPredictor:
    """Loads one fine-tuned checkpoint and exposes ``predict`` / ``predict_proba``."""

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        self.device = device or pick_device()
        # The base checkpoint is loaded first and then overwritten by our
        # fine-tuned weights, so transformers prints a noisy "MISSING/UNEXPECTED"
        # load report on every start. Silence it for a clean demo/CLI log.
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name: str = ckpt["model_name"]
        self.label_names: tuple[str, ...] = tuple(ckpt.get("label_names", LABEL_NAMES))
        self.max_length: int = int(ckpt["config"].get("max_length", 128))
        self.amp_dtype: str = ckpt["config"].get("amp_dtype", "fp16")
        self.train_config: dict[str, Any] = dict(ckpt["config"])
        self.val_metrics: dict[str, Any] = dict(ckpt.get("val_metrics") or {})
        self.best_epoch: int | None = ckpt.get("epoch")
        self.tokenizer = load_tokenizer(self.model_name)
        self.model = build_model(self.model_name, num_labels=len(self.label_names), device=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.run_name = self.checkpoint_path.parent.name
        self.test_metrics: dict[str, Any] | None = None
        test_path = RUNS_DIR / f"{self.run_name}_test.json"
        if test_path.exists():
            self.test_metrics = read_json(test_path)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return f"GPU - {torch.cuda.get_device_name(0)}"
        return "CPU"

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Class probabilities, shape ``(n_texts, n_classes)``."""
        if not texts:
            return np.zeros((0, len(self.label_names)), dtype=np.float32)
        return predict_loader(
            self.model,
            list(texts),
            self.tokenizer,
            self.max_length,
            self.device,
            batch_size=32,
            amp_dtype=self.amp_dtype,
        )

    def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """One dict per input: predicted label, confidence and all probabilities."""
        probs = self.predict_proba(texts)
        results = []
        for row in probs:
            idx = int(row.argmax())
            results.append(
                {
                    "label": self.label_names[idx],
                    "confidence": float(row[idx]),
                    "probabilities": {name: float(p) for name, p in zip(self.label_names, row)},
                }
            )
        return results


def default_checkpoint() -> Path:
    """Pick the checkpoint of the run with the best validation macro-F1."""
    candidates: list[tuple[float, Path]] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        if path.name.endswith("_test.json") or path.name.startswith(("smoke_", "baseline")):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "best_val" in payload:
            candidates.append((payload["best_val"]["val_macro_f1"], Path(payload["checkpoint"])))
    if not candidates:
        # Fall back to any checkpoint directory that exists.
        fallback = sorted(CKPT_DIR.glob("*/best.pt"))
        if fallback:
            return fallback[0]
        raise FileNotFoundError("no fine-tuned checkpoint found; run `python -m src.train` first")
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for raw text.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--text", action="append", default=None, help="Repeatable input string")
    args = parser.parse_args()

    ckpt = args.checkpoint or default_checkpoint()
    predictor = SentimentPredictor(ckpt)
    texts = args.text or ["I absolutely love this!", "This is the worst thing I have ever seen."]
    for text, result in zip(texts, predictor.predict(texts)):
        probs = " ".join(f"{k}={v:.3f}" for k, v in result["probabilities"].items())
        print(f"{result['label']:>8} ({result['confidence']:.1%}) | {text}\n          {probs}")


if __name__ == "__main__":
    main()
