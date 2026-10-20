"""Model + tokenizer loading helpers (kept trivial so training code stays readable)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import NUM_LABELS


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer that belongs to ``model_name``."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # DistilBERT has no separate pad token; every model we use here already has one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name: str, num_labels: int = NUM_LABELS, device: torch.device | None = None) -> PreTrainedModel:
    """Load a pretrained encoder with a fresh classification head."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    # Some checkpoints ship with a pad token id of -1; make it explicit.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = 0
    if device is not None:
        model.to(device)
    return model


def count_parameters(model: PreTrainedModel) -> dict[str, int]:
    """Total and trainable parameter counts (for the README / demo)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def model_accepts_token_type_ids(model: PreTrainedModel) -> bool:
    """DistilBERT/RoBERTa do not use token_type_ids; BERT does."""
    return bool(getattr(model.config, "type_vocab_size", 0))


def forward_batch(model: PreTrainedModel, batch: dict[str, torch.Tensor]) -> Any:
    """Call the model with only the keys it understands."""
    if "token_type_ids" in batch and not model_accepts_token_type_ids(model):
        batch = {k: v for k, v in batch.items() if k != "token_type_ids"}
    return model(**batch)
