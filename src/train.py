"""Custom PyTorch training loop for TweetEval sentiment - no ``transformers.Trainer``.

Read this file top-to-bottom: every block is commented so that the whole
training procedure can be explained line by line in an interview.

Pipeline
--------
1. fixed seeds + device selection (CUDA required, CPU fallback with a warning)
2. load cached stratified splits, tokenize, build DataLoaders
3. load a pretrained encoder + fresh 3-way classification head
4. AdamW with weight decay skipped on biases/LayerNorm
5. linear warmup -> linear decay learning-rate schedule
6. mixed precision (fp16 + GradScaler, or bf16 without scaler)
7. per-epoch validation, best-checkpoint saving on val macro-F1, early stopping
8. everything logged to local MLflow (``mlruns/``) and ``results/runs/<run>.json``

The test split is intentionally *not* touched here: the model is selected on
validation and evaluated on test exactly once in ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .config import (
    CKPT_DIR,
    EXPERIMENTS,
    FIG_DIR,
    LABEL_NAMES,
    MLFLOW_EXPERIMENT,
    MLRUNS_DIR,
    ROOT,
    RUNS_DIR,
    DataConfig,
    TrainConfig,
    dataclass_dict,
    ensure_dirs,
    gpu_info,
    pick_device,
    set_seed,
    write_json,
)
from .data import build_dataloaders, load_splits
from .model import build_model, count_parameters, forward_batch, load_tokenizer

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """AdamW with no weight decay on biases and LayerNorm weights.

    Decaying norm/bias parameters is a classic small mistake: those parameters
    should not be pulled toward zero, only the weight matrices should.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if any(nd in name for nd in no_decay) else decay_params).append(param)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    """Linear warmup for the first ``warmup_ratio`` of steps, then linear decay to 0.

    Formula used by the scheduler: ``lr = base_lr * min(step / warmup, (total - step) / (total - warmup))``.
    Warmup avoids destroying the pretrained weights with a huge first update.
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0.0, (total_steps - current_step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    """Run one full pass over the training set and return epoch statistics."""
    model.train()
    total_loss = 0.0
    n_examples = 0
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    grad_norms: list[float] = []
    nonfinite_grad_steps = 0
    use_amp = device.type == "cuda"
    amp_dtype = AMP_DTYPES[cfg.amp_dtype]

    pbar = tqdm(loader, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # 1) zero gradients (set_to_none=True is slightly faster / less memory)
        optimizer.zero_grad(set_to_none=True)

        # 2) forward pass in mixed precision
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = forward_batch(model, batch)
            loss = outputs.loss

        # 3) backward pass. GradScaler multiplies the loss by a scale factor so
        #    tiny fp16 gradients do not underflow to zero; for bf16 it is disabled.
        scaler.scale(loss).backward()

        # 4) unscale before clipping so the norm is in real units, then clip
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        # 5) optimizer step through the scaler (it skips steps on inf/nan grads)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 6) bookkeeping
        bs = batch["labels"].size(0)
        total_loss += float(loss.detach().item()) * bs
        n_examples += bs
        preds.append(outputs.logits.detach().argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        gn = float(grad_norm)
        if math.isfinite(gn):
            grad_norms.append(gn)
        else:
            # An fp16 overflow made the scaler skip this step; count it instead of
            # poisoning the epoch mean with inf.
            nonfinite_grad_steps += 1
        if step % cfg.log_every == 0:
            pbar.set_postfix(
                loss=f"{total_loss / n_examples:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return {
        "train_loss": total_loss / n_examples,
        "train_acc": float(accuracy_score(y_true, y_pred)),
        "train_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_nonfinite_steps": nonfinite_grad_steps,
    }


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: str = "fp16",
) -> dict[str, Any]:
    """Evaluate a model; returns loss, accuracy, macro-F1 and raw predictions.

    ``@torch.no_grad()`` disables autograd, which saves both time and memory.
    The model is put in ``eval()`` mode so dropout is off and BatchNorm (if any)
    uses running statistics - forgetting this is a classic silent bug.
    """
    model.eval()
    total_loss, n_examples = 0.0, 0
    all_probs, all_labels = [], []
    use_amp = device.type == "cuda"
    dtype = AMP_DTYPES[amp_dtype]

    for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            outputs = forward_batch(model, batch)
        bs = batch["labels"].size(0)
        total_loss += float(outputs.loss.item()) * bs
        n_examples += bs
        all_probs.append(torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = probs.argmax(axis=1)
    return {
        "loss": total_loss / n_examples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Plotting / tracking helpers
# ---------------------------------------------------------------------------
def plot_curves(history: list[dict[str, float]], run_name: str) -> Path:
    """Save loss + validation-metric curves for the run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    axes[0].set_title(f"{run_name}: loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], marker="o", label="val accuracy")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], marker="o", label="val macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{run_name}: validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    path = FIG_DIR / f"{run_name}_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def git_commit() -> str:
    """Short git commit hash of the working tree (or 'unknown')."""
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for TweetEval sentiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=None,
                        help="Name of a preset from src.config.EXPERIMENTS")
    parser.add_argument("--run-name", default=None, help="MLflow run / checkpoint name")
    parser.add_argument("--model-name", default=None, help="HF model id (e.g. distilbert-base-uncased)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=sorted(AMP_DTYPES), default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="Train/val on a small subset (debugging)")
    parser.add_argument("--smoke", action="store_true", help="Tiny 1-epoch run to catch bugs")
    parser.add_argument("--freeze-encoder", action="store_true", default=None,
                        help="Ablation: freeze the encoder and train only the classification head")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    ensure_dirs()

    # 1) Build the run configuration: preset, then explicit CLI overrides.
    cfg = TrainConfig()
    if args.experiment:
        for key, value in EXPERIMENTS[args.experiment].items():
            setattr(cfg, key, value)
        cfg.run_name = args.experiment
    for key, value in vars(args).items():
        if key in {"experiment", "smoke", "no_mlflow"} or value is None:
            continue
        setattr(cfg, key, value)
    if args.smoke:
        cfg.subset = args.subset or 512
        cfg.epochs = 1
        cfg.run_name = f"smoke_{cfg.run_name}"
        cfg.log_every = 10
    elif args.run_name:
        cfg.run_name = args.run_name

    data_cfg = DataConfig(max_length=cfg.max_length, seed=cfg.seed, num_workers=cfg.num_workers)

    # 2) Reproducibility + hardware
    set_seed(cfg.seed)
    device = pick_device()
    gpu = gpu_info()
    print(f"[info] device={device} | gpu={gpu.get('gpu_name', 'CPU')} | torch={gpu['torch']}")

    # 3) Data
    splits = load_splits()
    print(f"[info] splits: train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")
    tokenizer = load_tokenizer(cfg.model_name)
    loaders = build_dataloaders(
        splits, tokenizer, data_cfg, cfg.batch_size, cfg.eval_batch_size, device, subset=cfg.subset
    )
    print(f"[info] train batches/epoch={len(loaders['train'])} val batches={len(loaders['val'])}")

    # 4) Model
    model = build_model(cfg.model_name, device=device)
    if cfg.freeze_encoder:
        # Ablation: keep the pretrained encoder fixed and learn only the head.
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("[info] encoder frozen - only the classification head is trained")
    n_params = count_parameters(model)
    print(f"[info] parameters: {n_params['trainable']:,} trainable / {n_params['total']:,} total")

    # 5) Optimizer + scheduler (scheduler needs the TOTAL number of update steps)
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = len(loaders["train"]) * cfg.epochs
    scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    # 6) Mixed precision: fp16 needs a GradScaler, bf16 does not (much wider range).
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.amp_dtype == "fp16")
    if cfg.amp_dtype == "bf16" and use_amp and not gpu.get("bf16_supported", False):
        print("[warn] bf16 requested but not supported by this GPU - using fp16 instead")
        cfg.amp_dtype = "fp16"
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    run_dir = CKPT_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_record: dict[str, Any] = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "config": dataclass_dict(cfg),
        "data": dataclass_dict(data_cfg),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "parameters": n_params,
        "gpu": gpu,
        "git_commit": git_commit(),
    }

    # 7) MLflow tracking (local file store -> ./mlruns)
    if not args.no_mlflow:
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_ctx = mlflow.start_run(run_name=cfg.run_name)
        mlflow.log_params({**dataclass_dict(cfg), **dataclass_dict(data_cfg), "params_total": n_params["total"]})
        mlflow.set_tags(
            {
                "dataset": "tweet_eval/sentiment",
                "model": cfg.model_name,
                "gpu": gpu.get("gpu_name", "CPU"),
                "git_commit": run_record["git_commit"],
                "stage": "smoke" if args.smoke else "final",
            }
        )
        mlflow.log_dict(run_record, "run_config.json")
    else:
        run_ctx = None

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, device, cfg, epoch)
        val_stats = run_eval(model, loaders["val"], device, cfg.amp_dtype)
        epoch_time = time.time() - epoch_start
        train_stats["epoch"] = epoch
        train_stats["epoch_time_s"] = epoch_time
        train_stats["val_loss"] = val_stats["loss"]
        train_stats["val_accuracy"] = val_stats["accuracy"]
        train_stats["val_macro_f1"] = val_stats["macro_f1"]
        train_stats["val_weighted_f1"] = val_stats["weighted_f1"]
        train_stats["lr"] = scheduler.get_last_lr()[0]
        train_stats["examples_per_sec"] = len(loaders["train"].dataset) / epoch_time
        if device.type == "cuda":
            train_stats["gpu_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        history.append(train_stats)
        print(
            f"[epoch {epoch}] train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} time={epoch_time:.0f}s"
        )
        if run_ctx is not None:
            mlflow.log_metrics(
                {k: float(v) for k, v in train_stats.items() if k != "epoch"},
                step=epoch,
            )

        # 8) Best-checkpoint selection on VALIDATION macro-F1 (never test).
        if val_stats["macro_f1"] > best_f1:
            best_f1 = val_stats["macro_f1"]
            best_epoch = epoch
            best_metrics = {
                "val_loss": float(val_stats["loss"]),
                "val_accuracy": float(val_stats["accuracy"]),
                "val_macro_f1": float(val_stats["macro_f1"]),
                "val_weighted_f1": float(val_stats["weighted_f1"]),
            }
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": cfg.model_name,
                    "label_names": list(LABEL_NAMES),
                    "config": dataclass_dict(cfg),
                    "val_metrics": best_metrics,
                    "epoch": epoch,
                },
                run_dir / "best.pt",
            )
            tokenizer.save_pretrained(run_dir / "tokenizer")
            print(f"[info] new best model at epoch {epoch} (val macro-F1={best_f1:.4f}) -> {run_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"[info] early stopping: no val macro-F1 improvement for {cfg.patience} epochs")
                break

    train_time = time.time() - train_start
    peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else 0.0
    peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**2, 1) if device.type == "cuda" else 0.0
    print(f"[done] trained in {train_time:.0f}s | peak GPU alloc={peak_alloc} MB reserved={peak_reserved} MB")

    # 9) Save the run record (the source of results/experiments.csv) and artifacts.
    run_record.update(
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_val": best_metrics,
            "train_time_s": round(train_time, 1),
            "peak_gpu_alloc_mb": peak_alloc,
            "peak_gpu_reserved_mb": peak_reserved,
            "checkpoint": (run_dir / "best.pt").relative_to(ROOT).as_posix(),
            "amp_dtype": cfg.amp_dtype,
            "freeze_encoder": cfg.freeze_encoder,
            "n_train": len(loaders["train"].dataset),
            "n_val": len(loaders["val"].dataset),
        }
    )
    write_json(RUNS_DIR / f"{cfg.run_name}.json", run_record)
    curves_path = plot_curves(history, cfg.run_name)

    if run_ctx is not None:
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.log_metrics({"train_time_s": train_time, "peak_gpu_alloc_mb": peak_alloc})
        mlflow.log_artifact(str(curves_path), artifact_path="plots")
        mlflow.log_dict(run_record, "run_record.json")
        mlflow.set_tag("best_epoch", best_epoch)
        mlflow.end_run()

    return run_record


if __name__ == "__main__":
    main()
