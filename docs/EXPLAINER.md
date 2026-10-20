# How this project works — a plain-language walkthrough

This document explains the whole pipeline in the order the code runs, so you can
walk an interviewer through it without memorising anything. File references are
real paths in this repo.

---

## 0. The 30-second pitch

> I fine-tuned transformers for 3-class sentiment classification on TweetEval
> (negative / neutral / positive). I wrote the PyTorch training loop myself —
> mixed precision, AdamW with decoupled weight decay, linear warmup + decay,
> gradient clipping, per-epoch validation, best-checkpoint selection and early
> stopping — and tracked every run in MLflow. A TF-IDF + logistic-regression
> baseline gets **0.597** macro-F1 on the held-out test set; the best fine-tuned
> model (RoBERTa-base, selected on validation) gets **0.709**. The baseline has
> the same splits and a 5-point C grid tuned on validation, so the +0.112 is a
> fair comparison. I then did an error analysis on 24 real misclassified tweets
> and shipped a Gradio demo.

(All numbers are the actual test results from `results/experiments.csv`.)

---

## 1. Data (`src/data.py`)

1. `load_dataset("tweet_eval", "sentiment")` downloads the official splits:
   45k train, 2k validation, 12k test tweets with 3 labels.
2. I **merge train + validation** and re-split them 90/10 with `random_state=42`,
   stratified on the label (42,827 train / 4,759 validation). Why: the official
   validation split is tiny (2k), which makes model selection noisy. The official
   **test split (12,284) is untouched** and is used once, at the very end, by the
   model chosen on validation.
3. Splits are cached as parquet in `data/splits/` so every run is identical and
   fast, without re-downloading.
4. `TweetDataset` tokenizes each tweet with the model's own tokenizer,
   truncating to `max_length` tokens. No padding yet.
5. `PadCollator` pads **each batch** only to the longest sequence in that batch
   ("dynamic padding"). A tweet with 8 tokens wastes almost nothing, unlike
   padding everything to 96 tokens up front.
6. `DataLoader` shuffles the training set with a seeded generator; validation and
   test are never shuffled.

**Class imbalance.** Neutral is the smallest class. That is why the headline
metric is **macro-F1** (each class counts equally) and not accuracy (dominated
by the majority class). Accuracy is still reported alongside.

---

## 2. Baseline (`src/baseline.py`)

TF-IDF over word 1–2 grams **and** character 3–5 grams, concatenated, feeding a
multinomial logistic regression.

* Word n-grams capture vocabulary and simple negation ("not good").
* Character n-grams make it robust to the typos/slang that dominate tweets.
* Features are fit on **train only**; `C` is chosen on **validation** macro-F1;
  the chosen model is scored on test once.

Why bother? Because a transformer is only worth its cost if it beats a strong
linear model on the same split. The gap between the two is the actual story of
the project.

---

## 3. The training loop (`src/train.py`) — the important part

### 3.1 Setup

```python
set_seed(cfg.seed)          # Python, NumPy, torch, CUDA + deterministic cuDNN
device = pick_device()      # CUDA if available, loud warning + CPU fallback
model = build_model(...)    # pretrained encoder + fresh 3-way classification head
```

The classification head is randomly initialised, so the first epoch is a
"warm-up" for the head while the encoder only gets gently nudged. DistilBERT and
RoBERTa are interchangeable here — only `--model-name` changes.

### 3.2 Optimizer

```python
AdamW([
    {"params": decay_params,    "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-5)
```

Weight decay is applied to weight matrices only. Biases and LayerNorm gains are
**not** decayed — pulling them toward zero is a small but real bug. AdamW
decouples weight decay from the gradient update (unlike Adam + L2), which is why
it is the default for transformer fine-tuning.

### 3.3 Learning-rate schedule

```python
total_steps = len(train_loader) * epochs
warmup_steps = 0.1 * total_steps
lr(step) = base_lr * min(step / warmup, (total - step) / (total - warmup))
```

A linear warmup for the first 10% of updates, then linear decay to zero.
Warmup matters because the new classification head produces large gradients at
the start; a big first update could damage the pretrained weights. Decay lets the
model settle into a minimum at the end.

### 3.4 One training step (the heart of it)

```python
optimizer.zero_grad(set_to_none=True)          # 1. clear old gradients
with torch.autocast("cuda", dtype=torch.float16):
    outputs = model(**batch)                   # 2. forward in mixed precision
    loss = outputs.loss                        #    cross-entropy, mean over batch
scaler.scale(loss).backward()                  # 3. backward, gradients scaled up
scaler.unscale_(optimizer)                     # 4. undo scaling to clip in real units
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 5. cap gradient norm
scaler.step(optimizer)                         # 6. optimizer step (skipped on inf/nan)
scaler.update()                                # 7. adapt the scale factor
scheduler.step()                               # 8. next learning rate
```

**Mixed precision.** The forward/backward pass runs in fp16, which is ~2× faster
and halves activation memory on this GPU. fp16 has a small dynamic range, so
`GradScaler` multiplies the loss by a large constant before `backward()` and
divides it back out before the optimizer step; if gradients overflow to `inf`,
that step is skipped and the scale is reduced automatically. (bf16 needs no
scaler — it has fp32's exponent range — and is available as `--amp-dtype bf16`.)

**Gradient clipping.** `clip_grad_norm_` rescales the whole gradient vector when
its L2 norm exceeds 1.0. Transformers occasionally produce very large gradients
on a single weird batch; clipping prevents that one batch from destroying the
weights. The code logs the mean gradient norm per epoch so you can see it.

### 3.5 Validation, checkpointing, early stopping

After every epoch:

```python
val = run_eval(model, val_loader, ...)   # model.eval() + torch.no_grad()
if val.macro_f1 > best_f1:               # selection metric = VALIDATION macro-F1
    save best.pt + tokenizer
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience: stop
```

`model.eval()` turns dropout off; `torch.no_grad()` stops autograd from storing
activations. Forgetting either is a classic silent bug (worse and slower
validation).

The best checkpoint is chosen **only** on validation macro-F1. The test split is
never looked at during training — otherwise the reported test score would be
optimistically biased ("test-set tuning").

One subtlety worth mentioning in an interview: on the winning RoBERTa run,
validation **loss** rose after epoch 1 (0.587 → 0.617 → 0.627) while validation
**macro-F1** kept improving (0.721 → 0.723 → 0.735). The model was becoming more
confident, not worse at ranking. Early stopping on loss would have thrown away
the best model, which is why the selection metric is the metric we actually
report. It is also evidence that the probabilities are overconfident.

### 3.6 What is tracked

* MLflow params: every hyper-parameter, model name, dataset, git commit.
* MLflow per-epoch metrics: train loss/accuracy/F1, val loss/accuracy/F1, LR,
  epoch time, GPU peak memory.
* Artifacts: training curves, run config, full run record.
* `results/runs/<run>.json` on disk — the source of `results/experiments.csv`.
* Console: loss, val metrics, GPU peak allocation.

Peak GPU memory comes from `torch.cuda.max_memory_allocated()` (tensors only) and
`max_memory_reserved()` (caching allocator), both logged after training.

---

## 4. Evaluation (`src/evaluate.py`)

The selected checkpoint is loaded once and run over the official test split:

* accuracy, macro-F1, weighted-F1
* per-class precision/recall/F1/support
* confusion matrix (counts + row-normalised percentages)
* a CSV of **every** misclassified tweet with its confidence and all three
  probabilities, which feeds the error analysis
* a confidence histogram: correct predictions cluster near 1.0, while errors are
  shifted lower but still include 896 predictions ≥ 0.90 — useful evidence that
  the softmax is overconfident.

---

## 5. Error analysis (`results/error_analysis.md`)

The errors CSV is read and grouped into recurring failure patterns (sarcasm,
negation, mixed sentiment, the neutral/opinion boundary, short tweets that need
world knowledge, and probable label noise). The report quotes 24 **real** tweets
verbatim — each verified against the errors CSV — with true/predicted label,
confidence and a hypothesis, plus two concrete fixes. This is where most of the
interview conversation will happen, so read it before the interview.

---

## 6. Demo (`app.py`)

A Gradio Blocks app loads the best checkpoint (GPU if available), exposes a text
box, a confidence bar chart across the three classes, one-click real examples and
a model card with the actual test metrics. `src/predict.py` is the shared
inference helper (also usable from the CLI) so the app and the evaluation cannot
disagree about preprocessing.

---

## 7. Five interview questions (with short answers)

**Q1. Why macro-F1 instead of accuracy?**
The classes are imbalanced (neutral is ~45% of train, positive ~39%, negative
~16%) and the *test* split is skewed differently again. Accuracy rewards the
majority class; macro-F1 averages F1 over the three classes equally, so a model
that gives up on negatives (exactly what the baseline does: 46% recall) is
penalised. I report both so the trade-off is visible.

**Q2. Why not use `transformers.Trainer`?**
To show I understand what it abstracts: the optimizer parameter groups, the
warmup/decay schedule, AMP scaling, clipping, the eval loop and checkpoint
selection. Writing it by hand also made the MLflow logging and the early-stopping
logic explicit.

**Q3. Why is test accuracy not used for model selection?**
Selecting on test leaks information: with enough runs, the best test score is
partly luck. I select on validation and report one test number for the winner.
That is also why the variant rows in `results/experiments.csv` show validation
metrics only, while the baseline and the winner have test metrics.

**Q4. What does the GradScaler actually do?**
fp16 can only represent numbers down to ~6e-8; small gradients underflow to
zero. The scaler multiplies the loss by a large factor (e.g. 65536) so gradients
land in fp16's usable range, then divides the gradients back (unscale) before
clipping/stepping. If it sees inf/NaN it skips the step and lowers the scale.
This run had 1–4 skipped steps per epoch out of ~1,339.

**Q5. How do you know the model is not just memorising?**
The model generalises to a test split it never saw (0.709 macro-F1), and the
frozen-encoder ablation (only 593k trainable parameters) drops to 0.636, so the
gain comes from fine-tuning the encoder rather than from the head. I am honest
that there *is* a generalisation gap: on the winning run train accuracy reaches
0.827 while validation accuracy is 0.747, and validation loss rises after epoch 1.
The model starts to overfit after 3 epochs; early stopping and best-epoch
selection handle it, and the test score confirms the gap is not catastrophic.

**Bonus — Q6. Why dynamic padding?**
Padding every tweet to `max_length` wastes compute on pad tokens; padding per
batch means most batches are padded to ~40 tokens instead of 96, roughly halving
the work per step at equal accuracy.

**Bonus — Q7. What would you do with more compute?**
Try DeBERTa-v3-base and BERTweet (Twitter-specific pretraining), then distil the
best model; also try class-weighted loss or threshold tuning for neutral,
temperature scaling for calibration, and pseudo-labelling on unlabelled tweets.
