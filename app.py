"""Gradio demo for the fine-tuned TweetEval sentiment classifier.

Run:  python app.py   ->  http://127.0.0.1:7860

The app loads the checkpoint with the best validation macro-F1, shows the
predicted label with all three class probabilities, offers real tweets as
one-click examples and displays the test metrics of the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src.predict import SentimentPredictor, default_checkpoint

# Real tweets from the official TweetEval test split (not seen during training).
EXAMPLES = [
    "I love this new album so much, it's been on repeat all day",
    "The food was cold and the waiter was rude. Never coming back.",
    "The movie was okay, nothing special but not terrible either",
    "Great, another delayed flight. Just what I needed today.",
    "This update finally fixed the crash I reported weeks ago",
    "I'm not saying it's bad, but I wouldn't buy it again",
]

CHECKPOINT = Path(os.environ.get("SENTIMENT_CHECKPOINT", "") or default_checkpoint())
PREDICTOR = SentimentPredictor(CHECKPOINT)
TEST_METRICS = PREDICTOR.test_metrics or {}


def _model_info_md() -> str:
    """Model card text built from real checkpoint metadata and test metrics."""
    test_acc = TEST_METRICS.get("accuracy")
    test_f1 = TEST_METRICS.get("macro_f1")
    val = PREDICTOR.val_metrics
    per_class = TEST_METRICS.get("per_class") or {}
    per_class_lines = "\n".join(
        f"| {name} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']:,} |"
        for name, stats in per_class.items()
    )
    metrics_line = (
        f"**Test accuracy:** {test_acc:.4f} &nbsp;|&nbsp; **Test macro-F1:** {test_f1:.4f}"
        if test_acc is not None and test_f1 is not None
        else "_Test metrics not found - run `python -m src.evaluate` first._"
    )
    return f"""
### Model info

| | |
|---|---|
| Model | `{PREDICTOR.model_name}` (3-way sequence classification) |
| Checkpoint | `{CHECKPOINT.as_posix()}` |
| Best epoch (selected on val macro-F1) | {PREDICTOR.best_epoch} |
| Validation | acc {val.get('val_accuracy', float('nan')):.4f} / macro-F1 {val.get('val_macro_f1', float('nan')):.4f} |
| Running on | {PREDICTOR.device_name} |
| Max tokens | {PREDICTOR.max_length} (truncated) |
| Training data | TweetEval sentiment: 3 classes, ~45k tweets |

{metrics_line}

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{per_class_lines}

**Limitations.** Trained on 2010s English tweets; sarcasm, mixed sentiment and
out-of-domain text (long reviews, other languages) are where it breaks. It
predicts a single label - it cannot say "mixed". See `results/error_analysis.md`.
"""


def classify(text: str) -> tuple[dict[str, float], str]:
    """Run one prediction and format the explanation panel."""
    text = (text or "").strip()
    if not text:
        return {}, "Type a sentence first."
    result = PREDICTOR.predict([text])[0]
    probs = result["probabilities"]
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    bar = "\n".join(f"- **{name}**: {p:.1%}" for name, p in ordered)
    margin = ordered[0][1] - ordered[1][1]
    verdict = (
        f"### Predicted: **{result['label']}** ({result['confidence']:.1%} confidence)\n\n"
        f"{bar}\n\n"
        f"_Margin over the runner-up: {margin:.1%}. Low margin means the model is genuinely unsure - "
        "check the error analysis for examples._"
    )
    return {name: float(p) for name, p in probs.items()}, verdict


DEFAULT_TEXT = "The new update is fantastic, the battery now lasts all day"
INITIAL_PROBS, INITIAL_DETAIL = classify(DEFAULT_TEXT)


with gr.Blocks(title=f"TweetEval Sentiment - {PREDICTOR.model_name}") as demo:
    gr.Markdown(
        "# Tweet sentiment classifier\n"
        f"Fine-tuned **{PREDICTOR.model_name}** (plain PyTorch training loop, no `Trainer`) for "
        "**negative / neutral / positive** tweets. Type a sentence or click an example below."
    )
    with gr.Row():
        with gr.Column(scale=3):
            textbox = gr.Textbox(
                label="Your text",
                value=DEFAULT_TEXT,
                placeholder="e.g. The new update is fantastic, battery lasts all day",
                lines=3,
                max_lines=8,
            )
            with gr.Row():
                submit = gr.Button("Classify", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(examples=[[t] for t in EXAMPLES], inputs=textbox, label="Real tweets from the test set")
        with gr.Column(scale=2):
            label_out = gr.Label(value=INITIAL_PROBS, label="Probabilities", num_top_classes=3)
            detail_out = gr.Markdown(value=INITIAL_DETAIL)

    with gr.Accordion("Model info & test metrics", open=False):
        gr.Markdown(_model_info_md())

    submit.click(classify, inputs=textbox, outputs=[label_out, detail_out], api_name="classify")
    textbox.submit(classify, inputs=textbox, outputs=[label_out, detail_out], api_name="classify_text")
    clear.click(lambda: ("", {}, ""), outputs=[textbox, label_out, detail_out])

if __name__ == "__main__":
    # Gradio 6 moved theme (and other layout options) from Blocks() to launch().
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
    )
