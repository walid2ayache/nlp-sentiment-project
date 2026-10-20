"""Classical baseline: TF-IDF (word + character n-grams) + Logistic Regression.

Why this baseline matters
-------------------------
If a 66M-parameter transformer cannot beat a linear model on bag-of-ngrams, the
extra complexity is not justified. The baseline is therefore the yardstick for
every fine-tuning experiment and uses the *same* train/val/test splits.

Hyper-parameter C is selected on the validation split only; the test split is
used once, for the final number reported in the README.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion

from .config import (
    CKPT_DIR,
    FIG_DIR,
    LABEL_NAMES,
    MLFLOW_EXPERIMENT,
    MLRUNS_DIR,
    RESULTS_DIR,
    ROOT,
    RUNS_DIR,
    BaselineConfig,
    dataclass_dict,
    ensure_dirs,
    gpu_info,
    set_seed,
    write_json,
)
from .data import clean_for_baseline, load_splits
from .plotting import (
    classification_report_df,
    plot_confusion_matrix,
    save_classification_report_md,
)
from .train import git_commit


def build_features(cfg: BaselineConfig) -> FeatureUnion:
    """Word 1-2 grams + character 3-5 grams inside words.

    Word n-grams capture vocabulary and simple negation phrases ("not good"),
    character n-grams make the model robust to the typos and slang that are
    everywhere in tweets.
    """
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, cfg.word_ngram_max),
        max_features=cfg.max_features_word,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
        max_features=cfg.max_features_char,
        min_df=3,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def evaluate_split(clf: LogisticRegression, X, y: np.ndarray) -> dict[str, Any]:
    """Accuracy + macro-F1 + probabilities for one split."""
    probs = clf.predict_proba(X)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "weighted_f1": float(f1_score(y, preds, average="weighted")),
        "y_true": y,
        "y_pred": preds,
        "probs": probs,
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline")
    parser.add_argument("--run-name", default=BaselineConfig().run_name)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    cfg = BaselineConfig(run_name=args.run_name)
    set_seed(cfg.seed)
    print(f"[info] device=CPU (scikit-learn) | gpu available for later experiments: {gpu_info().get('gpu_name')}")

    # ---- data: identical splits as the transformer experiments -------------
    splits = load_splits()
    X_text = {name: splits[name]["text"].map(clean_for_baseline).tolist() for name in splits}
    y = {name: splits[name]["label"].to_numpy() for name in splits}
    print(f"[info] train={len(y['train']):,} val={len(y['val']):,} test={len(y['test']):,}")

    # ---- features: fit on TRAIN only (no leakage from val/test) -----------
    t0 = time.time()
    featurizer = build_features(cfg)
    X_train = featurizer.fit_transform(X_text["train"])
    X_val = featurizer.transform(X_text["val"])
    X_test = featurizer.transform(X_text["test"])
    feat_time = time.time() - t0
    print(f"[info] features: {X_train.shape[1]:,} dims, {X_train.nnz:,} non-zeros, {feat_time:.0f}s")

    # ---- tiny grid search on VALIDATION macro-F1 -------------------------
    grid_results: list[dict[str, float]] = []
    best = {"macro_f1": -1.0, "C": None, "clf": None, "val": None}
    train_times: dict[str, float] = {}
    for C in cfg.C_values:
        t = time.time()
        # scikit-learn >= 1.9 dropped multiclass support from liblinear, so we use
        # the multinomial L-BFGS solver (the default for multiclass problems).
        clf = LogisticRegression(C=C, max_iter=cfg.max_iter, solver="lbfgs", random_state=cfg.seed)
        clf.fit(X_train, y["train"])
        fit_time = time.time() - t
        train_times[str(C)] = fit_time
        val = evaluate_split(clf, X_val, y["val"])
        grid_results.append({"C": C, "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "fit_time_s": fit_time})
        print(f"[grid] C={C:<5} val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} ({fit_time:.0f}s)")
        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "C": C, "clf": clf, "val": val}

    clf: LogisticRegression = best["clf"]
    assert clf is not None, "no baseline model was trained"
    val = best["val"]

    # ---- final number: test set, touched once ----------------------------
    test = evaluate_split(clf, X_test, y["test"])
    print(f"[test] acc={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f} (C={best['C']})")

    # ---- trivial reference: always predict the majority class ------------
    majority = int(np.bincount(y["train"]).argmax())
    trivial = {
        "label": LABEL_NAMES[majority],
        "val_accuracy": float(accuracy_score(y["val"], np.full_like(y["val"], majority))),
        "test_accuracy": float(accuracy_score(y["test"], np.full_like(y["test"], majority))),
        "test_macro_f1": float(f1_score(y["test"], np.full_like(y["test"], majority), average="macro")),
    }

    # ---- report + artefacts ---------------------------------------------
    report = classification_report_df(y["test"], test["y_pred"], LABEL_NAMES)
    plot_confusion_matrix(
        y["test"], test["y_pred"], LABEL_NAMES,
        title=f"TF-IDF + LogReg - test (macro-F1 {test['macro_f1']:.3f})",
        path=FIG_DIR / "baseline_confusion_matrix.png",
    )
    save_classification_report_md(report, "Baseline - test classification report", RESULTS_DIR / "baseline_classification_report.md")
    RESULTS_DIR.joinpath("baseline_grid.csv").write_text(
        "\n".join(["C,val_accuracy,val_macro_f1,fit_time_s"] + [f"{r['C']},{r['val_accuracy']:.6f},{r['val_macro_f1']:.6f},{r['fit_time_s']:.1f}" for r in grid_results]) + "\n",
        encoding="utf-8",
    )
    model_path = CKPT_DIR / "baseline_tfidf_logreg.joblib"
    joblib.dump({"pipeline": clf, "featurizer": featurizer, "C": best["C"], "label_names": list(LABEL_NAMES)}, model_path)

    record: dict[str, Any] = {
        "run_name": cfg.run_name,
        "stage": "baseline",
        "model_name": "tfidf_word+char + logistic_regression",
        "config": dataclass_dict(cfg),
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(y["train"])),
        "n_val": int(len(y["val"])),
        "n_test": int(len(y["test"])),
        "feature_time_s": round(feat_time, 1),
        "fit_time_s": round(train_times[str(best["C"])], 1),
        "total_time_s": round(feat_time + sum(train_times.values()), 1),
        "grid": grid_results,
        "best_C": best["C"],
        "best_val": {"val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"]},
        "test": {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"], "weighted_f1": test["weighted_f1"]},
        "trivial_majority": trivial,
        "gpu": gpu_info(),
        "git_commit": git_commit(),
        "checkpoint": model_path.relative_to(ROOT).as_posix(),
        "amp_dtype": "n/a",
    }
    write_json(RUNS_DIR / f"{cfg.run_name}.json", record)

    if not args.no_mlflow:
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(
                {
                    "model": "tfidf+logreg",
                    "word_ngram_max": cfg.word_ngram_max,
                    "char_ngram_min": cfg.char_ngram_min,
                    "char_ngram_max": cfg.char_ngram_max,
                    "max_features_word": cfg.max_features_word,
                    "max_features_char": cfg.max_features_char,
                    "best_C": best["C"],
                    "n_features": int(X_train.shape[1]),
                    "n_train": len(y["train"]),
                }
            )
            mlflow.set_tags({"dataset": "tweet_eval/sentiment", "stage": "baseline", "device": "cpu",
                             "git_commit": record["git_commit"]})
            for row in grid_results:  # one step per candidate C
                mlflow.log_metrics({"grid_val_accuracy": row["val_accuracy"], "grid_val_macro_f1": row["val_macro_f1"]}, step=cfg.C_values.index(row["C"]))
            mlflow.log_metrics(
                {
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "test_accuracy": test["accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "fit_time_s": record["fit_time_s"],
                }
            )
            mlflow.log_artifact(str(FIG_DIR / "baseline_confusion_matrix.png"), artifact_path="plots")
            mlflow.log_artifact(str(RESULTS_DIR / "baseline_classification_report.md"), artifact_path="reports")
            mlflow.log_dict(record, "run_record.json")

    return record


if __name__ == "__main__":
    main()
