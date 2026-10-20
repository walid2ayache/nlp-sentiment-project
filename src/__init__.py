"""TweetEval sentiment project source package.

Environment defaults are set here (before any submodule imports MLflow,
Transformers or the tokenizers library) so every entry point behaves the same.
"""

import os

# MLflow >= 3.x refuses the classic local ``mlruns/`` file store unless this
# opt-in is set. This project deliberately uses the local file store because it
# is self-contained (no background database server needed for a portfolio repo).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Windows without Developer Mode cannot create the symlinks HF uses for its
# cache; the warning is noise, and the cache still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Avoid forking issues / noisy warnings when the DataLoader uses workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
