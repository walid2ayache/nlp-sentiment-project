# EDA - TweetEval sentiment

Dataset: `tweet_eval/sentiment`  |  classes: negative, neutral, positive

| split | n | negative | neutral | positive | median words | p99 words |
|---|---|---|---|---|---|---|
| train | 42,827 | 6,663 (15.56%) | 19,371 (45.23%) | 16,793 (39.21%) | 20 | 29 |
| val | 4,759 | 740 (15.55%) | 2,153 (45.24%) | 1,866 (39.21%) | 20 | 29 |
| test | 12,284 | 3,972 (32.33%) | 5,937 (48.33%) | 2,375 (19.33%) | 15 | 27 |

Plots: `results/figures/eda_class_balance.png`, `results/figures/eda_length_distribution.png`.

## Observations

1. **Class imbalance** - `neutral` is the most common training class (45.23%). Accuracy alone would therefore be misleading, so **macro-F1 is the headline metric** (all three classes weigh the same) and accuracy is reported next to it.
2. **Label shift between validation and test** - the official test split is *not* drawn from the same label distribution: `positive` is 39.21% of train but 19.33% of test (-19.9 percentage points). This is a property of TweetEval, not of our split. It means test scores are systematically lower than validation scores, and it is one of the limitations discussed in the README. We do **not** re-balance the test set: that would be tuning on test.
3. **Short texts** - the median tweet is 20 words and 99% are at most 29 words, so almost nothing is truncated at 96 tokens; tokenisation is cheap and each epoch is fast.
4. **Noisy language** - @mentions, URLs, hashtags, slang and typos mean a purely lexical model (TF-IDF) will miss sarcasm and negation; that is the gap the transformer is expected to close.
5. **Duplicates** - 0 duplicate training texts were removed before splitting.

## Tokenizer coverage

On a 4,000-tweet sample the tokenizer produced a median of 29 tokens (p95 = 43, p99 = 50). With `max_length=96`, **0.0%** of tweets get truncated.
