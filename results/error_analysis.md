# Error analysis — RoBERTa-base on TweetEval sentiment

**Model:** `roberta-base` fine-tuned with the custom loop in `src/train.py`
**Checkpoint:** `checkpoints/e4_roberta_lr2e-5/best.pt` (epoch 3, selected on validation macro-F1 = 0.7348)
**Split:** official TweetEval test, 12,284 tweets, evaluated **once** (`python -m src.evaluate`)

| metric | value |
|---|---|
| accuracy | 0.7078 |
| macro-F1 | 0.7088 |
| weighted-F1 | 0.7067 |
| misclassified | 3,589 / 12,284 (29.2%) |
| mean confidence on errors | 0.759 |
| errors with confidence ≥ 0.90 | 896 (25.0%) |

Per-class test results (from `results/e4_roberta_lr2e-5_test_metrics.json`):

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| negative | 0.703 | 0.779 | 0.739 | 3,972 |
| neutral | 0.750 | 0.636 | 0.688 | 5,937 |
| positive | 0.641 | 0.769 | 0.699 | 2,375 |

**First observation:** the model is good at spotting *polarity* (negative recall 0.779,
positive recall 0.769) but weak at the *neutral* class (recall 0.636). Precision on
positive is only 0.641, i.e. the model over-fires "positive" on tweets that are
merely news or mildly upbeat.

## Where the errors are

| true → predicted | # errors | mean confidence | median words |
|---|---:|---:|---:|
| neutral → negative | 1,257 | 0.777 | 16 |
| neutral → positive | 906 | 0.784 | 13 |
| negative → neutral | 761 | 0.709 | 17 |
| positive → neutral | 499 | 0.751 | 13 |
| negative → positive | 117 | 0.727 | 16 |
| positive → negative | 49 | 0.743 | 16 |

**95.4%** of all errors involve the neutral class (as the true or the predicted
label). The interesting cases are therefore not "happy vs sad" but "is this tweet
expressing an opinion at all?".

The examples below are quoted verbatim from
`results/errors_e4_roberta_lr2e-5_test.csv`. Some are political tweets (the test
set is US-politics-heavy); a couple contain strong language. The scores are the
model's softmax probability for the predicted class.

---

## Pattern 1 — Sarcasm, irony and figurative language

The model reads sentiment words literally. Sarcasm inverts the meaning and is the
hardest pattern for a 125M-parameter encoder trained on 43k tweets.

| tweet | true | predicted | conf | hypothesis |
|---|---|---|---:|---|
| I'm Dying😂😂😂 #TheWalkingDead | positive | negative | 0.98 | "dying" is hyperbole for laughter, not distress; the hashtag names a TV show the user loves. The model sees a death word and a fandom hashtag and predicts negative. |
| this picture drawn by muhammad ali is KILLING ME | positive | negative | 0.98 | "killing me" = hilarious. Capitalisation intensifies the literal reading instead of the idiomatic one. |
| @user Wow...this is one deadly statement! Reminding me of the Greta Muhammad Ali. | positive | negative | 0.97 | "deadly" is praise slang here ("devastatingly good"); "Wow" is positive. The literal negative adjective wins. |
| She totally Melania'd that speech #UsefulNewVerbs | negative | positive | 0.99 | Requires 2016 context (the plagiarism scandal) and the invented verb. There is no negative lexicon, so the model guesses positive from the proper-noun pattern. |
| Ah yeah, but these are the *good* immigrants, nice well spoken white non-Muslim boys. Not bad hombres. | negative | positive | 0.99 | Ironic deployment of "good/nice/not bad" to mock a stereotype. Every surface cue is positive, and the negation "not bad" is scoped as praise. |

## Pattern 2 — Negation, contrast and mixed sentiment

Composition is hard: one sentiment word plus "but", "not" or an emoticon can flip
the label, and the model often focuses on the wrong clause.

| tweet | true | predicted | conf | hypothesis |
|---|---|---|---:|---|
| I just can't stop listening to this :( Leonard Cohen - Leaving the Table via @user | positive | negative | 0.97 | "can't stop" is an idiom of enthusiasm; the sad emoticon refers to Cohen's death. The model latches onto "can't" + ":(", ignoring the idiom. |
| We love our green vehicles (@Tesla Model S and @user Mirai) but sometimes lack of infrastructure makes the V8-pow… | neutral | positive | 0.99 | Praise followed by a complaint; the dataset also truncates it with "…". The model weights "love" and ignores "lack". |
| @user I'm totally looking forward to Yakuza 0 and Persona 5! But after XIII I'm not sure I can ever love again | neutral | positive | 0.97 | Hyperbolic complaint after an earlier bad game. Labeled neutral (mixed); the model keys on "looking forward". |
| you did have a valentines last year tho . 😕🤔 but that's cool too | neutral | positive | 0.99 | Concessive "but that's cool" plus a skeptical emoji. The clause that actually carries the attitude is the first one. |
| melania gave an amazing speech at the rnc, too bad it wasn't hers! #michelleobamaftw (Vine by @user | neutral | positive | 1.00 | Sarcastic correction with a hashtag that reveals the speaker's side. "amazing speech" dominates and the model is 100% confident. |
| Grayson Allen should have gone to the #NBA after last season. His stock wont ever b that high again and that's not a shot at him as a player | neutral | negative | 0.91 | The writer explicitly disclaims criticism ("not a shot at him") while still being critical. The label is neutral; the model follows the critical framing. |
| .Why animal testing doesn't work! #CambridgeKillers | neutral | negative | 0.99 | "doesn't work" flips the polarity of "work" correctly, but this is a campaign statement, labeled neutral. Arguably the model is right and the label is debatable. |

## Pattern 3 — Where does "neutral" end? News, commentary and opinion

Most of the remaining errors are tweets that *report* or *comment* rather than
express a clear feeling. This is where the annotation guideline, not the language,
decides the label.

| tweet | true | predicted | conf | hypothesis |
|---|---|---|---:|---|
| And here's one of the deplorables proving my point. | neutral | negative | 0.98 | A political in-group label with negative connotations. The labeler apparently treated it as neutral political commentary; the model reacts to the slur-like token. |
| Bookends to a Disastrous Presidencyhttps://t.co/OXdnZ86rvD #Obama's pathetic final non-victory lap around the world. #UniteBlue #TCOT | neutral | negative | 0.99 | "Disastrous" and "pathetic" are unambiguous, yet the tweet is labeled neutral. Either the label is wrong or the annotation treats political attack as non-emotional. The model's prediction looks more defensible than the label. |
| Supermodel @user is proving she's in the business of making history and shattering barriers:… | neutral | positive | 1.00 | Celebrity-news style praise. The label treats reporting as neutral; the model treats the positive framing as sentiment. |
| The gold standard in Christmas caroling: Detroit Mar Thoma Church. No doubt about it! 🏆🎄 | neutral | positive | 0.99 | Praise plus "No doubt" (negation) in a community-news tweet. The model ignores the negation and predicts positive. |
| @user so many deplorables..need a basket the size of Russia | neutral | negative | 0.99 | Sarcastic political commentary; "deplorables" carries the sentiment. Labeled neutral, so this is again a boundary case between opinion and report. |

## Pattern 4 — Very short tweets and world knowledge

| tweet | true | predicted | conf | hypothesis |
|---|---|---|---:|---|
| ‘Kim Fatty the Third’ | negative | neutral | 0.99 | Four words, a mocking nickname for Kim Jong-un. Negative meaning comes entirely from world knowledge; there is no sentiment word, so the model defaults to neutral. |
| Saakashvili Founds New Party, Begins Struggle for Power in Kiev – Nolan Peterson | negative | neutral | 0.98 | A news headline. "Struggle" is political, not emotional; the model correctly reads it as factual, while the label encodes the article's negative framing. |
| #thewalkingdead  😍 sweet dreams | neutral | positive | 0.99 | A fandom tweet with a heart-eyes emoji, labeled neutral. The emoji carries the affection; the label is arguably wrong. |

## Pattern 5 — Probable label noise / genuinely ambiguous labels

I looked at the highest-confidence errors and a non-trivial share look
*mislabelled* rather than misclassified. Of the 24 examples in this report, I
would defend 4 as label errors (marked below) and 2 more as debatable. That is a
realistic ceiling for this dataset: TweetEval test labels were produced by
crowd annotation on tweets without conversation context.

| tweet | true | predicted | conf | hypothesis |
|---|---|---|---:|---|
| i am so blessedben_verlander's photo | neutral | positive | 1.00 | "blessed" is a textbook positive word; the tweet looks like a photo caption that lost a space. Most likely a **label error** (or a caption the annotator read as neutral). |
| #Deplorables = ;-) #RockNRoll !#JoanJett & the #Blackhearts -I #Love #RockNRoll via @user | neutral | positive | 1.00 | Music fandom with "#Love" and a wink; labeled neutral, probably because the annotator saw a link share. Debatable, but the positive prediction is reasonable. |
| LOT of 2 Allen Edmonds Grayson leather dress loafters Brown/Black career office #AllenEdmonds #Grays | positive | neutral | 0.99 | An eBay-style product listing. The positive label is hard to justify; the model's neutral prediction looks **more correct than the label**. |
| Now Playing on #Channel80: Ricky Martin - Casi Un Bolero - | positive | neutral | 0.99 | A "now playing" radio tweet with no evaluative words. Positive label likely reflects the annotator's taste rather than the text. |

---

## Cross-cutting findings

1. **Neutral is the bottleneck, not polarity.** 95% of errors involve neutral, and
   neutral recall (0.636) is ~14 points below the two polar classes. Any further
   gain has to come from modelling "is there an opinion here?", which is an
   annotation-guideline question as much as a modelling question.
2. **The model is confidently wrong.** Mean confidence on errors is 0.759 and 25%
   of errors exceed 0.90, including a 1.00 on the sarcastic "amazing speech"
   tweet. On this distribution the softmax score is *not* a reliable uncertainty
   signal; calibration (temperature scaling on validation) would make the demo
   more honest.
3. **Label shift inflates validation.** Validation (same distribution as train:
   15.6% negative) gives 0.7348 macro-F1; the official test (32.3% negative)
   gives 0.7088. The gap is mostly distribution, not overfitting: negative recall
   is actually high (0.779), while neutral recall drops.
4. **Political/news tweets dominate high-confidence errors.** The test set is
   heavy on 2016 US politics; sarcasm and in-group labels there require world
   knowledge that a 43k-tweet fine-tune does not provide.
5. **Emoji and slang are only partly handled.** 😍 is used positively, but literal
   words such as "dying", "killing" and "deadly" are read literally, which is
   exactly the opposite of how Twitter uses them.

## Two concrete fixes (and how I would validate them without touching test)

**1. Rebalance the loss toward neutral + calibrate.**
Add label smoothing (0.05–0.1) and a class-weighted cross-entropy (neutral weight
≈1.3–1.5), then pick the weight on **validation** neutral recall. Expected effect:
fewer neutral→positive errors (currently 906) at the cost of some positive
precision; macro-F1 should move up because neutral F1 is the weakest link.
Temperature scaling fitted on validation would also fix the overconfidence
(25% of errors above 0.90).

**2. Clean the labels / use a stronger, context-aware model.**
Manually re-annotate a stratified sample of ~200 neutral test errors (I estimate
~15–20% are label errors from the examples above, but that is a hypothesis, not a
measurement). If confirmed, apply label smoothing or drop the worst training
labels, and compare against `deberta-v3-base` / `BERTweet`, which handle irony and
Twitter-specific syntax better. Evaluate every change on validation only, then run
the winner once on test exactly as in this report.
