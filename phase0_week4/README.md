# Week 4: FinBERT Sentiment Labeling

## What's in this folder

| File | Purpose |
|---|---|
| `setup.py` | Run once — installs `torch`, `transformers`, `scikit-learn`, `pandas` |
| `week4_finbert.py` | Validates FinBERT against your hand-labels, then auto-labels Week 3's database |
| `week4_finetune.py` | Fine-tunes FinBERT's classification head on your corrected labels |
| `week4_validation_disagreements.csv` | Headlines where FinBERT and your labels disagree, for review |
| `finetuned_finbert/` | Saved fine-tuned model (not currently used in production — see below) |

## What changed from Week 2/3

- Week 2: you hand-labeled 69 headlines manually
- Week 3: the pipeline started collecting more headlines, `sentiment_label` left `NULL`
- Week 4: FinBERT (`ProsusAI/finbert`) fills in that column automatically — but only
  after being validated against your own labels first

## How to run

```bash
python3 setup.py
python3 week4_finbert.py      # validate + auto-label Week 3's database
python3 week4_finetune.py     # optional — see "Fine-tuning" below
```

## What actually happened, honestly

The validation step exists to catch exactly what it caught: **zero-shot FinBERT
only agreed with the original Week 2 labels 31.9% of the time** — barely above
random chance for a 3-class problem. Two rounds of investigation followed:

1. **Manual review** of the disagreements caught real labeling mistakes (e.g.
   "Sensex climbs 374 points" had been mislabeled negative) — pushed agreement
   from 31.9% → 42.0%.
2. **Web research** on the underlying news events (RBI's August 2026 policy
   decision, the July 2026 US jobs report) surfaced more errors AND a real
   methodological point: **label by how markets react, not how the news reads
   on its face.** A weak jobs report is "bad news" economically but stocks
   rallied on it (lower odds of a Fed hike) — that's the correct label for a
   project predicting market reactions. This pushed agreement to 69.6%.

That's where it plateaued. The remaining ~20 disagreements are genuinely
ambiguous headlines (mixed signals in one sentence) where both readings are
defensible — not more mistakes to hunt down.

## Fine-tuning: tried, and it didn't help (yet)

`week4_finetune.py` fine-tunes only FinBERT's classification head (the BERT
encoder stays frozen) — full fine-tuning on 69 examples would just memorize
them. Cross-validated across 5 folds, it scored **65.2% (± 10.6)** — worse
than zero-shot's 69.6%, with a huge spread across folds (57.1% to 85.7%).

That spread is the real finding: **69 examples is too small to draw a
reliable conclusion from fine-tuning at all.** The saved model in
`finetuned_finbert/` is not currently used for production labeling —
zero-shot FinBERT remains the better choice until there's more data.

**Before accepting that conclusion, `week4_finetune_sweep.py` ruled out
"bad hyperparameters" as the actual cause** — swept 4 configs (baseline,
lower LR, class-weighted loss, and a conservative combo) through the same
5-fold CV. Best result: 68.0% (± 6.2), still short of zero-shot's 69.6%.
The lower-LR config did meaningfully tighten the variance (±6.2 vs the
baseline's ±10.5), so learning rate was a real lever — just not enough of
one to close the gap at this sample size. None of the 4 configs beat
zero-shot, confirming this is genuinely a data ceiling, not a tuning
problem.

## Next: accumulate more data, then retry

The real lever left isn't more clever techniques on this dataset — it's a
bigger dataset. Let `phase0_week3/week3_pipeline.py` run for a while,
label a larger batch (aim for 200+), then re-run `week4_finetune.py`.
Fine-tuning tends to actually help once there's enough data for the model
to learn a real pattern instead of memorizing noise.

## Labeling convention established this week

**Label by expected market reaction, not by surface-level "good/bad news."**
A weak jobs report can be bullish for stocks (rate-cut hopes); a growth
upgrade can coincide with a rally even if some sub-sectors fall. Apply this
consistently to any new labeling you do.

## Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: torch` / `transformers` / `sklearn` | Run `setup.py` again |
| First run is slow | Downloading FinBERT's ~400MB weights — one-time, then cached |
| `WEEK3_DB not found` | Run `phase0_week3/week3_pipeline.py --once` at least once first |

## Next: Phase 2
Use `phase1_sector_map.csv` to automatically route each labeled headline to
the stocks/sectors it affects (e.g. an RBI headline → flag every Banking
stock), building the dataset Phase 3's XGBoost model will train on.
