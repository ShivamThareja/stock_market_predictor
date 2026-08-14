# Week 4: FinBERT Sentiment Labeling

## What's in this folder

| File | Purpose |
|---|---|
| `setup.py` | Run once — installs `torch`, `transformers`, `scikit-learn`, `pandas` |
| `week4_finbert.py` | Validates FinBERT against your hand-labels, then auto-labels Week 3's database |
| `week4_finetune.py` | Fine-tunes FinBERT's classification head on your corrected labels (69 examples) |
| `week4_label_rules.py` | Targeted rule-based corrections for known FinBERT failure patterns (rate hold/hike/cut, market rally/fall words) |
| `week4_finetune_v2.py` | Retries fine-tuning once the DB crossed 200+ examples — see "Round 2" below |
| `week4_validation_disagreements.csv` | Headlines where FinBERT and your labels disagree, for review |
| `week4_label_rules_changes.csv` | Every headline `week4_label_rules.py` changed, and which rule fired |
| `finetuned_finbert/`, `finetuned_finbert_v2/` | Saved fine-tuned models (neither currently used in production — see below) |

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

## Round 2: rule-based corrections + fine-tuning retry at 386 examples

Once the DB crossed 200+ labeled headlines, two follow-ups:

**`week4_label_rules.py`** — rather than re-running fine-tuning blind,
first added targeted corrections for patterns FinBERT gets wrong in a
predictable direction: RBI/Fed rate **hold** headlines it calls negative
(should be neutral-at-worst), **hike** headlines it calls positive
(should lean negative), **cut** headlines it calls negative (should lean
positive), and market rally/fall verbs on headlines it left neutral.
Each rule is gated on FinBERT's *current* call, not a blind keyword
override — an early draft that blanket-relabeled any headline containing
"rally" broke immediately on "Profit-taking... stall India bond rally"
(bearish, despite the word). 24 of 386 headlines got corrected this way.
Original FinBERT output is preserved in `sentiment_label_finbert` for
audit — nothing here is destructive.

**`week4_finetune_v2.py`** — retried fine-tuning, but NOT by just
pointing the original script at more rows. 386 of those labels are
FinBERT's own output (plus the 24 rule corrections) — fine-tuning
against its own prior output and validating on a slice of that same
pool would be circular, guaranteed to look good without proving
anything. Instead: trained on the 317 DB examples that are NOT also in
Week 2's 69 hand-labeled set (confirmed zero overlap by construction),
then evaluated ONLY against those untouched 69 human labels — the same
independent ground truth zero-shot was scored against.

**Result: 69.6%, exactly tying zero-shot.** More data plus rule
corrections did not move the needle on genuinely human-judged accuracy.
Kept zero-shot FinBERT + `week4_label_rules.py`'s corrections as
production labeling; `finetuned_finbert_v2/` is saved but unused, same
as the original fine-tune. This is a real, honest negative result, not
a bug — worth knowing the ceiling is elsewhere before spending more time
here.

## Next: accumulate more data, then retry

Fine-tuning has now been tried at 69 and at ~300 examples, tied or lost
to zero-shot both times. The remaining disagreements look genuinely
ambiguous (mixed-signal headlines), which is consistent with this being
close to FinBERT's real ceiling on this task rather than a data-size
problem anymore. If you want to push further, the highest-leverage next
step is probably more/better human labels (a larger, more careful Week 2
round) rather than another fine-tuning attempt on the same labeling
process — fine-tuning can't exceed the quality of what it's trained to
imitate.

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
