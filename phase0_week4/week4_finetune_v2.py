"""
PHASE 0 — WEEK 4 (continued): Fine-tuning FinBERT on the full corrected dataset
Project: Global Financial News → Indian Stock Market Predictor
Goal: Retry fine-tuning now that there's actually enough data —
      week4_finetune.py's own conclusion was "69 examples is too
      small... accuracy on a 200+ example dataset will be far more
      trustworthy." The pipeline has since collected 386 headlines,
      and week4_label_rules.py applied targeted corrections on top
      of FinBERT's own calls. This crosses that threshold.

Why this ISN'T just "run week4_finetune.py on 386 rows instead of 69":
  386 of those "labels" are FinBERT's own zero-shot output (with ~24
  rule-based corrections layered on) — NOT independent human
  judgment. Fine-tuning a model to predict its own prior output, then
  validating against a held-out slice of THAT SAME set, is circular:
  it would trivially score well without proving the model learned
  anything a human would agree with.

  The only genuine, independent ground truth this project has is
  Week 2's 69 hand-labeled examples. So the design here is:
    TRAIN on the 386-example DB, MINUS those exact 69 headlines
    (confirmed all 69 exist verbatim in the DB — trivial to leak
    otherwise), using the rule-corrected labels as training targets.
    EVALUATE only on the untouched 69 human labels — the same set
    zero-shot FinBERT scored 69.6% against, so the comparison is
    apples-to-apples with what's already documented.

  This tests the real question: does pretraining the classifier head
  on a larger (noisier, rule-assisted) pool generalize BETTER to
  actual human judgment than either zero-shot or the original
  69-example-only fine-tune (65.2%, worse than zero-shot)?

Same head-only approach as week4_finetune.py, same reasoning for why
(freeze the encoder, train only the classifier head — full fine-tuning
on a dataset this size would still be small relative to FinBERT's
110M parameters).

Run: python3 week4_finetune_v2.py
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

WEEK2_LABELED_CSV = Path("../phase0_week2/week2_labeling_template.csv")
WEEK3_DB = Path("../phase0_week3/news_pipeline.db")
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = Path("finetuned_finbert_v2")

TRAIN_EPOCHS = 10
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
VAL_FRACTION = 0.15   # held out from the DB training pool itself, to watch for overfitting during training

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
label2id = {"positive": 0, "negative": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}


print("=" * 60)
print("  WEEK 4: Fine-tuning FinBERT on the full corrected dataset")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Build train set (DB minus Week 2's 69) and
# the untouched eval set (Week 2's 69 human labels)
# ─────────────────────────────────────────────

week2 = pd.read_csv(WEEK2_LABELED_CSV)
week2 = week2[week2["sentiment_label"].notna()].copy()
week2["sentiment_label"] = week2["sentiment_label"].astype(str).str.strip().str.lower()
week2 = week2[week2["sentiment_label"].isin(label2id)].reset_index(drop=True)
eval_headlines = set(week2["headline"])

conn = sqlite3.connect(WEEK3_DB)
db = pd.read_sql("SELECT headline, sentiment_label FROM news_articles WHERE sentiment_label IS NOT NULL", conn)
conn.close()

train_df = db[~db["headline"].isin(eval_headlines)].reset_index(drop=True)
print(f"\nEval set (Week 2 human labels, untouched):  {len(week2)}")
print(f"Train set (DB minus those {len(eval_headlines)} exact headlines): {len(train_df)}")
print("\nTrain label distribution:")
print(train_df["sentiment_label"].value_counts().to_string())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_texts = train_df["headline"].tolist()
train_labels = train_df["sentiment_label"].map(label2id).values
eval_texts = week2["headline"].tolist()
eval_labels = week2["sentiment_label"].map(label2id).values

rng = np.random.RandomState(42)
perm = rng.permutation(len(train_texts))
n_val = max(1, int(len(perm) * VAL_FRACTION))
val_idx, fit_idx = perm[:n_val], perm[n_val:]
fit_texts = [train_texts[i] for i in fit_idx]
fit_labels = [train_labels[i] for i in fit_idx]
val_texts = [train_texts[i] for i in val_idx]
val_labels = [train_labels[i] for i in val_idx]


def make_head_only_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    for param in model.bert.parameters():
        param.requires_grad = False
    return model.to(DEVICE)


def train_one_epoch(model, texts_subset, labels_subset, optimizer):
    model.train()
    indices = np.random.permutation(len(texts_subset))
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        batch_texts = [texts_subset[i] for i in batch_idx]
        batch_labels = torch.tensor([labels_subset[i] for i in batch_idx]).to(DEVICE)
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded, labels=batch_labels)
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, texts_subset, labels_subset):
    model.eval()
    correct = 0
    all_preds = []
    for start in range(0, len(texts_subset), BATCH_SIZE):
        batch_texts = texts_subset[start:start + BATCH_SIZE]
        batch_labels = labels_subset[start:start + BATCH_SIZE]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        preds = model(**encoded).logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        correct += (preds == np.array(batch_labels)).sum()
    return correct / len(texts_subset), all_preds


# ─────────────────────────────────────────────
# STEP 2: Train on the larger pool, watch the
# held-out DB slice so we can see overfitting coming
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print(f"Training head-only fine-tune on {len(fit_texts)} examples "
      f"(holding out {len(val_texts)} more from the DB pool to monitor)")
print(f"{'─' * 60}")
print(f"Device: {DEVICE}\n")

model = make_head_only_model()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

for epoch in range(TRAIN_EPOCHS):
    loss = train_one_epoch(model, fit_texts, fit_labels, optimizer)
    val_acc, _ = evaluate(model, val_texts, val_labels)
    print(f"  Epoch {epoch + 1:>2}/{TRAIN_EPOCHS}  loss={loss:.4f}  held-out-DB-slice acc={val_acc * 100:.1f}%")


# ─────────────────────────────────────────────
# STEP 3: The real test — accuracy against Week 2's
# untouched human labels, same set zero-shot scored
# 69.6% on
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("RESULTS — accuracy against Week 2's 69 human-labeled examples")
print(f"{'─' * 60}")

human_acc, human_preds = evaluate(model, eval_texts, eval_labels)
print(f"  Zero-shot FinBERT (documented earlier):        69.6%")
print(f"  Original fine-tune, 69 examples (documented):   65.2% (worse than zero-shot)")
print(f"  This run — fine-tuned on {len(fit_texts)} examples:  {human_acc * 100:.1f}%")

if human_acc > 0.696:
    print(f"\n  Beats zero-shot by {(human_acc - 0.696) * 100:+.1f} pts — worth adopting for production labeling.")
else:
    print(f"\n  Does not clearly beat zero-shot ({(human_acc - 0.696) * 100:+.1f} pts) — "
          f"keep zero-shot FinBERT + rule corrections as production labeling.")

print("\n  Breakdown by human label:")
week2_eval = week2.copy()
week2_eval["predicted"] = [id2label[p] for p in human_preds]
for label in ["positive", "negative", "neutral"]:
    subset = week2_eval[week2_eval["sentiment_label"] == label]
    if len(subset) == 0:
        continue
    subset_acc = (subset["predicted"] == label).mean() * 100
    print(f"    {label:<10} {len(subset):>3} headlines  ->  matched {subset_acc:.0f}%")

OUTPUT_DIR.mkdir(exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n  Model saved to {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
