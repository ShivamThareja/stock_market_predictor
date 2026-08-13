"""
PHASE 0 — WEEK 4 (continued): Fine-tuning FinBERT on your own labels
Project: Global Financial News → Indian Stock Market Predictor
Goal: Close the accuracy gap that manual review alone couldn't close
      (69.6% agreement, plateaued) by adjusting FinBERT's weights to
      match YOUR labeling patterns instead of just its own.

Why head-only fine-tuning, not full fine-tuning:
  You have 69 labeled examples. FinBERT is a ~110M parameter model.
  Full fine-tuning on 69 examples would almost certainly just
  memorize them rather than learn anything that generalizes — the
  model has vastly more capacity than your dataset has information.
  Instead, we FREEZE the BERT encoder (all its language understanding
  stays intact) and only train the small classification head on top
  (3 output units: positive/negative/neutral). This is the standard
  "linear probing" approach for tiny datasets — far less prone to
  overfitting, and still lets the model learn your specific
  calibration (e.g. "market-reaction sentiment, not surface sentiment"
  — the convention established during Week 4's manual review).

Why 5-fold cross-validation, not a single train/val split:
  A single 80/20 split on 69 examples leaves ~14 examples for
  validation — too few to trust a percentage (one flipped row moves
  the score by ~7 points). 5-fold CV trains and evaluates 5 times,
  rotating which ~14 examples are held out each time, then averages —
  a much more honest estimate of how well this generalizes, given
  how little data there is.

Be honest with yourself about what this number means: even averaged
over 5 folds, ~69 examples is a small dataset. Treat the reported
accuracy as a rough signal, not a certified number — and prioritize
labeling more data via the Week 3 pipeline over squeezing more out
of fine-tuning tricks on this small set.

Run: python3 week4_finetune.py
"""

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

WEEK2_LABELED_CSV = Path("../phase0_week2/week2_labeling_template.csv")
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = Path("finetuned_finbert")

N_FOLDS = 5
EPOCHS_PER_FOLD = 8      # small dataset, few epochs — more would just overfit further
FINAL_EPOCHS = 12        # slightly more for the final full-data fit
LEARNING_RATE = 2e-4     # higher than typical full-fine-tune LR — we're only
                          # training a small linear head, not the whole network
BATCH_SIZE = 8

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
# STEP 1: Load labeled data
# ─────────────────────────────────────────────

print("=" * 60)
print("  PHASE 0 — WEEK 4: Fine-tuning FinBERT on your labels")
print("=" * 60)

df = pd.read_csv(WEEK2_LABELED_CSV)
df = df[df["sentiment_label"].notna()]
df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
df = df[df["sentiment_label"].isin({"positive", "negative", "neutral"})].reset_index(drop=True)

print(f"\nLoaded {len(df)} labeled headlines.")
print(df["sentiment_label"].value_counts().to_string())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label2id = {"positive": 0, "negative": 1, "neutral": 2}  # matches FinBERT's own config

texts = df["headline"].tolist()
labels = df["sentiment_label"].map(label2id).values


def make_head_only_model():
    """Fresh FinBERT with the BERT encoder frozen — only the classifier head trains."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    for param in model.bert.parameters():
        param.requires_grad = False
    return model.to(DEVICE)


def train_one_epoch(model, texts_subset, labels_subset, optimizer):
    model.train()
    indices = np.random.permutation(len(texts_subset))
    total_loss = 0.0
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
    return total_loss / max(1, len(range(0, len(indices), BATCH_SIZE)))


@torch.no_grad()
def evaluate(model, texts_subset, labels_subset):
    model.eval()
    correct = 0
    for start in range(0, len(texts_subset), BATCH_SIZE):
        batch_texts = texts_subset[start:start + BATCH_SIZE]
        batch_labels = labels_subset[start:start + BATCH_SIZE]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        preds = model(**encoded).logits.argmax(dim=-1).cpu().numpy()
        correct += (preds == np.array(batch_labels)).sum()
    return correct / len(texts_subset)


# ─────────────────────────────────────────────
# STEP 2: 5-fold cross-validation — honest performance estimate
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print(f"STEP A: {N_FOLDS}-fold cross-validation (head-only fine-tuning)")
print(f"{'─' * 60}")
print(f"Device: {DEVICE}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    model = make_head_only_model()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS_PER_FOLD):
        train_one_epoch(model, train_texts, train_labels, optimizer)

    acc = evaluate(model, val_texts, val_labels)
    fold_accuracies.append(acc)
    print(f"  Fold {fold}/{N_FOLDS}: {acc * 100:.1f}% on {len(val_idx)} held-out examples")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\n  Cross-validated accuracy: {mean_acc * 100:.1f}% (± {std_acc * 100:.1f})")
print(f"  Zero-shot FinBERT (no fine-tuning) was: 69.6%")
print(f"  {'✓ Fine-tuning helped.' if mean_acc > 0.696 else '⚠ Fine-tuning did not clearly beat zero-shot at this data size.'}")
print(f"\n  Note: with only {len(df)} examples split 5 ways, each fold's score is based on")
print(f"  ~{len(df)//N_FOLDS} held-out examples — the ± spread above matters as much as the average.")


# ─────────────────────────────────────────────
# STEP 3: Final model — train on ALL labeled data, save it
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("STEP B: Training the final model on all labeled data")
print(f"{'─' * 60}")

final_model = make_head_only_model()
optimizer = torch.optim.AdamW(final_model.classifier.parameters(), lr=LEARNING_RATE)

for epoch in range(FINAL_EPOCHS):
    loss = train_one_epoch(final_model, texts, list(labels), optimizer)
    if (epoch + 1) % 4 == 0:
        print(f"  Epoch {epoch + 1}/{FINAL_EPOCHS}  loss={loss:.4f}")

train_acc = evaluate(final_model, texts, list(labels))
print(f"\n  Training-set accuracy: {train_acc * 100:.1f}%")
print("  (This is NOT a held-out score — it's expected to look strong since the")
print("   model has now seen all of these examples. Trust the cross-validation")
print("   number above for a realistic estimate, not this one.)")

OUTPUT_DIR.mkdir(exist_ok=True)
final_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n  Model saved to {OUTPUT_DIR}/")


print("\n" + "=" * 60)
print("  WEEK 4 FINE-TUNING COMPLETE")
print("=" * 60)
print(f"""
What you just built:
  ✓ Cross-validated a head-only fine-tune to get an honest accuracy estimate
  ✓ Trained a final model on all {len(df)} labeled examples
  ✓ Saved it to {OUTPUT_DIR}/ — loadable with:
      AutoModelForSequenceClassification.from_pretrained("{OUTPUT_DIR}")

The real lesson from today, more than the exact accuracy number:
{len(df)} labeled examples is genuinely small for fine-tuning. The biggest
lever left isn't more clever training tricks — it's more labeled data.
Let phase0_week3's pipeline run for a while to accumulate more headlines,
label a bigger batch, and re-run this script. Accuracy on a 200+ example
dataset will be far more trustworthy than squeezing more out of 69.

Next: Phase 2 — route labeled headlines to affected stocks/sectors
using phase1_sector_map.csv, building the dataset Phase 3's XGBoost
model will train on.
""")
