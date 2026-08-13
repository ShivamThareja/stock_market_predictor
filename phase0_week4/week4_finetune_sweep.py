"""
PHASE 0 — WEEK 4: Hyperparameter sweep — is 65.2% a data-size ceiling,
or just a bad choice of learning rate/class weighting?

week4_finetune.py's single config (lr=2e-4, no class weighting) scored
65.2% (±10.6) via 5-fold CV, below zero-shot's 69.6%. Before concluding
"69 examples is too small, period," this sweeps a few genuinely different
configs through the SAME honest 5-fold CV to rule out "wrong
hyperparameters" as the actual cause.

Configs tried:
  A) baseline (same as week4_finetune.py) — lr=2e-4, no class weights
  B) lower LR — lr=5e-5, no class weights — less aggressive head updates
  C) class-weighted loss — lr=2e-4, weighted for imbalance (33/19/17 split)
  D) lower LR + class weights + weight decay — most conservative config

Run: python3 week4_finetune_sweep.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

WEEK2_LABELED_CSV = Path("../phase0_week2/week2_labeling_template.csv")
MODEL_NAME = "ProsusAI/finbert"
N_FOLDS = 5
EPOCHS_PER_FOLD = 8
BATCH_SIZE = 8
ZERO_SHOT_BASELINE = 0.696

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CONFIGS = {
    "A_baseline (lr=2e-4)":              dict(lr=2e-4, weight_decay=0.0,  class_weighted=False),
    "B_lower_lr (lr=5e-5)":              dict(lr=5e-5, weight_decay=0.0,  class_weighted=False),
    "C_class_weighted (lr=2e-4)":        dict(lr=2e-4, weight_decay=0.0,  class_weighted=True),
    "D_conservative (lr=5e-5+wd+cw)":    dict(lr=5e-5, weight_decay=0.01, class_weighted=True),
}


print("=" * 60)
print("  WEEK 4: Fine-tuning hyperparameter sweep")
print("=" * 60)

df = pd.read_csv(WEEK2_LABELED_CSV)
df = df[df["sentiment_label"].notna()]
df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
df = df[df["sentiment_label"].isin({"positive", "negative", "neutral"})].reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label2id = {"positive": 0, "negative": 1, "neutral": 2}
texts = df["headline"].tolist()
labels = df["sentiment_label"].map(label2id).values

class_weights_full = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels)
class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float32).to(DEVICE)

print(f"\nLoaded {len(df)} labeled headlines. Device: {DEVICE}")
print(f"Class weights (balanced): positive={class_weights_full[0]:.2f}  "
      f"negative={class_weights_full[1]:.2f}  neutral={class_weights_full[2]:.2f}")


def make_head_only_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    for param in model.bert.parameters():
        param.requires_grad = False
    return model.to(DEVICE)


def train_one_epoch(model, texts_subset, labels_subset, optimizer, class_weighted):
    model.train()
    indices = np.random.permutation(len(texts_subset))
    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        batch_texts = [texts_subset[i] for i in batch_idx]
        batch_labels = torch.tensor([labels_subset[i] for i in batch_idx]).to(DEVICE)

        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)

        if class_weighted:
            loss = F.cross_entropy(outputs.logits, batch_labels, weight=class_weights_tensor)
        else:
            loss = F.cross_entropy(outputs.logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


results = {}

for config_name, cfg in CONFIGS.items():
    print(f"\n{'─' * 60}")
    print(f"Config: {config_name}  {cfg}")
    print(f"{'─' * 60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        model = make_head_only_model()
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        for epoch in range(EPOCHS_PER_FOLD):
            train_one_epoch(model, train_texts, train_labels, optimizer, cfg["class_weighted"])

        acc = evaluate(model, val_texts, val_labels)
        fold_accuracies.append(acc)
        print(f"  Fold {fold}/{N_FOLDS}: {acc * 100:.1f}%")

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    results[config_name] = (mean_acc, std_acc)
    print(f"  --> mean: {mean_acc * 100:.1f}% (± {std_acc * 100:.1f})")


print("\n" + "=" * 60)
print("  SWEEP RESULTS")
print("=" * 60)
print(f"\n  {'Config':<30} {'CV Accuracy':<20}")
print(f"  {'-'*30} {'-'*20}")
print(f"  {'Zero-shot (no fine-tune)':<30} {ZERO_SHOT_BASELINE*100:.1f}% (fixed)")
best_name, (best_mean, best_std) = max(results.items(), key=lambda x: x[1][0])
for name, (mean, std) in results.items():
    marker = "  <-- best" if name == best_name else ""
    print(f"  {name:<30} {mean*100:.1f}% (± {std*100:.1f}){marker}")

print()
if best_mean > ZERO_SHOT_BASELINE + best_std:
    print(f"  ✓ '{best_name}' beats zero-shot beyond its own noise margin — worth adopting.")
elif best_mean > ZERO_SHOT_BASELINE:
    print(f"  ~ '{best_name}' nominally beats zero-shot, but not beyond its own ± spread —")
    print(f"    not a confident win given the noise at this sample size.")
else:
    print(f"  ✗ No config beat zero-shot. Confirms this is a genuine data-size ceiling,")
    print(f"    not a hyperparameter problem. More labeled data is the real next step.")
