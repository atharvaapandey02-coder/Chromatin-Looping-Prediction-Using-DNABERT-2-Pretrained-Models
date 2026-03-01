import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig,
)
from torchmetrics.classification import (
    Accuracy, F1Score, MatthewsCorrCoef, BinaryAveragePrecision
)

# ── User‐configurable paths ──────────────────────────────────────────────────────
BEST_MODEL_PATH = "./best_model_chr7"
TEST_FILE       = r"D:\Major Project from 27th March 2025\All_DNA_Sequences_chr8_balanced.txt"
METRIC_DIR      = r"D:\Major Project from 27th March 2025\Performance_chromosome_models"
# ────────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics (binary)
metric_accuracy = Accuracy(task="binary").to(device)
metric_f1       = F1Score(task="binary", average="weighted").to(device)
metric_mcc      = MatthewsCorrCoef(task="binary").to(device)
metric_auprc    = BinaryAveragePrecision().to(device)

# ── Load model & tokenizer ─────────────────────────────────────────────────────
print("Loading best saved model…")
tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)

# Force binary head in case the saved head got scrambled
config = AutoConfig.from_pretrained(BEST_MODEL_PATH, trust_remote_code=True)
config.num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(
    BEST_MODEL_PATH,
    config=config,
    trust_remote_code=True
).to(device)

# ── Load & tokenize test data ───────────────────────────────────────────────────
def load_data(fp):
    texts, labels = [], []
    with open(fp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq, lbl = line.rsplit(",", 1)
            texts.append(seq)
            labels.append(int(lbl))
    return {"text": texts, "label": labels}

test_dict    = load_data(TEST_FILE)
test_dataset = Dataset.from_dict(test_dict)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

test_dataset = (
    test_dataset
    .map(tokenize_fn, batched=True)
    .remove_columns(["text"])
    .with_format("torch")
)

# ── Run prediction ──────────────────────────────────────────────────────────────
print("Running predictions on chr8…")
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

pred_out = trainer.predict(test_dataset)

# ── Stack logits into a (N,2) array ─────────────────────────────────────────────
logits_np = pred_out.predictions

# If HF returns a tuple (e.g. (logits, hidden_states)), take the first element
if isinstance(logits_np, tuple):
    logits_np = logits_np[0]

# Now logits_np may be:
#  - a single ndarray of shape (N, 2)
#  - a list of batch-sized ndarrays → we vstack them
if isinstance(logits_np, list):
    logits_np = np.vstack(logits_np)

# Finally ensure it’s an ndarray
logits_np = np.asarray(logits_np)
# Sanity check
assert logits_np.ndim == 2 and logits_np.shape[1] == 2, (
    f"Expected logits shape (N,2), got {logits_np.shape}"
)

# Convert to tensor
logits = torch.from_numpy(logits_np).to(device)

# ── Stack labels into a (N,) array ──────────────────────────────────────────────
labels_np = pred_out.label_ids
if isinstance(labels_np, list):
    labels_np = np.concatenate(labels_np)
labels_np = np.asarray(labels_np)
labels = torch.from_numpy(labels_np).to(device)

# ── Compute predictions & metrics ──────────────────────────────────────────────
preds = torch.argmax(logits, dim=-1)

test_acc  = metric_accuracy(preds, labels).item()
test_f1   = metric_f1(preds, labels).item()
test_mcc  = metric_mcc(preds, labels).item()
try:
    probs = torch.softmax(logits, dim=-1)[:, 1]
    test_auprc = metric_auprc(probs, labels).item()
except Exception:
    print(f"Warning: cannot compute AUPRC for logits shape {logits.shape}")
    test_auprc = 0.0

# ── Save results ────────────────────────────────────────────────────────────────
os.makedirs(METRIC_DIR, exist_ok=True)
df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "MCC", "AUPRC"],
    "Value":  [test_acc, test_f1, test_mcc, test_auprc]
})
out_path = os.path.join(METRIC_DIR, "chr7_test_on_chr8_metrics.csv")
df.to_csv(out_path, index=False)

print("\n Testing complete — metrics saved to:", out_path)
print(df)

torch.cuda.empty_cache()
