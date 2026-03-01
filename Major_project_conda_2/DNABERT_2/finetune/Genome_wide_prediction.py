import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainerCallback,
)
from torchmetrics.classification import Accuracy, F1Score, MatthewsCorrCoef, BinaryAveragePrecision

# Directory path
METRIC_DIR = r"D:\Major Project from 27th March 2025\Performance_chromosome_models"

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Torchmetrics on GPU
metric_accuracy = Accuracy(task="binary").to(device)
metric_f1 = F1Score(task="binary", average="weighted").to(device)
metric_mcc = MatthewsCorrCoef(task="binary").to(device)
metric_auprc = BinaryAveragePrecision().to(device)

# Callback to clear CUDA cache before and after evaluation
class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
    def on_evaluate_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

# Data loader function
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seq_part, label_str = line.rsplit(",", 1)
                texts.append(seq_part.strip())
                labels.append(int(label_str))
            except Exception as e:
                print(f"Skipping line due to error: {e}\nLine: {line}")
    return {"text": texts, "label": labels}

# Model prediction function
def predict_with_model(batch_size: int = 32):
    accuracy_matrix = []
    f1_matrix      = []
    mcc_matrix     = []
    auprc_matrix   = []

    while True:
        model_path = input("\nEnter the model folder path (or 'exit'): ").strip().strip('"')
        if model_path.lower() == "exit":
            break

        data_folder = input("Enter the path to the unseen-data folder: ").strip().strip('"')
        if not os.path.isdir(model_path) or not os.path.isdir(data_folder):
            print("Invalid model or data-folder path. Try again.")
            continue

        print("Loading model and tokenizer...")
        model     = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer)

        model.eval()
        metric_accuracy.reset()
        metric_f1.reset()
        metric_mcc.reset()
        metric_auprc.reset()

        model_chrom = model_path.split("_")[-1]
        print(f"Model chromosome: {model_chrom}")

        acc_row, f1_row, mcc_row, auprc_row = [], [], [], []

        for fname in os.listdir(data_folder):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(data_folder, fname)
            print(f"\n→ Processing {fname}")

            # load and basic tokenize (no padding)
            data = load_data(path)
            encodings = tokenizer(
                data["text"],
                padding=False,
                truncation=True,
                max_length=512,
            )

            # prepare samples for DataLoader
            samples = []
            for i in range(len(data["text"])):
                samples.append({
                    "input_ids":     encodings["input_ids"][i],
                    "attention_mask":encodings["attention_mask"][i],
                    "labels":        data["label"][i],
                })

            dl = DataLoader(
                samples,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )

            all_logits = []
            all_preds  = []
            all_labels = []

            with torch.no_grad():
                for batch in dl:
                    labels = batch.pop("labels").to(device)
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    out    = model(**inputs)
                    logits = out.logits
                    preds  = logits.argmax(dim=-1)

                    all_logits.append(logits)
                    all_preds.append(preds)
                    all_labels.append(labels)

            logits_tensor = torch.cat(all_logits, dim=0)
            preds_tensor  = torch.cat(all_preds, dim=0)
            labs_tensor   = torch.cat(all_labels, dim=0)

            # compute metrics
            acc   = metric_accuracy(preds_tensor, labs_tensor).item()
            f1    = metric_f1       (preds_tensor, labs_tensor).item()
            mcc   = metric_mcc      (preds_tensor, labs_tensor).item()
            try:
                probs = logits_tensor.softmax(dim=-1)[:, 1]
                auprc = metric_auprc(probs, labs_tensor).item()
            except Exception:
                auprc = 0.0

            print(f"Results: Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, AUPRC={auprc:.4f}")

            acc_row.append(acc)
            f1_row.append(f1)
            mcc_row.append(mcc)
            auprc_row.append(auprc)

            metric_accuracy.reset()
            metric_f1.reset()
            metric_mcc.reset()
            metric_auprc.reset()
            torch.cuda.empty_cache()

        accuracy_matrix.append(acc_row)
        f1_matrix     .append(f1_row)
        mcc_matrix    .append(mcc_row)
        auprc_matrix  .append(auprc_row)

    # save to CSV
    os.makedirs(METRIC_DIR, exist_ok=True)
    pd.DataFrame(accuracy_matrix).to_csv(os.path.join(METRIC_DIR, "accuracy_matrix.csv"), index=False)
    pd.DataFrame(f1_matrix)     .to_csv(os.path.join(METRIC_DIR, "f1_score_matrix.csv"), index=False)
    pd.DataFrame(mcc_matrix)    .to_csv(os.path.join(METRIC_DIR, "mcc_matrix.csv"), index=False)
    pd.DataFrame(auprc_matrix)  .to_csv(os.path.join(METRIC_DIR, "auprc_matrix.csv"), index=False)

    print("\nAccuracy Matrix:\n", np.array(accuracy_matrix))
    print("\nF1 Score Matrix:\n", np.array(f1_matrix))
    print("\nMCC Matrix:\n", np.array(mcc_matrix))
    print("\nAUPRC Matrix:\n", np.array(auprc_matrix))


if __name__ == "__main__":
    predict_with_model(batch_size=32)
