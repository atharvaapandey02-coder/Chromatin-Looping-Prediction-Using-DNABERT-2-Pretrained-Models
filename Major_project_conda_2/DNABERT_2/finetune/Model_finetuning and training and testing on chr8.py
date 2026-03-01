import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
    
)
import torch
from torchmetrics.classification import (
    Accuracy, F1Score, MatthewsCorrCoef, BinaryAveragePrecision
)
from sklearn.model_selection import train_test_split

# Directory paths
BASE_DIR = r"D:\Major Project from 27th March 2025\DNA Sequences subsets"
METRIC_DIR = r"D:\Major Project from 27th March 2025\Performance_chromosome_models"
TEST_FILE = r"D:\Major Project from 27th March 2025\All_DNA_Sequences_chr8_balanced.txt"

# Model identifier
MODEL_ID = "zhihan1996/DNABERT-2-117M"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Torchmetrics on GPU
metric_accuracy = Accuracy(task="binary").to(device)
metric_f1 = F1Score(task="binary", average="weighted").to(device)
metric_mcc = MatthewsCorrCoef(task="binary").to(device)
metric_auprc = BinaryAveragePrecision().to(device)

# To store best model checkpoints
best_checkpoints = []

# Callback to clear CUDA cache before and after evaluation
class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
    def on_evaluate_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

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

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    raw_logits, labels = eval_pred
    if isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0]

    logits = torch.tensor(raw_logits).to(device)
    labels = torch.tensor(labels).to(device)
    preds = torch.argmax(logits, dim=-1)

    acc = metric_accuracy(preds, labels).item()
    f1 = metric_f1(preds, labels).item()
    mcc = metric_mcc(preds, labels).item()
    try:
        auprc = metric_auprc(logits.softmax(dim=-1)[:, 1], labels).item()
    except IndexError:
        print(f"Warning: logits shape {logits.shape}, cannot compute AUPRC.")
        auprc = 0.0
    
    # Reset metrics after computing
    metric_accuracy.reset()
    metric_f1.reset()
    metric_mcc.reset()
    metric_auprc.reset()


    return {"accuracy": acc, "f1": f1, "mcc": mcc, "auprc": auprc}

# Only two chromosome number each time
for chrom_num in list(range(7, 8)):
    print(f"\n========== Starting for Chromosome {chrom_num} ==========")

    chrom_file = os.path.join(BASE_DIR, f"All_DNA_Sequences_chr{chrom_num}_balanced.txt")
    train_dict = load_data(chrom_file)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_dict["text"], train_dict["label"], test_size=0.2, random_state=42
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"]).with_format("torch")
    val_dataset = val_dataset.map(tokenize_function, batched=True).remove_columns(["text"]).with_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        trust_remote_code=True
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"./checkpoints_chr{chrom_num}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=2,
        lr_scheduler_type="linear",
        warmup_steps=100,
    )

    epoch_metrics = []

    def compute_and_store_metrics(eval_pred):
        metrics = compute_metrics(eval_pred)
        epoch_metrics.append(metrics)
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_and_store_metrics,
        callbacks=[ClearCacheCallback()]
    )

    trainer.train()

    best_model_path = f"./best_model_chr{chrom_num}"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    val_df = pd.DataFrame(epoch_metrics)
    val_df.index.name = "epoch"
    val_df.to_csv(os.path.join(METRIC_DIR, f"chr{chrom_num}_val_metrics.csv"))

    best_checkpoints.append({
        "Chromosome": chrom_num,
        "Best Checkpoint": trainer.state.best_model_checkpoint
    })

    torch.cuda.empty_cache()

    # Explicitly load the best model before testing on chr8
    print(f"\nLoading best model from {best_model_path} for testing on chr8...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path, trust_remote_code=True).to(device)
    best_tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    test_dict = load_data(TEST_FILE)
    test_dataset = Dataset.from_dict(test_dict)
    test_dataset = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"]).with_format("torch")

    print(f"Running test predictions for Chromosome {chrom_num} model on chr8...")

    test_trainer = Trainer(
        model=best_model,
        tokenizer=best_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    pred_out = test_trainer.predict(test_dataset)

    # 1) pull out raw logits
    logits_np = pred_out.predictions
    if isinstance(logits_np, tuple):           # (logits, hidden_states)
        logits_np = logits_np[0]
    if isinstance(logits_np, list):            # list of (batch,2) arrays
        logits_np = np.vstack(logits_np)       # → (N,2)
    logits_np = np.asarray(logits_np)          # ensure single array
    assert logits_np.ndim == 2 and logits_np.shape[1] == 2

    # 2) same for labels
    labels_np = pred_out.label_ids
    if isinstance(labels_np, list):
        labels_np = np.concatenate(labels_np)
    labels_np = np.asarray(labels_np)

    # 3) convert once
    logits = torch.from_numpy(logits_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    preds  = torch.argmax(logits, dim=-1)


    test_acc = metric_accuracy(preds, labels).item()
    test_f1 = metric_f1(preds, labels).item()
    test_mcc = metric_mcc(preds, labels).item()
    try:
        test_auprc = metric_auprc(logits.softmax(dim=-1)[:, 1], labels).item()
    except IndexError:
        print(f"Warning: logits shape {logits.shape}, cannot compute AUPRC.")
        test_auprc = 0.0


    # Reset metrics after test evaluation
    metric_accuracy.reset()
    metric_f1.reset()
    metric_mcc.reset()
    metric_auprc.reset()    

    test_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "MCC", "AUPRC"],
        "Value": [test_acc, test_f1, test_mcc, test_auprc]
    })
    test_df.to_csv(os.path.join(METRIC_DIR, f"chr{chrom_num}_test_on_chr8_metrics.csv"), index=False)
    print(f"Saved test metrics for chr{chrom_num} to chr8.")

    torch.cuda.empty_cache()

checkpoint_df = pd.DataFrame(best_checkpoints)
checkpoint_df.to_csv(os.path.join(METRIC_DIR, "best_model_checkpoints.csv"), index=False)
print("\nSaved all best model checkpoints.")

# ========================= PREDICTION LOOP ==============================

def predict_with_model():
    while True:
        model_path = input("\nEnter the model folder path (e.g., ./best_model_chr1) or type 'exit' to stop: ").strip()
        if model_path.lower() == "exit":
            break

        data_file = input("Enter the full path of the unseen data file: ").strip()
        if not os.path.exists(model_path) or not os.path.exists(data_file):
            print("Invalid model path or data file path. Please try again.")
            continue

        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        data = load_data(data_file)
        dataset = Dataset.from_dict(data)
        dataset = dataset.map(tokenize_function, batched=True).remove_columns(["text"]).with_format("torch")

        print("Running predictions...")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        output = trainer.predict(dataset)

        # unpack & stack
        logits_np = output.predictions
        if isinstance(logits_np, tuple):
            logits_np = logits_np[0]
        if isinstance(logits_np, list):
            logits_np = np.vstack(logits_np)
        logits_np = np.asarray(logits_np)
        labels_np = output.label_ids
        if isinstance(labels_np, list):
            labels_np = np.concatenate(labels_np)
        labels_np = np.asarray(labels_np)

        logits = torch.from_numpy(logits_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)
        preds  = torch.argmax(logits, dim=-1)


        acc = metric_accuracy(preds, labels).item()
        f1 = metric_f1(preds, labels).item()
        mcc = metric_mcc(preds, labels).item()
        try:
            auprc = metric_auprc(logits.softmax(dim=-1)[:, 1], labels).item()
        except IndexError:
            print(f"Warning: logits shape {logits.shape}, cannot compute AUPRC.")
            auprc = 0.0

        print(f"\nPrediction Results:\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nMCC: {mcc:.4f}\nAUPRC: {auprc:.4f}")
        torch.cuda.empty_cache()

# Start prediction loop after training
predict_with_model()
