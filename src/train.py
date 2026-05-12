"""
Training script for LLM Jailbreak Detector.

Uses HuggingFace Trainer API with BertForSequenceClassification.
Logs metrics to results/training_log.txt and plots to results/plots/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    EarlyStoppingCallback,
)

# Add src/ to path so we can import model
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, MODEL_NAME

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR   = BASE_DIR / "results"
MODEL_DIR     = RESULTS_DIR / "models" / "bert-jailbreak"
PLOTS_DIR     = RESULTS_DIR / "plots"
LOG_FILE      = RESULTS_DIR / "training_log.txt"

MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_STEPS = 100   # linear warmup to prevent early instability
SEED         = 42


# ---------------------------------------------------------------------------
# Data loading & tokenization
# ---------------------------------------------------------------------------

def load_split(split: str) -> Dataset:
    path = PROCESSED_DIR / f"{split}.csv"
    df = pd.read_csv(path)[["text", "label"]]
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize(ds: Dataset, tokenizer) -> Dataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
    ds = ds.map(_tok, batched=True, batch_size=512, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        keep.append("token_type_ids")
    ds.set_format("torch", columns=keep)
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall"   : recall_score(labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class FileLogCallback(TrainerCallback):
    """Writes training logs to a text file in real time."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self._train_logs: list[dict] = []
        self._eval_logs:  list[dict] = []
        self._write(f"=== Jailbreak Detector — Training Log ===\n")
        self._write(f"Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write(f"Model : {MODEL_NAME}\n")
        self._write(f"Epochs: {EPOCHS} | BS: {BATCH_SIZE} | LR: {LR} | MaxLen: {MAX_LENGTH}\n")
        self._write("=" * 44 + "\n\n")

    def _write(self, msg: str):
        print(msg, end="", flush=True)
        if not self._f.closed:
            self._f.write(msg)
            self._f.flush()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if not logs:
            return
        entry = {**logs, "step": state.global_step, "epoch": state.epoch}
        is_eval = "eval_loss" in logs

        if is_eval:
            self._eval_logs.append(entry)
            self._write(
                f"[Eval  step={state.global_step:4d} epoch={state.epoch:.2f}] "
                + " | ".join(
                    f"{k.replace('eval_','')}={v:.4f}"
                    for k, v in logs.items() if isinstance(v, float)
                )
                + "\n"
            )
        else:
            self._train_logs.append(entry)
            if state.global_step % 20 == 0:
                self._write(
                    f"[Train step={state.global_step:4d} epoch={state.epoch:.2f}] "
                    + " | ".join(
                        f"{k}={v:.4f}"
                        for k, v in logs.items() if isinstance(v, float)
                    )
                    + "\n"
                )

    def on_train_end(self, args, state, control, **kw):
        self._write(f"\nEnd   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._f.close()

    @property
    def train_logs(self): return self._train_logs

    @property
    def eval_logs(self): return self._eval_logs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curve(train_logs: list[dict], eval_logs: list[dict], save_path: Path) -> None:
    if not train_logs or not eval_logs:
        print("[warn] Not enough log data to plot training curve.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    train_steps = [l["step"] for l in train_logs if "loss" in l]
    train_loss  = [l["loss"] for l in train_logs if "loss" in l]
    eval_steps  = [l["step"] for l in eval_logs]
    eval_loss   = [l["eval_loss"] for l in eval_logs]

    ax.plot(train_steps, train_loss,  color="steelblue", alpha=0.7, label="Train loss")
    ax.plot(eval_steps,  eval_loss,   color="tomato", marker="o", linewidth=2, label="Val loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()

    # --- Accuracy & F1 ---
    ax2 = axes[1]
    if eval_logs and "eval_accuracy" in eval_logs[0]:
        acc = [l["eval_accuracy"] for l in eval_logs]
        f1  = [l["eval_f1"]       for l in eval_logs]
        ax2.plot(eval_steps, acc, color="green",  marker="o", linewidth=2, label="Val Accuracy")
        ax2.plot(eval_steps, f1,  color="orange", marker="s", linewidth=2, label="Val F1 (macro)")
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Score")
        ax2.set_title("Validation Metrics")
        ax2.legend()

    plt.suptitle("BERT Jailbreak Detector — Training Curve", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved: {save_path}")


def plot_confusion_matrix(trainer: Trainer, val_ds: Dataset, save_path: Path) -> None:
    preds_out = trainer.predict(val_ds)
    preds  = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids

    cm = confusion_matrix(labels, preds)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Jailbreak"],
        yticklabels=["Normal", "Jailbreak"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation Set)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = (
        torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    )

    print("=" * 50)
    print("  LLM Jailbreak Detector - BERT Training")
    print("=" * 50)
    print(f"  Device : {device.upper()}  ({gpu_name})")
    print(f"  Epochs : {EPOCHS} | BS : {BATCH_SIZE} | LR : {LR}\n")

    # Data
    print("Loading & tokenizing datasets ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = tokenize(load_split("train"), tokenizer)
    val_ds    = tokenize(load_split("val"),   tokenizer)
    print(f"  Train : {len(train_ds):,}  |  Val : {len(val_ds):,}\n")

    # Model
    model = build_model()
    print()

    # Steps per epoch (for eval/save frequency)
    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)

    training_args = TrainingArguments(
        output_dir                  = str(MODEL_DIR),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_steps                = WARMUP_STEPS,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 20,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        seed                        = SEED,
        report_to                   = "none",
        fp16                        = False,   # disabled: caused BERT encoder collapse
    )

    log_cb = FileLogCallback(LOG_FILE)

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [
            log_cb,
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # Train
    print(f"Starting training ...\n")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/60:.1f} min\n")

    # Final eval
    print("Final validation metrics:")
    val_metrics = trainer.evaluate()
    metric_keys = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_loss"]
    for k in metric_keys:
        if k in val_metrics:
            print(f"  {k:25s}: {val_metrics[k]:.4f}")

    # Save model & tokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModel saved : {MODEL_DIR}")

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "val_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: round(v, 6) for k, v in val_metrics.items() if isinstance(v, float)},
            f, indent=2,
        )
    print(f"Metrics saved : {metrics_path}")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curve(
        log_cb.train_logs, log_cb.eval_logs,
        PLOTS_DIR / "training_curve.png",
    )
    plot_confusion_matrix(trainer, val_ds, PLOTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
