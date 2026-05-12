"""
BERT vs RoBERTa comparison: metrics table + Val F1 learning curve overlay.
Reads: results/val_metrics.json, results/roberta_metrics.json
Writes: results/plots/bert_vs_roberta.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR  = Path(__file__).parent.parent
RESULTS   = BASE_DIR / "results"
PLOTS_DIR = RESULTS / "plots"


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_comparison(bert: dict, roberta: dict) -> None:
    header = f"{'Metric':<20} {'BERT':>12} {'RoBERTa':>12} {'Delta':>10}"
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  BERT vs RoBERTa — Comparison")
    print("=" * len(header))
    print(header)
    print(sep)

    keys = [
        ("Val Accuracy",  "val",  "accuracy"),
        ("Val F1 (macro)","val",  "f1_macro"),
        ("Val Precision", "val",  "precision"),
        ("Val Recall",    "val",  "recall"),
        ("Test Accuracy", "test", "accuracy"),
        ("Test F1 (macro)","test","f1_macro"),
        ("Test Precision","test", "precision"),
        ("Test Recall",   "test", "recall"),
    ]
    for label, split, key in keys:
        b = bert[split][key]
        r = roberta[split][key]
        d = r - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<18} {b:>12.4f} {r:>12.4f} {sign}{d:>9.4f}")

    print(sep)
    # Conclusion
    b_f1 = bert["test"]["f1_macro"]
    r_f1 = roberta["test"]["f1_macro"]
    if r_f1 > b_f1:
        winner = f"RoBERTa (+{r_f1-b_f1:.4f} test F1)"
    elif b_f1 > r_f1:
        winner = f"BERT (+{b_f1-r_f1:.4f} test F1)"
    else:
        winner = "Tie (identical test F1)"
    print(f"\n  Conclusion: {winner} wins on test set")
    print("=" * len(header) + "\n")


def plot_comparison(bert: dict, roberta: dict, save_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Val F1 learning curve per epoch ---
    ax = axes[0]
    bert_h    = bert.get("epoch_history", [])
    roberta_h = roberta.get("epoch_history", [])

    if bert_h:
        epochs_b = [e["epoch"] for e in bert_h]
        f1s_b    = [e["val_f1"] for e in bert_h]
        ax.plot(epochs_b, f1s_b, "o-", color="steelblue", linewidth=2, label="BERT")
    if roberta_h:
        epochs_r = [e["epoch"] for e in roberta_h]
        f1s_r    = [e["val_f1"] for e in roberta_h]
        ax.plot(epochs_r, f1s_r, "s-", color="tomato", linewidth=2, label="RoBERTa")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1 (macro)")
    ax.set_title("Validation F1 per Epoch")
    ax.set_ylim(0.9, 1.01)
    ax.legend()

    # --- Right: Bar chart for final metrics ---
    ax2 = axes[1]
    metrics_labels = ["Val\nAccuracy", "Val\nF1", "Test\nAccuracy", "Test\nF1"]
    bert_vals    = [bert["val"]["accuracy"], bert["val"]["f1_macro"],
                    bert["test"]["accuracy"], bert["test"]["f1_macro"]]
    roberta_vals = [roberta["val"]["accuracy"], roberta["val"]["f1_macro"],
                    roberta["test"]["accuracy"], roberta["test"]["f1_macro"]]

    x      = range(len(metrics_labels))
    width  = 0.35
    bars_b = ax2.bar([i - width/2 for i in x], bert_vals,    width, label="BERT",    color="steelblue", alpha=0.85)
    bars_r = ax2.bar([i + width/2 for i in x], roberta_vals, width, label="RoBERTa", color="tomato",    alpha=0.85)

    ax2.set_ylim(0.95, 1.005)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylabel("Score")
    ax2.set_title("Final Metrics Comparison")
    ax2.legend()

    for bar, val in [(b, v) for b, v in zip(bars_b, bert_vals)] + \
                    [(b, v) for b, v in zip(bars_r, roberta_vals)]:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

    plt.suptitle("BERT vs RoBERTa — Jailbreak Detector", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {save_path}")


def main():
    bert_path    = RESULTS / "val_metrics.json"
    roberta_path = RESULTS / "roberta_metrics.json"

    if not bert_path.exists():
        raise FileNotFoundError(f"Missing: {bert_path}")
    if not roberta_path.exists():
        raise FileNotFoundError(f"Missing: {roberta_path}")

    bert    = load(bert_path)
    roberta = load(roberta_path)

    print_comparison(bert, roberta)
    plot_comparison(bert, roberta, PLOTS_DIR / "bert_vs_roberta.png")


if __name__ == "__main__":
    main()
