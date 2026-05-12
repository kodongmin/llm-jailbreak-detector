"""
Extended analysis: jailbreak type classification, detection rate by type,
false positive / false negative case study.

Outputs:
  results/plots/jailbreak_type_analysis.png
  results/error_analysis.txt
"""

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

BASE_DIR  = Path(__file__).parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "results" / "models" / "bert-jailbreak" / "best"
RESULTS   = BASE_DIR / "results"
PLOTS_DIR = RESULTS / "plots"

MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Jailbreak type classifier (rule-based, priority order)
# ---------------------------------------------------------------------------

TYPE_RULES = [
    ("DAN / Developer-Mode",
     r"\bDAN\b|do anything now|developer mode|jailbreak|no restrictions|without restrictions"),
    ("Prompt Injection",
     r"ignore (all |your |previous |prior |the )?(previous |prior |above |all )?(instructions?|rules?|guidelines?|constraints?|filters?)"
     r"|disregard|override|forget (your |all )?instructions?"
     r"|bypass|system prompt"),
    ("Roleplay / Act-as",
     r"\bact as\b|\bacting as\b|you are now|you're now|pretend (you are|to be|you're)"
     r"|role.?play|in the role of|take on the (role|persona|character)"),
    ("Hypothetical / Fictional",
     r"\bhypothetical(ly)?\b|\bfictional(ly)?\b|in a (story|novel|fiction|scenario)"
     r"|imagine (you|that|if|a world)|what if (you|there)"),
    ("Persona / Impersonation",
     r"\bAIM\b|\bDUDE\b|evil|malicious|unrestricted (AI|model|version)"
     r"|without (any |moral |ethical )?(constraints?|restrictions?|limits?)"
     r"|your (true |real |inner )?(self|nature|purpose)"),
]

TYPE_OTHER = "Other / Mixed"


def classify_type(text: str) -> str:
    t = text.lower()
    for name, pattern in TYPE_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return TYPE_OTHER


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_all(texts: list[str], model, tokenizer, device: str, batch_size: int = 32):
    """Returns (preds, jailbreak_probs) — both are lists of length len(texts)."""
    import torch.nn.functional as F
    model.eval()
    preds, jb_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        jb_probs.extend(probs[:, 1].cpu().numpy().tolist())
    return preds, jb_probs


# ---------------------------------------------------------------------------
# Plot: jailbreak type analysis
# ---------------------------------------------------------------------------

def plot_type_analysis(type_df: pd.DataFrame, save_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: sample count per type
    ax = axes[0]
    order = type_df.sort_values("count", ascending=False)["jailbreak_type"].tolist()
    plot_df1 = type_df.set_index("jailbreak_type").loc[order].reset_index()
    sns.barplot(
        data=plot_df1, x="count", y="jailbreak_type",
        hue="jailbreak_type", legend=False,
        palette="Blues_d", ax=ax,
    )
    ax.set_title("Jailbreak Prompts by Type\n(Test Set)", fontsize=12)
    ax.set_xlabel("Sample count")
    ax.set_ylabel("")
    for bar, val in zip(ax.patches, plot_df1["count"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=9)

    # Right: detection rate per type
    ax2 = axes[1]
    order2 = type_df.sort_values("detection_rate", ascending=False)["jailbreak_type"].tolist()
    plot_df2 = type_df.set_index("jailbreak_type").loc[order2].reset_index()
    colors = ["#d73027" if r < 0.85 else "#4dac26" if r >= 0.95 else "#f1a340"
              for r in plot_df2["detection_rate"]]
    sns.barplot(
        data=plot_df2, x="detection_rate", y="jailbreak_type",
        hue="jailbreak_type", legend=False,
        palette=colors, ax=ax2,
    )
    ax2.axvline(0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Detection Rate by Jailbreak Type\n(Jailbreak Recall per Type)", fontsize=12)
    ax2.set_xlabel("Detection rate (recall)")
    ax2.set_ylabel("")
    for bar, val in zip(ax2.patches, plot_df2["detection_rate"]):
        ax2.text(min(bar.get_width() + 0.01, 1.03), bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=9)
    ax2.legend(loc="lower right")

    plt.suptitle("Jailbreak Type Analysis — BERT Detector", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Error analysis report
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return "\n".join(indent + line for line in textwrap.wrap(text, width))


def build_error_report(
    fp_rows: pd.DataFrame,
    fn_rows: pd.DataFrame,
    type_df: pd.DataFrame,
    fp_zero: bool = False,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  Error Analysis Report — BERT Jailbreak Detector")
    lines.append("  Test Set: 300 Normal + 300 Jailbreak")
    lines.append("=" * 72)

    # --- Type summary table ---
    lines.append("\n[Jailbreak Type Detection Summary]\n")
    lines.append(f"  {'Type':<35} {'N':>5} {'Detected':>8} {'Rate':>8}")
    lines.append("  " + "-" * 60)
    for _, row in type_df.sort_values("detection_rate").iterrows():
        lines.append(
            f"  {row['jailbreak_type']:<35} {int(row['count']):>5}"
            f" {int(row['detected']):>8} {row['detection_rate']:>7.1%}"
        )
    lines.append("")

    # --- False Positives ---
    lines.append("=" * 72)
    if fp_zero:
        lines.append("  FALSE POSITIVES: NONE (0 actual FPs on test set)")
        lines.append("  Showing 5 'Near-FP' cases: normal prompts with highest")
        lines.append("  jailbreak class probability (though still correctly classified)")
    else:
        lines.append("  FALSE POSITIVES (Normal -> predicted Jailbreak)")
        lines.append("  Why: normal prompts mistakenly flagged as jailbreak")
    lines.append("=" * 72)
    for i, (_, row) in enumerate(fp_rows.iterrows(), 1):
        tag = "Near-FP" if fp_zero else "FP"
        lines.append(f"\n[{tag}-{i}]")
        lines.append(f"  Jailbreak prob : {row['jb_prob']:.4f}")
        lines.append(f"  Token length   : {row['token_len']}")
        lines.append(f"  Text:\n{wrap(row['text'])}")
        lines.append(f"  Analysis: {row['fp_reason']}")

    # --- False Negatives ---
    lines.append("\n" + "=" * 72)
    lines.append("  FALSE NEGATIVES (Jailbreak -> predicted Normal)")
    lines.append("  Why: jailbreak prompts that slipped past the detector")
    lines.append("=" * 72)
    for i, (_, row) in enumerate(fn_rows.iterrows(), 1):
        lines.append(f"\n[FN-{i}]")
        lines.append(f"  Jailbreak type : {row['jailbreak_type']}")
        lines.append(f"  Token length   : {row['token_len']}")
        lines.append(f"  Text:\n{wrap(row['text'])}")
        lines.append(f"  Analysis: {row['fn_reason']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic reason inference
# ---------------------------------------------------------------------------

def infer_fp_reason(text: str, token_len: int) -> str:
    t = text.lower()
    reasons = []
    if token_len >= 120:
        reasons.append("very long instruction (truncated at 128 tokens) resembles jailbreak length pattern")
    if re.search(r"ignore|bypass|override|forget", t):
        reasons.append("contains imperative verbs (ignore/bypass/override) common in jailbreak prompts")
    if re.search(r"act as|pretend|roleplay|imagine", t):
        reasons.append("contains role-framing language (act as/pretend/imagine) associated with jailbreaks")
    if re.search(r"without|no restriction|unrestricted", t):
        reasons.append("phrase 'without/no restriction' triggers jailbreak pattern")
    if re.search(r"how to|step.by.step|instruct", t):
        reasons.append("imperative instructional phrasing similar to jailbreak task framing")
    if not reasons:
        reasons.append("unusual phrasing or rare vocabulary caused overlap with jailbreak feature space")
    return "; ".join(reasons)


def infer_fn_reason(text: str, jb_type: str, token_len: int) -> str:
    t = text.lower()
    reasons = []
    if token_len < 30:
        reasons.append("very short prompt — key jailbreak signals likely truncated or absent")
    if jb_type == "Hypothetical / Fictional":
        reasons.append("hypothetical framing is linguistically close to normal creative writing tasks")
    if jb_type == "Persona / Impersonation":
        reasons.append("persona-based attack uses indirect language that avoids explicit jailbreak markers")
    if jb_type == TYPE_OTHER:
        reasons.append("novel/uncommon attack pattern not well represented in training data")
    if re.search(r"story|novel|fiction|creative|write", t):
        reasons.append("creative writing context masks harmful intent within seemingly benign request")
    if re.search(r"for (research|educational|academic|study)", t):
        reasons.append("educational/research framing reduces surface similarity to known jailbreak patterns")
    if not reasons:
        reasons.append("subtle phrasing without explicit jailbreak keywords; relies on context understood only at inference")
    return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    print("Loading test set ...")
    test_df = pd.read_csv(PROC_DIR / "test.csv")

    print("Running inference ...")
    preds, jb_probs = predict_all(test_df["text"].tolist(), model, tokenizer, device)
    test_df["pred"]    = preds
    test_df["jb_prob"] = jb_probs

    # Classify jailbreak types (only for JB samples)
    jb_mask = test_df["label"] == 1
    test_df.loc[jb_mask, "jailbreak_type"] = test_df.loc[jb_mask, "text"].apply(classify_type)
    test_df.loc[~jb_mask, "jailbreak_type"] = "Normal"

    # Overall metrics
    correct = (test_df["label"] == test_df["pred"]).sum()
    total   = len(test_df)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.4f}")

    fp_df = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)]
    fn_df = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)]
    print(f"False Positives : {len(fp_df)}")
    print(f"False Negatives : {len(fn_df)}")
    fp_zero = len(fp_df) == 0

    # -----------------------------------------------------------------------
    # 1. Type analysis
    # -----------------------------------------------------------------------
    jb_df = test_df[jb_mask].copy()
    type_stats = []
    for jb_type in jb_df["jailbreak_type"].unique():
        subset   = jb_df[jb_df["jailbreak_type"] == jb_type]
        detected = (subset["pred"] == 1).sum()
        type_stats.append({
            "jailbreak_type":  jb_type,
            "count":           len(subset),
            "detected":        detected,
            "detection_rate":  detected / len(subset),
        })
    type_df = pd.DataFrame(type_stats).sort_values("count", ascending=False).reset_index(drop=True)

    print("\nDetection rate by type:")
    for _, row in type_df.iterrows():
        print(f"  {row['jailbreak_type']:<35} {int(row['count']):>4} samples  {row['detection_rate']:.1%}")

    plot_type_analysis(type_df, PLOTS_DIR / "jailbreak_type_analysis.png")

    # -----------------------------------------------------------------------
    # 2. Error case selection
    # -----------------------------------------------------------------------
    # FP: if no actual FPs, use normal samples with highest jailbreak probability (near-FP)
    if fp_zero:
        print("\n[Note] No actual False Positives - using 5 'near-FP' normal samples")
        print("       (highest jailbreak class probability among correctly classified normals)")
        nm_df = test_df[(test_df["label"] == 0) & (test_df["pred"] == 0)]
        fp_sample = nm_df.nlargest(5, "jb_prob").copy()
        fp_sample["near_fp"] = True
    else:
        fp_sample = fp_df.sort_values("token_len", ascending=False).head(5).copy()
        fp_sample["near_fp"] = False

    fp_sample["fp_reason"] = [
        infer_fp_reason(row["text"], row["token_len"])
        for _, row in fp_sample.iterrows()
    ]

    # Select up to 5 FN cases: prefer diverse types
    fn_sample_rows = []
    seen_types: set = set()
    for _, row in fn_df.sort_values("token_len").iterrows():
        fn_sample_rows.append(row)
        seen_types.add(row["jailbreak_type"])
        if len(fn_sample_rows) >= 5:
            break
    fn_sample = pd.DataFrame(fn_sample_rows).copy()
    fn_sample["fn_reason"] = [
        infer_fn_reason(row["text"], row["jailbreak_type"], row["token_len"])
        for _, row in fn_sample.iterrows()
    ]

    # -----------------------------------------------------------------------
    # 3. Save error analysis report
    # -----------------------------------------------------------------------
    report = build_error_report(fp_sample, fn_sample, type_df, fp_zero=fp_zero)
    report_path = RESULTS / "error_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
