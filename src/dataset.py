"""
Preprocessing pipeline for jailbreak detection.

Length-bias mitigation (strategies b + a + c combined):
  b) Re-source normal prompts using instruction + input concatenation (Alpaca)
     and dolly-15k, which produces naturally longer instructions (~200-400 chars)
  a) Filter out normal prompts shorter than MIN_NORMAL_CHARS (50)
  c) BERT max_length=128 so both classes receive equal truncation at token level

Train / Val / Test split: 70 / 15 / 15, stratified by label.
Outputs: data/processed/{train,val,test}.csv  (columns: text, label, token_len)
"""

import random
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME        = "bert-base-uncased"
MAX_LENGTH        = 128
MIN_NORMAL_CHARS  = 50
TARGET_PER_CLASS  = 2000
RANDOM_SEED       = 42


# ---------------------------------------------------------------------------
# Normal prompt sourcing with length augmentation
# ---------------------------------------------------------------------------

def _alpaca_instruction_input(min_chars: int) -> list[str]:
    """Return Alpaca prompts as 'instruction + input' where total >= min_chars."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for row in ds:
        parts = [row["instruction"].strip(), row.get("input", "").strip()]
        text  = " ".join(p for p in parts if p)
        if len(text) >= min_chars:
            out.append(text)
    return out


def _dolly_instruction_context(min_chars: int) -> list[str]:
    """Return dolly-15k prompts as 'instruction + context' where total >= min_chars."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    out = []
    for row in ds:
        parts = [row["instruction"].strip(), row.get("context", "").strip()]
        text  = " ".join(p for p in parts if p)
        if len(text) >= min_chars:
            out.append(text)
    return out


def collect_augmented_normal(n: int = TARGET_PER_CLASS) -> list[str]:
    """
    Collect normal prompts with length >= MIN_NORMAL_CHARS.
    Uses instruction+input concat (Alpaca) then supplements with dolly-15k.
    """
    prompts: list[str] = []

    print(f"  Trying tatsu-lab/alpaca (instruction+input, min {MIN_NORMAL_CHARS} chars) ...")
    try:
        batch = _alpaca_instruction_input(MIN_NORMAL_CHARS)
        prompts.extend(batch)
        print(f"    OK — {len(batch):,} prompts")
    except Exception as e:
        print(f"    Skip ({type(e).__name__}: {e})")

    if len(prompts) < n:
        print(f"  Trying databricks/databricks-dolly-15k (instruction+context) ...")
        try:
            batch = _dolly_instruction_context(MIN_NORMAL_CHARS)
            prompts.extend(batch)
            print(f"    OK — {len(batch):,} prompts (total {len(prompts):,})")
        except Exception as e:
            print(f"    Skip ({type(e).__name__}: {e})")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(prompts)

    seen: set[str] = set()
    unique = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:n]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset(jailbreak: list[str], normal: list[str]) -> pd.DataFrame:
    n = min(len(jailbreak), len(normal))
    df = pd.concat([
        pd.DataFrame({"text": jailbreak[:n], "label": 1}),
        pd.DataFrame({"text": normal[:n],    "label": 0}),
    ], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    train_df, tmp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=tmp["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def add_token_lengths(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    enc = tokenizer(
        df["text"].tolist(),
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    df = df.copy()
    df["token_len"] = [sum(m) for m in enc["attention_mask"]]
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print()
    print("=" * 56)
    print("  Split Statistics")
    print("=" * 56)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        jb = (df["label"] == 1).sum()
        nm = (df["label"] == 0).sum()
        print(f"  {name:5s} : {len(df):>5,} total  |  JB={jb:>4,}  Normal={nm:>4,}")

    full = pd.concat([train_df, val_df, test_df])
    full["char_len"] = full["text"].str.len()
    jb_rows = full[full["label"] == 1]
    nm_rows = full[full["label"] == 0]

    print()
    print("  Char-length after bias mitigation:")
    jb_avg = jb_rows["char_len"].mean()
    nm_avg = nm_rows["char_len"].mean()
    print(f"    Jailbreak avg  : {jb_avg:>7.0f} chars")
    print(f"    Normal avg     : {nm_avg:>7.0f} chars")
    print(f"    Ratio (JB/N)   : {jb_avg / nm_avg:>7.1f}x  (was ~30x before)")

    if "token_len" in full.columns:
        print()
        print(f"  Token-length (BERT max_length={MAX_LENGTH}):")
        jb_tok = jb_rows["token_len"].mean()
        nm_tok = nm_rows["token_len"].mean()
        print(f"    Jailbreak avg  : {jb_tok:>7.1f} tokens")
        print(f"    Normal avg     : {nm_tok:>7.1f} tokens")
        print(f"    Ratio (JB/N)   : {jb_tok / nm_tok:>7.1f}x")

    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 56)
    print("  LLM Jailbreak Detector - Preprocessing")
    print("=" * 56)

    # Load jailbreak prompts from raw dataset
    raw_path = RAW_DIR / "dataset_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run src/data_collection.py first. Expected: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    jailbreak_texts = raw_df[raw_df["label"] == 1]["text"].tolist()
    print(f"\nLoaded {len(jailbreak_texts):,} jailbreak prompts from raw dataset")

    # Re-source normal prompts with length augmentation
    print(f"\nAugmenting normal prompts ...")
    normal_texts = collect_augmented_normal(TARGET_PER_CLASS)
    print(f"  => {len(normal_texts):,} normal prompts ready\n")

    # Build balanced dataset
    print("Building balanced dataset ...")
    df = build_dataset(jailbreak_texts, normal_texts)
    print(f"  => {len(df):,} total samples ({df['label'].value_counts().to_dict()})")

    # Split
    print("\nSplitting 70 / 15 / 15 ...")
    train_df, val_df, test_df = split_dataset(df)

    # Tokenize
    print(f"\nTokenizing with {MODEL_NAME} (max_length={MAX_LENGTH}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df = add_token_lengths(train_df, tokenizer)
    val_df   = add_token_lengths(val_df,   tokenizer)
    test_df  = add_token_lengths(test_df,  tokenizer)

    # Save
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(  PROCESSED_DIR / "val.csv",   index=False, encoding="utf-8")
    test_df.to_csv( PROCESSED_DIR / "test.csv",  index=False, encoding="utf-8")
    print("  Saved to data/processed/{train,val,test}.csv")

    print_stats(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
