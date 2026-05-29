"""
기존 processed 데이터셋과 GPT 생성 데이터셋을 합쳐
새로운 통합 데이터셋을 만드는 스크립트.

출력:
  data/processed/combined_train.csv
  data/processed/combined_val.csv
  data/processed/combined_test.csv
  data/processed/combined_all.csv
"""

import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

GENERATED_PATH = Path("data/raw/generated_dataset.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42


def load_existing() -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{split}.csv"
        df = pd.read_csv(path, usecols=["text", "label"])
        df["source"] = "original"
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"기존 데이터: {len(combined):,}개")
    print(f"  레이블 분포: {dict(combined['label'].value_counts())}")
    return combined


def load_generated() -> pd.DataFrame:
    df = pd.read_csv(GENERATED_PATH, usecols=["text", "label"])
    df["source"] = "generated"
    print(f"\nGPT 생성 데이터: {len(df):,}개")
    print(f"  레이블 분포: {dict(df['label'].value_counts())}")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # 전체 저장
    df[["text", "label"]].to_csv(
        PROCESSED_DIR / "combined_all.csv", index=False, encoding="utf-8"
    )

    # 8:1:1 split (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = PROCESSED_DIR / f"combined_{name}.csv"
        split_df[["text", "label"]].to_csv(out_path, index=False, encoding="utf-8")
        label_dist = dict(split_df["label"].value_counts())
        print(f"  combined_{name}.csv: {len(split_df):,}개, 레이블 {label_dist}")


def main():
    print("=" * 60)
    print("데이터셋 통합 시작")
    print("=" * 60)

    existing = load_existing()
    generated = load_generated()

    merged = pd.concat([existing, generated], ignore_index=True)
    merged = merged.dropna(subset=["text", "label"])
    merged["label"] = merged["label"].astype(int)

    print(f"\n통합 데이터: {len(merged):,}개")
    print(f"  레이블 분포: {dict(merged['label'].value_counts())}")
    print(f"  소스 분포: {dict(merged['source'].value_counts())}")

    print("\n분할 및 저장 중...")
    split_and_save(merged)

    print("\n완료!")
    print(f"저장 위치: {PROCESSED_DIR}/combined_*.csv")

    print("\n--- 통합 데이터 샘플 10개 ---")
    samples = merged.sample(10, random_state=RANDOM_SEED)
    for _, row in samples.iterrows():
        preview = row["text"][:80] + "..." if len(str(row["text"])) > 80 else row["text"]
        print(f"  [label={row['label']}, src={row['source']}] {preview}")


if __name__ == "__main__":
    main()
