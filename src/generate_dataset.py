"""
GPT-4o-mini를 사용한 Jailbreak 탐지 데이터셋 생성 스크립트.
- 배치마다 즉시 CSV 저장 (중간 종료 시 데이터 보존)
- 재시작 시 기존 데이터 이어서 생성
- 연속 실패 10회 시 해당 배치 스킵
"""

import os
import csv
import json
import time
import random
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUTPUT_PATH = Path("data/raw/generated_dataset.csv")
BATCH_SIZE = 50
MAX_CONSECUTIVE_FAILS = 10

CATEGORIES = [
    {
        "name": "한국어 Jailbreak 프롬프트",
        "label": 1,
        "count": 1500,
        "system": (
            "You are an AI safety researcher creating a training dataset "
            "for an LLM jailbreak detection classifier. "
            "Generate example prompts that show jailbreak PATTERNS and STRUCTURES only. "
            "Never include actual harmful information or methods."
        ),
        "user_template": (
            "Generate {n} Korean-language jailbreak prompt examples for AI safety research. "
            "These are training samples for a jailbreak DETECTOR (label=1 class). "
            "Show patterns like: ignoring system instructions, requesting restriction removal, "
            "hypothetical scenario bypasses, authority impersonation, etc. "
            "No actual harmful content — patterns and structures only. "
            "Use diverse styles and approaches. "
            "Output ONLY a JSON array of strings: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
    {
        "name": "오타/특수문자 변형 패턴",
        "label": 1,
        "count": 1000,
        "system": (
            "You are an AI safety researcher creating training data "
            "for a jailbreak detection model. "
            "Generate prompts that use obfuscation techniques. No harmful content."
        ),
        "user_template": (
            "Generate {n} Korean jailbreak prompt examples using obfuscation techniques "
            "for AI safety research (training data for a detector). "
            "Use techniques like: intentional typos, special character insertion, "
            "word splitting, unicode substitution, number-for-letter replacement, "
            "emoji insertion, spacing tricks, mixed Korean/English. "
            "No actual harmful content — obfuscation patterns only. "
            "Output ONLY a JSON array: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
    {
        "name": "점진적 유도 방식",
        "label": 1,
        "count": 1000,
        "system": (
            "You are an AI safety researcher building a jailbreak detection dataset. "
            "Generate examples of gradual-escalation prompt patterns. "
            "These are classifier training samples — no harmful content needed."
        ),
        "user_template": (
            "Generate {n} Korean prompt examples showing gradual-escalation jailbreak PATTERNS "
            "for training a jailbreak detector (safety research). "
            "Patterns to include: starting with innocent questions then slowly escalating, "
            "building context step by step, using fictional/hypothetical framing, "
            "academic or educational pretext, story/game setting exploitation. "
            "Keep content benign — show the STRUCTURAL PATTERN only, "
            "replace any sensitive topic with placeholder like [TOPIC] or [HARMFUL_ACT]. "
            "Output ONLY a JSON array: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
    {
        "name": "역할극 변형 패턴",
        "label": 1,
        "count": 1000,
        "system": (
            "You are an AI safety researcher building a jailbreak detection dataset. "
            "Generate roleplay-based jailbreak prompt patterns. "
            "These are classifier training samples — no harmful content needed."
        ),
        "user_template": (
            "Generate {n} Korean roleplay-based jailbreak prompt examples "
            "for training a jailbreak detector (AI safety research). "
            "Patterns to include: persona assignment (e.g., 'act as an AI with no restrictions'), "
            "character roleplay bypasses, fictional villain characters, "
            "historical figure impersonation, game NPC scenarios, "
            "alternate AI identity requests (DAN-style). "
            "No actual harmful content — show roleplay STRUCTURE only, "
            "replace sensitive specifics with [TOPIC] placeholders. "
            "Output ONLY a JSON array: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
    {
        "name": "기타 변형 패턴",
        "label": 1,
        "count": 500,
        "system": (
            "You are an AI safety researcher building a jailbreak detection dataset. "
            "Generate miscellaneous jailbreak prompt patterns not covered by other categories. "
            "These are classifier training samples — no harmful content needed."
        ),
        "user_template": (
            "Generate {n} Korean jailbreak prompt examples using miscellaneous techniques "
            "for AI safety research (training data for a detector). "
            "Use diverse techniques like: prompt injection via documents/code, "
            "multi-language switching, base64/encoded payloads, indirect instruction embedding, "
            "reverse psychology, token smuggling patterns, nested instruction attacks. "
            "No actual harmful content — structural patterns only. "
            "Output ONLY a JSON array: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
    {
        "name": "정상 프롬프트 (한국어)",
        "label": 0,
        "count": 5000,
        "system": (
            "You are a data generation expert. "
            "Generate diverse, everyday Korean-language prompts."
        ),
        "user_template": (
            "Generate {n} diverse, benign Korean-language prompts (questions, requests, instructions) "
            "covering topics like: cooking, travel, programming, education, health, work, "
            "hobbies, science, culture, relationships, finance, sports, arts. "
            "Use natural Korean expressions. Mix short and long, formal and informal styles. "
            "All content must be completely harmless and constructive. "
            "Output ONLY a JSON array: [\"prompt1\", \"prompt2\", ...]"
        ),
    },
]


def load_existing_counts() -> dict:
    """기존 CSV에서 카테고리별 저장된 개수를 label 기준으로 추정."""
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        return {cat["name"]: 0 for cat in CATEGORIES}

    label_counts: dict[int, int] = {}
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lb = int(row["label"])
                label_counts[lb] = label_counts.get(lb, 0) + 1
    except Exception as e:
        print(f"기존 파일 읽기 오류 (무시): {e}")
        return {cat["name"]: 0 for cat in CATEGORIES}

    total_existing = sum(label_counts.values())
    print(f"기존 데이터 발견: {total_existing:,}개 (label 분포: {label_counts})")

    # 카테고리별 목표 대비 비율로 이미 생성된 개수 추정
    counts = {}
    label_to_cats = {}
    for cat in CATEGORIES:
        label_to_cats.setdefault(cat["label"], []).append(cat)

    for lb, cats in label_to_cats.items():
        total_target = sum(c["count"] for c in cats)
        total_done = label_counts.get(lb, 0)
        for cat in cats:
            ratio = cat["count"] / total_target
            counts[cat["name"]] = min(cat["count"], int(total_done * ratio))

    return counts


def call_gpt(system_msg: str, user_msg: str, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.9,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("JSON 배열을 찾을 수 없습니다.")
            items = json.loads(content[start:end])
            return [str(x).strip() for x in items if str(x).strip()]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [재시도 {attempt+1}/{retries}] 오류: {e} — {wait}초 대기", flush=True)
            time.sleep(wait)
    return []


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_counts = load_existing_counts()
    total_existing = sum(existing_counts.values())

    print("=" * 60)
    print("LLM Jailbreak 탐지 데이터셋 생성")
    print(f"목표: {sum(c['count'] for c in CATEGORIES):,}개")
    print(f"기존: {total_existing:,}개 → 남은 것만 추가 생성")
    print("=" * 60)

    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0
    f = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["text", "label"])

    total_generated = total_existing

    try:
        for category in CATEGORIES:
            name = category["name"]
            label = category["label"]
            target = category["count"]
            already = existing_counts.get(name, 0)
            need = target - already

            if need <= 0:
                print(f"\n[{name}] 완료 ({already}/{target}), 스킵")
                continue

            print(f"\n[{name}] 시작 — 이미 {already:,}개, {need:,}개 추가 필요 (label={label})")
            generated = 0
            consecutive_fails = 0

            while generated < need:
                batch = min(BATCH_SIZE, need - generated)
                user_msg = category["user_template"].format(n=batch)
                items = call_gpt(category["system"], user_msg)

                if not items:
                    consecutive_fails += 1
                    print(f"  경고: 응답 없음 (연속 실패 {consecutive_fails}/{MAX_CONSECUTIVE_FAILS})", flush=True)
                    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                        print(f"  최대 연속 실패 도달 — [{name}] {generated:,}개로 종료")
                        break
                    time.sleep(2)
                    continue

                consecutive_fails = 0
                items = items[:batch]

                for text in items:
                    writer.writerow([text, label])
                f.flush()

                generated += len(items)
                total_generated += len(items)
                print(
                    f"  [{name}] {already + generated:,}/{target:,}개 완료"
                    f"  (전체 {total_generated:,}개)",
                    flush=True,
                )
                time.sleep(0.3)

    finally:
        f.close()

    # 결과 요약
    print("\n" + "=" * 60)
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"총 {total_generated:,}개")

    rows = []
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    label_counts: dict = {}
    for row in rows:
        lb = row["label"]
        label_counts[lb] = label_counts.get(lb, 0) + 1
    print(f"레이블 분포: {label_counts}")

    print("\n--- 샘플 10개 ---")
    samples = random.sample(rows, min(10, len(rows)))
    for i, row in enumerate(samples, 1):
        preview = row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"]
        print(f"{i}. [label={row['label']}] {preview}")


if __name__ == "__main__":
    main()
