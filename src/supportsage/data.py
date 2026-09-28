from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

SYSTEM_PROMPT = (
    "You are SupportSage, a concise and empathetic customer-support assistant. "
    "Give practical next steps, preserve placeholders in double braces, and never invent policy."
)


def to_messages(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"].strip()},
            {"role": "assistant", "content": row["response"].strip()},
        ],
        "category": row["category"],
        "intent": row["intent"],
    }


def build_splits(
    dataset_name: str,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> DatasetDict:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed)
    total = min(len(dataset), train_samples + eval_samples)
    selected = dataset.select(range(total)).map(to_messages, remove_columns=dataset.column_names)
    eval_size = min(eval_samples, max(1, total // 5))
    return selected.train_test_split(test_size=eval_size, seed=seed, shuffle=True)


def save_jsonl(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic SupportSage data splits.")
    parser.add_argument(
        "--dataset", default="bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--eval-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    splits = build_splits(args.dataset, args.train_samples, args.eval_samples, args.seed)
    save_jsonl(splits["train"], args.output_dir / "train.jsonl")
    save_jsonl(splits["test"], args.output_dir / "eval.jsonl")
    print(f"Wrote {len(splits['train'])} train and {len(splits['test'])} eval examples.")


if __name__ == "__main__":
    main()
