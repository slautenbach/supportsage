from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from supportsage.inference import SupportSageModel
from supportsage.metrics import score_response


def evaluate(
    model_name: str, adapter_path: str | None, data_path: str, limit: int
) -> dict[str, float]:
    model = SupportSageModel(model_name, adapter_path)
    dataset = load_dataset("json", data_files=data_path, split="train").select(range(limit))
    rows = []
    for row in dataset:
        prediction = model.generate(row["messages"][1]["content"])
        reference = row["messages"][2]["content"]
        rows.append(score_response(prediction, reference))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base or fine-tuned SupportSage.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data/processed/eval.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    scores = evaluate(args.model, args.adapter, args.data, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
