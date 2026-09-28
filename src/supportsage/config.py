from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    output_dir: str = "artifacts/supportsage-qwen-lora"
    seed: int = 42
    train_samples: int = 4000
    eval_samples: int = 400
    max_length: int = 512
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        return cls(**raw)
