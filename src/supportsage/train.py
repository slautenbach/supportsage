from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from supportsage.config import TrainConfig


def build_quantization_config(enabled: bool) -> BitsAndBytesConfig | None:
    if not enabled or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def train(config: TrainConfig, smoke: bool = False) -> None:
    data_files = {"train": "data/processed/train.jsonl", "test": "data/processed/eval.jsonl"}
    if not all(Path(path).exists() for path in data_files.values()):
        raise FileNotFoundError("Run `supportsage-prepare` before training.")

    datasets = load_dataset("json", data_files=data_files)
    if smoke:
        datasets["train"] = datasets["train"].select(range(min(16, len(datasets["train"]))))
        datasets["test"] = datasets["test"].select(range(min(8, len(datasets["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    output_dir = f"{config.output_dir}-smoke" if smoke else config.output_dir
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if smoke else config.num_train_epochs,
        max_steps=2 if smoke else -1,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        max_length=min(config.max_length, 256) if smoke else config.max_length,
        logging_steps=1 if smoke else config.logging_steps,
        eval_strategy="steps",
        eval_steps=1 if smoke else config.eval_steps,
        save_strategy="steps",
        save_steps=1 if smoke else config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        seed=config.seed,
        dataset_num_proc=None,
        model_init_kwargs={
            "device_map": "auto" if torch.cuda.is_available() else None,
            "quantization_config": build_quantization_config(config.use_4bit),
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        },
    )
    trainer = SFTTrainer(
        model=config.model_name,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SupportSage with LoRA.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", help="Override the model ID with a local or remote path.")
    parser.add_argument(
        "--smoke", action="store_true", help="Run two training steps for validation."
    )
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    if args.model:
        config = replace(config, model_name=args.model)
    train(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
