from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from supportsage.data import SYSTEM_PROMPT


class SupportSageModel:
    def __init__(self, model_name: str, adapter_path: str | None = None) -> None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 180, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args.update({"temperature": temperature, "top_p": 0.9})
        outputs = self.model.generate(**inputs, **generation_args)
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
