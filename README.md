# SupportSage

SupportSage is a reproducible LLM fine-tuning project that adapts
`Qwen/Qwen2.5-0.5B-Instruct` to customer-support response generation with LoRA.
It is designed to run on a 4 GB NVIDIA GPU and to show the full ML lifecycle:
data preparation, parameter-efficient training, held-out evaluation, inference,
an interactive demo, tests, and CI.

## What this project demonstrates

- Converts 26,872 public support conversations into chat-formatted examples.
- Uses deterministic train/evaluation splits to prevent cherry-picked results.
- Fine-tunes only low-rank adapters with optional 4-bit quantization.
- Compares responses using ROUGE-L, token F1, and placeholder preservation.
- Ships a Gradio demo and automated quality checks.

## Architecture

```text
Bitext dataset -> deterministic split -> Qwen chat template -> QLoRA training
                                                               |
held-out prompts -> base model + adapter -> metrics JSON <------+
                                      |
                                      +-> Gradio demo
```

## Quick start

Requirements: NVIDIA driver, Git, and `uv`. The lockfile pins a Python 3.12
environment even if the system Python is newer.

```powershell
uv sync --extra dev --extra quantization
uv run supportsage-prepare
uv run supportsage-train --smoke
uv run pytest
```

Run the full experiment:

```powershell
uv run supportsage-train
uv run supportsage-evaluate --output artifacts/base-evaluation.json
uv run supportsage-evaluate --adapter artifacts/supportsage-qwen-lora `
  --output artifacts/adapter-evaluation.json
uv run supportsage-demo
```

## Experiment design

The default configuration uses 4,000 training and 400 held-out examples, two
epochs, a global batch size of 16, and LoRA rank 16. Evaluation is intentionally
separate from training. The three metrics capture lexical similarity, content
overlap, and whether operational placeholders such as `{{Order Number}}` survive
generation.

Results belong in `artifacts/` and are excluded from Git because model adapters
and checkpoints are generated outputs. After a full run, add the metric values
to the table below.

| Model | ROUGE-L | Token F1 | Placeholder recall |
| --- | ---: | ---: | ---: |
| Base Qwen2.5-0.5B-Instruct | pending | pending | pending |
| SupportSage LoRA | pending | pending | pending |

### Verified smoke run

The two-step GPU smoke test completed on an NVIDIA RTX 3050 Ti (4 GB VRAM):

- Training runtime: 24.06 seconds
- Training loss: 3.09
- Evaluation loss: 2.768
- Evaluation token accuracy: 0.5224
- Saved adapter size: 4.35 MB

These figures validate the pipeline and are not presented as final model-quality results.

## Project layout

```text
configs/train.yaml          Training and LoRA hyperparameters
src/supportsage/data.py     Dataset download, formatting, and splitting
src/supportsage/train.py    Quantized LoRA fine-tuning
src/supportsage/evaluate.py Held-out generation and metrics
src/supportsage/app.py      Interactive Gradio demo
tests/                      Fast unit tests
```

## Resume bullet

> Built SupportSage, a domain-specific customer-support LLM by fine-tuning a
> 494M-parameter Qwen model with 4-bit LoRA; created a deterministic data and
> evaluation pipeline, measured baseline-to-adapter quality, and shipped an
> interactive Gradio demo with CI-tested Python tooling.

Only replace the bullet's claims with measured numbers after the full run.

## Data and model cards

- Dataset: [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

The source dataset and base model retain their respective licenses and terms.
This project does not include customer data or production credentials.
