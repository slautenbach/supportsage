from __future__ import annotations

import re
from collections import Counter

from rouge_score import rouge_scorer

PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}]+\}\}")


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(prediction.lower().split())
    expected = Counter(reference.lower().split())
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def placeholder_recall(prediction: str, reference: str) -> float:
    expected = set(PLACEHOLDER_PATTERN.findall(reference))
    if not expected:
        return 1.0
    found = set(PLACEHOLDER_PATTERN.findall(prediction))
    return len(expected & found) / len(expected)


def score_response(prediction: str, reference: str) -> dict[str, float]:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {
        "rouge_l": rouge.score(reference, prediction)["rougeL"].fmeasure,
        "token_f1": token_f1(prediction, reference),
        "placeholder_recall": placeholder_recall(prediction, reference),
    }

