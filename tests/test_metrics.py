import pytest

from supportsage.metrics import placeholder_recall, score_response, token_f1


def test_identical_responses_score_one() -> None:
    scores = score_response("Your order is ready", "Your order is ready")
    assert all(value == pytest.approx(1.0) for value in scores.values())


def test_placeholder_recall_detects_missing_fields() -> None:
    reference = "Track {{Order Number}} at {{Website URL}}."
    prediction = "Track {{Order Number}} in your account."
    assert placeholder_recall(prediction, reference) == 0.5


def test_token_f1_handles_empty_prediction() -> None:
    assert token_f1("", "expected text") == 0.0

