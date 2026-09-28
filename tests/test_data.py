from supportsage.data import SYSTEM_PROMPT, to_messages


def test_to_messages_preserves_metadata() -> None:
    row = {
        "instruction": " help me ",
        "response": " certainly ",
        "category": "ORDER",
        "intent": "cancel_order",
    }
    result = to_messages(row)
    assert result["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "help me"},
        {"role": "assistant", "content": "certainly"},
    ]
    assert result["intent"] == "cancel_order"

