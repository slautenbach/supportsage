from pathlib import Path

import pytest

from supportsage.config import TrainConfig


def test_config_loads_defaults_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train_samples: 24\nuse_4bit: false\n", encoding="utf-8")
    config = TrainConfig.from_yaml(config_path)
    assert config.train_samples == 24
    assert config.use_4bit is False
    assert config.lora_r == 16


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mystery: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        TrainConfig.from_yaml(config_path)

