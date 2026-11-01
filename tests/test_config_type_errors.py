"""Wrong field types stay within the config validation error contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from pydantic import BaseModel, ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi


def _reachable_models(annotation: Any) -> set[type[BaseModel]]:
    """Follow nested model fields, including models inside containers/unions."""
    found: set[type[BaseModel]] = set()
    pending = [annotation]
    while pending:
        current = pending.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            if current in found:
                continue
            found.add(current)
            pending.extend(field.annotation for field in current.model_fields.values())
        else:
            pending.extend(get_args(current))
    return found


@pytest.mark.parametrize("value", [0, True, "false", [], {}, 1.5])
def test_all_config_fields_validate_or_raise_validation_error(value: Any) -> None:
    escaped = []
    for model in sorted(_reachable_models(SoupConfig), key=lambda item: item.__name__):
        for field in model.model_fields:
            try:
                model.model_validate({field: value})
            except ValidationError:
                pass
            except TypeError as exc:
                escaped.append(f"{model.__name__}.{field}: {exc}")
    assert not escaped, "TypeErrors escaped field validation:\n" + "\n".join(escaped)


_WRONG_TYPES = [
    ("minillm_enabled", 0),
    ("echo_trap_halt", "false"),
    ("rl_checkpoint_include_optimizer", 0),
    ("reward_hack_rollback", "false"),
    ("fp8_attention", 0),
    ("num_labels", True),
    ("distill_temperature", True),
    ("unlearn_method", True),
]


def _config_text(field: str, value: Any) -> str:
    return yaml.safe_dump({
        "base": "org/model",
        "data": {"train": "train.jsonl"},
        "training": {field: value},
    })


@pytest.mark.parametrize(("field", "value"), _WRONG_TYPES)
def test_wrong_training_type_names_field_in_schema_and_string_loader(
    field: str, value: Any,
) -> None:
    with pytest.raises(ValidationError) as caught:
        SoupConfig.model_validate(yaml.safe_load(_config_text(field, value)))
    errors = caught.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("training", field)
    assert errors[0]["type"] == "value_error"
    assert field in errors[0]["msg"]

    with pytest.raises(ValueError, match=f"training -> {field}:"):
        load_config_from_string(_config_text(field, value))


@pytest.mark.parametrize("command", ["train", "plan"])
@pytest.mark.parametrize(("field", "value"), _WRONG_TYPES)
def test_wrong_training_type_has_consistent_cli_refusal(
    command: str, field: str, value: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("soup.yaml").write_text(_config_text(field, value), encoding="utf-8")
    result = CliRunner().invoke(app, [command, "--config", "soup.yaml"])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, output
    assert "Config validation error:" in output
    assert f"training -> {field}:" in output
    assert "TypeError:" not in output
    assert not Path("soup.tfstate").exists()


def test_valid_training_values_and_runtime_helper_contract_are_preserved() -> None:
    config = load_config_from_string(_config_text("minillm_enabled", False))
    assert TrainingConfig(num_labels=3).num_labels == 3
    assert config.training.minillm_enabled is False
    assert config.training.distill_temperature == TrainingConfig().distill_temperature
    # These existing info-consuming validators must retain their field context,
    # including on the declared minimum Pydantic 2.0.
    assert TrainingConfig(grpo_beta=0.2).grpo_beta == 0.2
    with pytest.raises(ValidationError, match="grpo_beta must not be a boolean"):
        TrainingConfig(grpo_beta=True)

    # Runtime callers retain the helper's TypeError; only the schema translates it.
    from soup_cli.utils.classifier import validate_num_labels

    with pytest.raises(TypeError, match="num_labels must not be bool"):
        validate_num_labels(True)
