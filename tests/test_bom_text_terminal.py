"""soup bom emit shows formats, paths and error text as typed, backslashes included."""

import json
import os
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import bom as bom_cmd
from soup_cli.registry.store import RegistryStore
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

runner = CliRunner()

_SHA = "a" * 64
_CONTROLS = "\x9b\u202e"

# A backslash separates directories on Windows, so a name that holds one cannot be a file there.
_NO_BACKSLASH_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="a backslash separates directories on Windows"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately. A backslash before
# an unmatched bracket and a trailing pair of backslashes are what markup rewrites.
_TEXTS = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    pytest.param("x\\[n", id="backslash-bracket"),
    pytest.param("n\\\\", id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=skip_colon_on_windows, id="emoji-code"),
    pytest.param("x\\[n", marks=_NO_BACKSLASH_ON_WINDOWS, id="backslash-bracket"),
    pytest.param("n\\\\", marks=_NO_BACKSLASH_ON_WINDOWS, id="trailing-backslashes"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr(bom_cmd, "console", Console(width=500, height=100))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))


def _coloured(monkeypatch):
    monkeypatch.setattr(
        bom_cmd,
        "console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )


def _emit(*args: str):
    return runner.invoke(
        app,
        [
            "bom", "emit", "--name", "adapter", "--base-model", "org/model",
            "--base-sha", _SHA, "--config-sha", _SHA, *args,
        ],
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _entry_id() -> str:
    with RegistryStore() as store:
        return store.push(
            name="phi", tag="v1", base_model="org/model", task="sft", run_id=None,
            config={"base": "org/model", "task": "sft"}, notes=None,
        )


# --- the format and the energy file -------------------------------------------


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unsupported_format_is_shown_as_typed(value):
    result = _emit("--format", value)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Unsupported --format: {_shown(value)} (use cyclonedx | spdx | both)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_rejected_energy_path_is_shown_as_typed(tmp_path, name):
    target = str(tmp_path / "elsewhere" / f"{name}.json")

    result = _emit("--energy", target)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # quote_path escapes C0 and C1 controls and leaves U+202E and aliases alone.
    quoted = _shown(quote_path(target))
    assert f"Energy path rejected: --energy {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_energy_file_is_shown_as_typed(name):
    result = _emit("--energy", f"sub/{name}.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    # The whole typed path is shown, directory included.
    assert f"Energy file not found: sub/{_shown(name)}.json" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _TEXTS)
def test_an_unreadable_energy_file_is_shown_as_typed(monkeypatch, value):
    Path("energy.json").write_text("{}", encoding="utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self.name == "energy.json":
            raise OSError(value)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _emit("--energy", "energy.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Cannot read energy file: {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("key", _TEXTS)
def test_an_invalid_energy_measurement_names_the_key_as_typed(key):
    Path("energy.json").write_text(json.dumps({key: 1}), encoding="utf-8")

    result = _emit("--energy", "energy.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Invalid energy measurement: " in text
    assert f"unexpected keyword argument '{_shown(key)}'" in text
    assert _not_shown(text)


# --- the written files ----------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_a_failed_write_is_shown_as_typed(tmp_path, name):
    target = str(tmp_path / "elsewhere" / name)

    result = _emit("--output", target)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Write failed: output {_shown(quote_path(target))} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_failed_write_of_both_formats_is_shown_as_typed(tmp_path, name):
    target = str(tmp_path / "elsewhere" / name)

    result = _emit("--format", "both", "--output", target)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"Write failed: output {_shown(quote_path(target + '.cdx.json'))} must stay" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_receipt_shows_the_written_path_as_typed(name):
    result = _emit("--output", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    written = _shown(os.path.realpath(f"{name}.json"))
    assert f"Wrote BOM (cyclonedx) -> {written}" in text
    assert _not_shown(text)
    assert Path(f"{name}.json").is_file()


@pytest.mark.parametrize("name", _NAMES)
def test_the_receipt_of_both_formats_shows_each_path_as_typed(name):
    result = _emit("--format", "both", "--output", name)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    lines = [" ".join(line.split()) for line in strip_ansi(result.output).splitlines()]
    assert lines == [
        f"Wrote CycloneDX BOM -> {_shown(os.path.realpath(f'{name}.cdx.json'))}",
        f"Wrote SPDX BOM -> {_shown(os.path.realpath(f'{name}.spdx.json'))}",
    ]
    assert Path(f"{name}.cdx.json").is_file()
    assert Path(f"{name}.spdx.json").is_file()


# --- the registry -------------------------------------------------------------


@pytest.mark.parametrize("ref", _TEXTS)
def test_a_failed_attach_names_the_reference_as_typed(ref):
    result = _emit("--output", "bom.json", "--attach-to-registry", ref)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"could not attach to registry: registry entry not found: {_shown(ref)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_attached_receipt_shows_the_bom_path_as_typed(name):
    entry_id = _entry_id()

    result = _emit("--output", f"{name}.json", "--attach-to-registry", entry_id)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    path = _shown(os.path.realpath(f"{name}.json"))
    assert f"Attached bom to registry entry {entry_id} ({path})" in text
    assert _not_shown(text)


# --- lines that stay ------------------------------------------------------------


def test_a_plain_unsupported_format_keeps_its_line():
    result = _emit("--format", "yaml")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert strip_ansi(result.output) == "Unsupported --format: yaml (use cyclonedx | spdx | both)\n"


def test_malformed_energy_json_keeps_its_line():
    Path("energy.json").write_text("{not json", encoding="utf-8")

    result = _emit("--energy", "energy.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert _plain(result).startswith("Malformed JSON in energy file: Expecting property name")


def test_the_machine_readable_bom_goes_to_stdout_untouched():
    result = _emit("--format", "spdx")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(result.output)["name"]


# --- styles ---------------------------------------------------------------------


def test_the_refusals_are_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)

    refused_format = _emit("--format", "yaml")
    missing = _emit("--energy", "missing.json")
    attach = _emit("--output", "bom.json", "--attach-to-registry", "nope")

    assert refused_format.output == (
        "\x1b[31mUnsupported --format: yaml (use cyclonedx | spdx | both)\x1b[0m\n"
    )
    assert missing.output == "\x1b[31mEnergy file not found: missing.json\x1b[0m\n"
    assert attach.output.endswith(
        "\x1b[31mError:\x1b[0m could not attach to registry: registry entry not found: nope\n"
    )


def test_a_failed_write_is_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)

    result = _emit("--output", "../outside.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert result.output.startswith("\x1b[31mWrite failed: output '../outside.json' must stay")
    assert result.output.endswith("\x1b[0m\n")


def test_the_receipts_keep_their_green_labels(monkeypatch):
    _coloured(monkeypatch)

    single = _emit("--output", "one.json")
    both = _emit("--format", "both", "--output", "two")

    assert single.output == (
        f"\x1b[32mWrote BOM (cyclonedx)\x1b[0m -> {os.path.realpath('one.json')}\n"
    )
    assert both.output == (
        f"\x1b[32mWrote CycloneDX BOM\x1b[0m -> {os.path.realpath('two.cdx.json')}\n"
        f"\x1b[32mWrote SPDX BOM\x1b[0m -> {os.path.realpath('two.spdx.json')}\n"
    )


def test_the_attached_receipt_has_a_green_label_a_bold_id_and_a_dim_path(monkeypatch):
    _coloured(monkeypatch)
    entry_id = _entry_id()

    result = _emit("--output", "three.json", "--attach-to-registry", entry_id)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    path = os.path.realpath("three.json")
    assert result.output.endswith(
        f"\x1b[32mAttached\x1b[0m bom to registry entry \x1b[1m{entry_id}\x1b[0m "
        f"\x1b[2m({path})\x1b[0m\n"
    )


def test_the_energy_and_write_refusals_are_red_as_a_whole(monkeypatch):
    _coloured(monkeypatch)
    Path("energy.json").write_text("{}", encoding="utf-8")
    Path("bad.json").write_text(json.dumps({"bogus": 1}), encoding="utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self.name == "energy.json":
            raise OSError("denied")
        return real_read_text(self, *args, **kwargs)

    outside = _emit("--energy", "../x.json")
    invalid = _emit("--energy", "bad.json")
    both = _emit("--format", "both", "--output", "../outside")
    monkeypatch.setattr(Path, "read_text", deny)
    unreadable = _emit("--energy", "energy.json")

    assert outside.output == (
        "\x1b[31mEnergy path rejected: --energy '../x.json' must stay under cwd\x1b[0m\n"
    )
    assert unreadable.output == "\x1b[31mCannot read energy file: denied\x1b[0m\n"
    assert invalid.output.startswith("\x1b[31mInvalid energy measurement: ")
    assert invalid.output.endswith("unexpected keyword argument 'bogus'\x1b[0m\n")
    assert both.output == (
        "\x1b[31mWrite failed: output '../outside.cdx.json' must stay under cwd\x1b[0m\n"
    )
