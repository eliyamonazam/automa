"""Tests for ``automa models`` and ``automa models import`` (D95). No model calls."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from automa import cli
from automa.config import AutomaConfig

CONFIG = """
base_url = "https://gateway.example/v1"
[tiers]
small = "claude-haiku-5-5"
large = "m-legacy"
[prices.m-legacy]
input_per_mtok = 3
output_per_mtok = 9
[models."claude-haiku-5-5"]
family = "anthropic"
capabilities = ["text", "code"]
input_per_mtok = 1
output_per_mtok = 5
[models."gpt-5"]
family = "openai"
capabilities = ["text", "code", "vision"]
input_per_mtok = 1.25
output_per_mtok = 10
[models."gpt-image-1"]
family = "openai"
capabilities = ["image_out"]
input_per_mtok = 5
output_per_mtok = 40
"""

PRICE_LIST = """model | input | output
claude-sonnet-5-5 | 969 | 4845
gpt-5 | 400 | 3200
m-legacy | 10 | 20
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for var in ("AUTOMA_CONFIG", "AUTOMA_API_KEY", "AUTOMA_BASE_URL", "AUTOMA_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)

    def boom(_: AutomaConfig) -> None:
        raise AssertionError("no provider may be built")

    monkeypatch.setattr(cli, "make_provider", boom)
    path = tmp_path / "automa.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def _models(config: Path, *args: str) -> Any:
    return CliRunner().invoke(cli.app, ["models", *args, "--config", str(config)])


def test_list_all(config: Path) -> None:
    r = _models(config)
    assert r.exit_code == 0, r.output
    lines = r.output.splitlines()
    assert lines[0].startswith("claude-haiku-5-5  family: anthropic")
    assert "tiers: small" in lines[0]
    assert lines[1].startswith("gpt-5 ") and "text, code, vision" in lines[1]
    assert "$1.25 in / $10 out per 1M tokens" in lines[1]
    assert lines[2].startswith("gpt-image-1 ")
    assert lines[3] == "priced in [prices] only (not catalogued): m-legacy"


def test_filter_capability(config: Path) -> None:
    r = _models(config, "--capability", "image_out")
    assert r.exit_code == 0
    assert [ln.split()[0] for ln in r.output.splitlines()] == ["gpt-image-1"]


def test_filter_family(config: Path) -> None:
    r = _models(config, "--family", "openai", "--capability", "code")
    assert [ln.split()[0] for ln in r.output.splitlines()] == ["gpt-5"]


def test_filter_no_match(config: Path) -> None:
    r = _models(config, "--family", "google")
    assert r.exit_code == 0
    assert r.output.strip() == "no catalogued model matches the filter"


def test_bad_capability_and_family(config: Path) -> None:
    assert _models(config, "--capability", "telepathy").exit_code == 2
    r = _models(config, "--family", "Open AI")
    assert r.exit_code == cli.EXIT_USER
    assert "invalid family" in r.output


def test_json(config: Path) -> None:
    r = _models(config, "--json")
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert [d["id"] for d in data] == ["claude-haiku-5-5", "gpt-5", "gpt-image-1"]
    assert data[0] == {
        "id": "claude-haiku-5-5",
        "family": "anthropic",
        "capabilities": ["text", "code"],
        "input_per_mtok": 1.0,
        "output_per_mtok": 5.0,
        "tiers": ["small"],
    }


def test_json_filtered_empty(config: Path) -> None:
    r = _models(config, "--json", "--family", "google")
    assert json.loads(r.output) == []


def test_empty_catalog(tmp_path: Path, config: Path) -> None:
    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    r = _models(empty)
    assert r.exit_code == 0
    assert "the model catalog is empty" in r.output


def test_broken_config_is_config_error(tmp_path: Path, config: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[models."x"]\nfamily = "a"\n', encoding="utf-8")
    assert _models(bad).exit_code == cli.EXIT_CONFIG


def test_list_writes_nothing(tmp_path: Path, config: Path) -> None:
    before = sorted(p.name for p in tmp_path.iterdir())
    _models(config)
    _models(config, "--json")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --- import -----------------------------------------------------------------------


def _import(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["models", "import", *args])


def _write_list(tmp_path: Path, text: str = PRICE_LIST) -> Path:
    path = tmp_path / "prices.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_import_prints_valid_toml(tmp_path: Path, config: Path) -> None:
    path = _write_list(tmp_path)
    before = config.read_text(encoding="utf-8")
    r = _import(str(path), "--format", "1xai", "--rate", "323000")
    assert r.exit_code == 0, r.output
    data = tomllib.loads(r.output)
    assert data["models"]["claude-sonnet-5-5"] == {
        "family": "anthropic",
        "capabilities": ["text", "code"],
        "input_per_mtok": 3,
        "output_per_mtok": 15,
    }
    assert data["models"]["gpt-5"]["input_per_mtok"] == 1.238391
    # The config file is never written.
    assert config.read_text(encoding="utf-8") == before


def test_import_default_format_and_persian_rate(tmp_path: Path, config: Path) -> None:
    r = _import(str(_write_list(tmp_path)), "--rate", "۳۲۳٬۰۰۰")
    assert r.exit_code == 0, r.output
    assert "323000 toman per USD" in r.output


def test_import_requires_rate(tmp_path: Path, config: Path) -> None:
    r = _import(str(_write_list(tmp_path)))
    assert r.exit_code == 2


@pytest.mark.parametrize("rate", ["0", "-1", "abc"])
def test_import_bad_rate(tmp_path: Path, config: Path, rate: str) -> None:
    r = _import(str(_write_list(tmp_path)), "--rate", rate)
    assert r.exit_code == cli.EXIT_USER
    assert "rate must be" in r.output


def test_import_unknown_format(tmp_path: Path, config: Path) -> None:
    r = _import(str(_write_list(tmp_path)), "--format", "csv", "--rate", "1")
    assert r.exit_code == 2


def test_import_missing_file(tmp_path: Path, config: Path) -> None:
    r = _import(str(tmp_path / "nope.txt"), "--rate", "323000")
    assert r.exit_code == cli.EXIT_USER
    assert "not found" in r.output


def test_import_not_utf8(tmp_path: Path, config: Path) -> None:
    path = tmp_path / "prices.txt"
    path.write_bytes(b"gpt-5 \xff 1\n")
    r = _import(str(path), "--rate", "323000")
    assert r.exit_code == cli.EXIT_USER
    assert "not UTF-8" in r.output


def test_import_too_large(tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MAX_PRICE_LIST_BYTES", 10)
    r = _import(str(_write_list(tmp_path)), "--rate", "323000")
    assert r.exit_code == cli.EXIT_USER
    assert "larger than" in r.output


def test_import_bad_line_prints_nothing_on_stdout(tmp_path: Path, config: Path) -> None:
    path = _write_list(tmp_path, PRICE_LIST + "broken line here\n")
    r = CliRunner().invoke(cli.app, ["models", "import", str(path), "--rate", "323000"])
    assert r.exit_code == cli.EXIT_USER
    assert "line 5" in r.output
    assert "[models." not in r.output


def test_import_escapes_control_chars_in_errors(tmp_path: Path, config: Path) -> None:
    path = _write_list(tmp_path, "gpt-5 1 1\nbad\x1b[31mid 1 1\n")
    r = _import(str(path), "--rate", "323000")
    assert r.exit_code == cli.EXIT_USER
    assert "\x1b" not in r.output


def test_import_warns_about_already_priced_models(tmp_path: Path, config: Path) -> None:
    r = _import(str(_write_list(tmp_path)), "--rate", "323000", "--config", str(config))
    assert r.exit_code == 0
    assert "warning: gpt-5 is already priced in [models]" in r.output
    assert "warning: m-legacy is already priced in [prices]" in r.output
    assert "claude-sonnet-5-5 is already" not in r.output


def test_import_with_broken_config_is_config_error(tmp_path: Path, config: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("tiers = 3\n", encoding="utf-8")
    r = _import(str(_write_list(tmp_path)), "--rate", "323000", "--config", str(bad))
    assert r.exit_code == cli.EXIT_CONFIG


def test_help_lists_models(config: Path) -> None:
    r = CliRunner().invoke(cli.app, ["--help"])
    assert "models" in r.output
    r = CliRunner().invoke(cli.app, ["models", "--help"])
    assert "--capability" in r.output and "--family" in r.output and "import" in r.output
    r = CliRunner().invoke(cli.app, ["models", "import", "--help"])
    assert "--rate" in r.output and "--format" in r.output
