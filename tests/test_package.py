"""Smoke tests for the package skeleton."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from typer.testing import CliRunner

import automa
from automa.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", automa.__version__)


def test_version_matches_pyproject() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == automa.__version__


def test_cli_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == automa.__version__


def test_env_example_has_no_real_secrets() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = [
        line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    assert values, ".env.example should define sample variables"
    for value in values:
        # Real provider keys tend to be long opaque tokens; samples must be placeholders.
        assert not re.fullmatch(r"(sk|or|ghp|gho)[-_][A-Za-z0-9_-]{16,}", value)


def test_gitignore_excludes_env_and_workspace() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "workspace/" in lines
