"""Guards for the CI workflow: it must run the checks and never touch secrets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_exists() -> None:
    assert WORKFLOW.is_file()


@pytest.mark.parametrize("command", ["ruff check", "ruff format --check", "pytest"])
def test_runs_required_checks(text: str, command: str) -> None:
    assert re.search(rf"run:\s*{re.escape(command)}\b", text)


def test_never_references_secrets(text: str) -> None:
    # Tests run with FakeProvider only; CI must not have access to any real key.
    assert "secrets." not in text
    assert "API_KEY" not in text.upper()


def test_read_only_token(text: str) -> None:
    assert re.search(r"^permissions:\s*\n\s+contents:\s*read\s*$", text, re.MULTILINE)
    assert "write" not in text


def test_structure_with_yaml(text: str) -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(text)
    triggers = data.get("on", data.get(True))  # YAML 1.1 parses bare `on` as True
    assert "push" in triggers
    job = data["jobs"]["test"]
    versions = job["strategy"]["matrix"]["python-version"]
    assert "3.11" in versions  # minimum supported version from pyproject
    assert job["timeout-minutes"] <= 30
