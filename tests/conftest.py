"""Shared test setup.

CI runners (GitHub Actions) set GITHUB_ACTIONS, which makes Typer/Rich force a
colour terminal. Help output then contains ANSI escape codes between characters
(e.g. "-" + "-role"), so plain substring checks such as ``"--role" in output``
fail only in CI. Typer reads these variables when it is first imported, so they
are cleared here, at conftest import time, before any test imports the CLI.
"""

from __future__ import annotations

import os

for _name in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS", "TTY_COMPATIBLE"):
    os.environ.pop(_name, None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ.setdefault("COLUMNS", "120")
