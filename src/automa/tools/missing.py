"""Recognise "program not installed" in a recorded tool error (M13, D152).

``run_command`` reports a program that is not on ``PATH`` as
``program not found: '<name>'`` (``repr`` of the program). This module only
*reads* that text back: :func:`missing_program` returns the program name when it
is a plain, safe name (letters, digits, ``.+_-``; no path, space or quote), and
:func:`install_command` suggests the Ubuntu (WSL, D90) command to install it.

Nothing here runs a command: the suggestion is text the owner may type himself.
A name that does not look safe gives ``None``, so recorded text can never put an
arbitrary command in front of the owner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

PROGRAM_NOT_FOUND = "program not found"

_NAME = r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}"
_PATTERN = re.compile(rf"{PROGRAM_NOT_FOUND}: '({_NAME})'(?![^\s;,)\]])")

# Programs whose Ubuntu package has a different name.
APT_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        "python": "python3",
        "pip": "python3-pip",
        "pip3": "python3-pip",
        "node": "nodejs",
        "javac": "default-jdk",
        "java": "default-jre",
        "go": "golang-go",
        "rustc": "rustc",
        "cargo": "cargo",
        "c++": "g++",
        "cc": "gcc",
        "pytest": "python3-pytest",
        "sqlite3": "sqlite3",
        "convert": "imagemagick",
        "ffmpeg": "ffmpeg",
    }
)


def missing_program(text: str | None) -> str | None:
    """The program named in a ``program not found: '<name>'`` error, if safe."""
    if not text:
        return None
    match = _PATTERN.search(text)
    return match.group(1) if match else None


def install_command(program: str) -> str:
    """``sudo apt install -y <package>`` for a name from :func:`missing_program`."""
    if not re.fullmatch(_NAME, program):
        raise ValueError(f"not a plain program name: {program!r}")
    return f"sudo apt install -y {APT_PACKAGES.get(program, program)}"
