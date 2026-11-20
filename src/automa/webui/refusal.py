"""Refusals the web UI answers with an error page, translatable by code (D147).

A :class:`Refusal` keeps the exact English ``message`` it always had (tests, logs and
English pages are unchanged) and, when the message is one of the interface's own
sentences, a structured ``code`` with its ``params``: the Persian page shows
``t("refusal.<code>")`` instead of searching the English text. A message built from
an underlying error (``str(e)`` of a store, workspace or planner failure) has no code
of its own; the Persian page then shows a short Persian lead and the English message
as a technical detail, left-to-right. Pure: no files, no network.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Self

from automa.webui.i18n import MESSAGES, t


def refusal_key(code: str) -> str:
    return f"refusal.{code}"


class Refusal(Exception):
    """A refused request: HTTP ``status``, the English ``message`` and, optionally,
    the ``code`` + ``params`` that translate it (or a translated ``lead`` sentence
    in front of an untranslated message)."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        params: Mapping[str, str | int] | None = None,
        lead: str | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None and refusal_key(code) not in MESSAGES:
            raise ValueError(f"unknown refusal code {code!r}")
        if lead is not None and refusal_key(lead) not in MESSAGES:
            raise ValueError(f"unknown refusal lead {lead!r}")
        if code is not None and lead is not None:
            raise ValueError("a refusal has a code or a lead, not both")
        self.status = status
        self.message = message
        self.code = code
        self.params: Mapping[str, str | int] = MappingProxyType(dict(params or {}))
        self.lead = lead

    @classmethod
    def coded(cls, http_status: int, code: str, /, **params: str | int) -> Self:
        """A refusal whose English message *is* the ``en`` template of ``code``
        (positional-only, so a parameter may be called ``status`` or ``code``)."""
        message = t(refusal_key(code), "en", **params)
        return cls(http_status, message, code=code, params=params)
