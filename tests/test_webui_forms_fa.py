"""Tests for the Persian new-idea form, thinking page and questions form (D145).

Pure rendering; no model, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from automa.webui import render as rd
from test_webui_i18n import LATIN_WORD, _strip_data
from test_webui_ideas import TOKEN, Env

FORMS = rd.ActionForms("tok")


def _interface(doc: str) -> str:
    """Interface text only: without the recorded goal/questions and form attributes."""
    doc = re.sub(r'<p class="goal" dir="auto">.*?</p>', "", doc, flags=re.S)
    doc = re.sub(r"<code dir=\"ltr\">.*?</code>", "", doc, flags=re.S)
    return _strip_data(doc)


def test_persian_new_idea_form() -> None:
    doc = rd.render_new_idea(FORMS, "fa")
    assert '<html lang="fa" dir="rtl">' in doc
    assert "ایدهٔ جدید" in doc and "ساختن برنامه" in doc
    assert 'placeholder="پروژه قراره چه کاری انجام بده؟"' in doc
    assert 'name="csrf" value="tok"' in doc and 'name="ask"' in doc and "required" in doc
    assert LATIN_WORD.findall(_interface(doc)) == []


@pytest.mark.parametrize("phase", ["questions", "plan"])
def test_persian_thinking_page(phase: str) -> None:
    doc = rd.render_thinking("p-20261120-101010-abcdef", phase, 12, "fa")
    assert '<html lang="fa" dir="rtl">' in doc
    assert "۱۲ ثانیه گذشته" in doc and "⏳ در حال فکر کردن…" in doc
    assert '<code dir="ltr">p-20261120-101010-abcdef</code>' in doc
    assert 'http-equiv="refresh"' in doc
    assert LATIN_WORD.findall(_interface(doc)) == []


def test_thinking_phases_differ_and_negative_elapsed_is_zero() -> None:
    q = rd.render_thinking("p1", "questions", -5, "fa")
    p = rd.render_thinking("p1", "plan", 0, "fa")
    assert "سؤال" in q and "برنامه و پیش‌بینی هزینه" in p
    assert "۰ ثانیه" in q


def test_persian_questions_form_keeps_recorded_text_and_escapes_it() -> None:
    doc = rd.render_clarify(
        "p-1", "یک <b>ابزار</b>", ["Which OS?", "<script>x</script>"], FORMS, "fa"
    )
    assert '<html lang="fa" dir="rtl">' in doc
    assert "<script>x</script>" not in doc and "&lt;script&gt;" in doc
    assert "&lt;b&gt;" in doc
    assert "۱. Which OS?" in doc and "۲. " in doc
    assert 'name="a0"' in doc and 'name="a1"' in doc and 'name="csrf" value="tok"' in doc
    assert "فرستادن جواب‌ها و ساختن برنامه" in doc
    # The questions are model output (recorded data); remove them before checking.
    for recorded in ("Which OS?", "&lt;script&gt;x&lt;/script&gt;"):
        doc = doc.replace(recorded, "")
    interface = _interface(doc)
    assert LATIN_WORD.findall(interface) == []


def test_english_forms_are_unchanged() -> None:
    idea = rd.render_new_idea(FORMS)
    assert "<h1>New idea</h1>" in idea and "Create plan" in idea
    assert "Ask clarifying questions first" in idea and "<title>Automa — new idea" in idea
    thinking = rd.render_thinking("p1", "plan", 3)
    assert "The Executive is making the plan and the cost estimate." in thinking
    assert "3 s so far · project p1" in thinking
    clarify = rd.render_clarify("p1", "goal", ["q?"], FORMS)
    assert "<h1>Questions for p1</h1>" in clarify and "← p1</a>" in clarify
    assert "1. q?" in clarify and "Automa — p1 questions" in clarify


def test_unknown_language_is_refused() -> None:
    with pytest.raises((ValueError, KeyError)):
        rd.render_new_idea(FORMS, "de")  # type: ignore[arg-type]


def test_persian_server_shows_persian_idea_and_question_pages(tmp_path: Path) -> None:
    env = Env(tmp_path, language="fa")
    form = env.get("/new").body.decode("utf-8")
    assert '<html lang="fa" dir="rtl">' in form and "ایدهٔ جدید" in form
    r = env.new("یک سایت کوچک")
    (pid,) = env.store.list_ids()
    assert dict(r.headers)["Location"] == f"/projects/{pid}/clarify"
    page = env.get(f"/projects/{pid}/clarify").body.decode("utf-8")
    assert '<html lang="fa" dir="rtl">' in page and "سؤال‌های پروژهٔ" in page
    assert "Who is it for?" in page and f'name="csrf" value="{TOKEN}"' in page
