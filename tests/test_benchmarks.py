"""Tests for the benchmark suite (``automa.benchmarks`` and ``benchmarks/*.toml``).

No model is involved anywhere: benchmark success is decided by deterministic
checks only. Each shipped benchmark is checked against a hand-written reference
solution (must pass) and broken variants (must fail on the right criterion).
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import pytest

from automa.agents.verifier import verifier_policy
from automa.benchmarks import (
    MAX_BENCHMARK_BUDGET_USD,
    MAX_BENCHMARK_FILE_BYTES,
    Benchmark,
    BenchmarkError,
    BenchmarkKind,
    BenchmarkOutcome,
    BenchmarkSuite,
    evaluate_benchmark,
    load_benchmark,
    parse_benchmark,
)
from automa.config import BudgetLimits
from automa.events import MemoryEventSink
from automa.plan.models import CheckResult

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
SRC_PKG = REPO / "src" / "automa" / "benchmarks"


@pytest.fixture(scope="module")
def suite() -> BenchmarkSuite:
    return BenchmarkSuite.from_dir(BENCH_DIR)


def _files(tmp_path: Path, contents: dict[str, str]) -> Path:
    files = tmp_path / "files"
    files.mkdir()
    for name, text in contents.items():
        (files / name).write_text(text, encoding="utf-8")
    return files


def _failed(outcome: BenchmarkOutcome) -> set[int]:
    return {c.criterion for c in outcome.checks if not c.passed}


# --- reference solutions -----------------------------------------------------

WC_PY = """\
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Count lines, words and characters.")
    parser.add_argument("path")
    args = parser.parse_args()
    try:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"wc.py: {args.path}: no such file", file=sys.stderr)
        return 1
    print(text.count("\\n"), len(text.split()), len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

WC_BYTES_PY = WC_PY.replace("len(text))", "len(text.encode('utf-8')))")

INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bean There</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>Bean There</h1>
  <section id="menu">
    <ul>
      <li class="drink">Espresso - $2.50</li>
      <li class="drink">Latte - $3.50</li>
      <li class="drink">Mocha - $4.00</li>
    </ul>
  </section>
  <section id="contact">
    <p>12 Bean Street. Open 7am - 6pm daily.</p>
  </section>
</body>
</html>
"""

STYLE_CSS = "body { font-family: sans-serif; }\n"


def _report(words: int = 400, links: int = 2, headings: list[str] | None = None) -> str:
    heads = headings or ["## Summary", "## Comparison", "## Recommendation", "## Sources"]
    body = "Setup concurrency backups " + " ".join(["word"] * max(0, words - 40))
    srcs = "\n".join(f"- [Source {i}](https://example.org/{i})" for i in range(links))
    parts = [
        "# SQLite vs PostgreSQL",
        heads[0],
        "SQLite fits a small desktop app.",
        heads[1],
        body,
        heads[2],
        "Use SQLite.",
        heads[3],
        srcs,
    ]
    return "\n\n".join(parts) + "\n"


# --- the shipped suite -------------------------------------------------------


def test_suite_has_three_benchmarks_one_per_kind(suite: BenchmarkSuite) -> None:
    assert suite.ids() == ["cli-word-count", "research-brief", "static-landing-page"]
    assert {b.kind for b in suite} == set(BenchmarkKind)
    assert len(suite) == 3


def test_shipped_budgets_fit_default_caps(suite: BenchmarkSuite) -> None:
    caps = BudgetLimits()
    for b in suite:
        assert 0 < b.budget_usd <= min(caps.project_usd, caps.daily_usd)
        assert b.answer  # unattended runs need an answer for clarifying questions


def test_shipped_criteria_are_all_deterministic(suite: BenchmarkSuite) -> None:
    for b in suite:
        assert all(c.check is not None for c in b.success_criteria)


def test_results_dir_and_readme_are_ignored_by_loader(suite: BenchmarkSuite) -> None:
    assert (BENCH_DIR / "results").is_dir()
    assert (BENCH_DIR / "README.md").is_file()
    assert "results" not in suite.ids()


def test_cli_reference_solution_passes(suite: BenchmarkSuite, tmp_path: Path) -> None:
    files = _files(tmp_path, {"wc.py": WC_PY, "README.md": "Run: python3 wc.py FILE\n"})
    out = evaluate_benchmark(suite.get("cli-word-count"), files, MemoryEventSink())
    assert out.passed, out.checks
    assert out.score == 1.0
    # check commands clean up after themselves
    assert sorted(p.name for p in files.iterdir()) == ["README.md", "wc.py"]


def test_cli_counting_bytes_fails_unicode_criterion(suite: BenchmarkSuite, tmp_path: Path) -> None:
    files = _files(tmp_path, {"wc.py": WC_BYTES_PY, "README.md": "x\n"})
    out = evaluate_benchmark(suite.get("cli-word-count"), files, MemoryEventSink())
    assert not out.passed
    assert _failed(out) == {3}


def test_cli_wrong_exit_codes_fail(suite: BenchmarkSuite, tmp_path: Path) -> None:
    bad = WC_PY.replace("return 1", "return 0").replace(
        'parser.add_argument("path")', 'parser.add_argument("path", nargs="?", default="")'
    )
    files = _files(tmp_path, {"wc.py": bad, "README.md": "x\n"})
    out = evaluate_benchmark(suite.get("cli-word-count"), files, MemoryEventSink())
    assert _failed(out) == {4, 5}


def test_cli_traceback_on_missing_file_fails(suite: BenchmarkSuite, tmp_path: Path) -> None:
    bad = WC_PY.replace("except FileNotFoundError:", "except PermissionError:")
    files = _files(tmp_path, {"wc.py": bad, "README.md": "x\n"})
    out = evaluate_benchmark(suite.get("cli-word-count"), files, MemoryEventSink())
    assert _failed(out) == {5}


def test_cli_missing_files_fail_everything(suite: BenchmarkSuite, tmp_path: Path) -> None:
    files = _files(tmp_path, {})
    bench = suite.get("cli-word-count")
    out = evaluate_benchmark(bench, files, MemoryEventSink())
    assert not out.passed
    assert out.passed_count == 0
    assert len(out.checks) == len(bench.success_criteria)


def test_web_reference_solution_passes(suite: BenchmarkSuite, tmp_path: Path) -> None:
    files = _files(tmp_path, {"index.html": INDEX_HTML, "style.css": STYLE_CSS})
    out = evaluate_benchmark(suite.get("static-landing-page"), files, MemoryEventSink())
    assert out.passed, out.checks


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        (lambda h, c: (h.replace("<!DOCTYPE html>", ""), c), {3}),
        (lambda h, c: (h.replace("<h1>Bean There</h1>", "<h2>Bean There</h2>"), c), {4}),
        (lambda h, c: (h.replace('id="contact"', 'id="about"'), c), {5}),
        (lambda h, c: (h.replace('<li class="drink">Mocha', "<li>Mocha"), c), {6}),
        (lambda h, c: (h.replace("</body>", "<script>1</script></body>"), c), {7}),
        (lambda h, c: (h, c + "@import url(https://fonts.example/x.css);\n"), {7}),
        (lambda h, c: (h.replace('href="style.css"', 'href="main.css"'), c), {2}),
    ],
)
def test_web_broken_variants_fail_the_right_criterion(
    suite: BenchmarkSuite, tmp_path: Path, change, failed: set[int]
) -> None:
    html, css = change(INDEX_HTML, STYLE_CSS)
    files = _files(tmp_path, {"index.html": html, "style.css": css})
    out = evaluate_benchmark(suite.get("static-landing-page"), files, MemoryEventSink())
    assert _failed(out) == failed


def test_research_reference_solution_passes(suite: BenchmarkSuite, tmp_path: Path) -> None:
    files = _files(tmp_path, {"report.md": _report()})
    out = evaluate_benchmark(suite.get("research-brief"), files, MemoryEventSink())
    assert out.passed, out.checks


@pytest.mark.parametrize(
    ("report", "failed"),
    [
        (_report(words=150), {3}),
        (_report(words=1500), {3}),
        (_report(links=1), {4}),
        (_report(headings=["## Summary", "## Recommendation", "## Comparison", "## Sources"]), {1}),
        (_report().replace("concurrency", "locking"), {2}),
        (_report().replace("https://", "http://"), {4}),
    ],
)
def test_research_broken_variants_fail_the_right_criterion(
    suite: BenchmarkSuite, tmp_path: Path, report: str, failed: set[int]
) -> None:
    files = _files(tmp_path, {"report.md": report})
    out = evaluate_benchmark(suite.get("research-brief"), files, MemoryEventSink())
    assert _failed(out) == failed


def test_links_outside_sources_section_do_not_count(suite: BenchmarkSuite, tmp_path: Path) -> None:
    report = _report(links=0).replace(
        "Use SQLite.", "Use SQLite. [a](https://example.org/a) [b](https://example.org/b)"
    )
    files = _files(tmp_path, {"report.md": report})
    out = evaluate_benchmark(suite.get("research-brief"), files, MemoryEventSink())
    assert _failed(out) == {4}


# --- evaluation goes through the Tool Layer ----------------------------------


def test_evaluation_emits_tool_events_with_project_id(
    suite: BenchmarkSuite, tmp_path: Path
) -> None:
    sink = MemoryEventSink()
    files = _files(tmp_path, {"report.md": _report()})
    evaluate_benchmark(suite.get("research-brief"), files, sink, project_id="bench-run-1")
    kinds = [e.type for e in sink.events]
    assert kinds and all(k == "tool.call" for k in kinds)
    assert {e.project_id for e in sink.events} == {"bench-run-1"}
    assert not any(e.type in {"tool.denied", "tool.approval_required"} for e in sink.events)


def test_evaluation_never_writes_files(suite: BenchmarkSuite, tmp_path: Path) -> None:
    assert "write_file" not in verifier_policy().allowed_tools
    files = _files(tmp_path, {"index.html": INDEX_HTML, "style.css": STYLE_CSS})
    before = {p.name: p.read_bytes() for p in files.iterdir()}
    evaluate_benchmark(suite.get("static-landing-page"), files, MemoryEventSink())
    assert {p.name: p.read_bytes() for p in files.iterdir()} == before


def test_evaluation_does_not_follow_symlinked_outputs(
    suite: BenchmarkSuite, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text(_report(), encoding="utf-8")
    files = _files(tmp_path, {})
    (files / "report.md").symlink_to(outside)
    out = evaluate_benchmark(suite.get("research-brief"), files, MemoryEventSink())
    assert 0 in _failed(out)


def test_evaluation_rejects_symlinked_files_dir(suite: BenchmarkSuite, tmp_path: Path) -> None:
    real = _files(tmp_path, {"report.md": _report()})
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(Exception, match="real directory"):
        evaluate_benchmark(suite.get("research-brief"), link, MemoryEventSink())


def test_check_command_has_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOMA_API_KEY", "sk-secret-value")
    bench = _bench(
        success_criteria=[
            {
                "description": "no key in env",
                "check": {
                    "type": "command",
                    "command": "python3 -c \"import os,sys; sys.exit(1 if any('sk-secret' in v "
                    'for v in os.environ.values()) else 0)"',
                },
            }
        ]
    )
    out = evaluate_benchmark(bench, _files(tmp_path, {}), MemoryEventSink())
    assert out.passed


# --- the model ---------------------------------------------------------------


def _data(**over: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "sample",
        "title": "Sample",
        "kind": "cli",
        "goal": "Do a thing.",
        "budget_usd": 0.5,
        "success_criteria": [
            {"description": "out exists", "check": {"type": "file_exists", "path": "out.txt"}}
        ],
    }
    data.update(over)
    return data


def _bench(**over: object) -> Benchmark:
    return Benchmark.model_validate(_data(**over))


def test_model_defaults() -> None:
    b = _bench()
    assert b.max_attempts == 4
    assert b.answer == ""
    assert b.kind is BenchmarkKind.CLI


@pytest.mark.parametrize(
    "over",
    [
        {"success_criteria": [{"description": "looks good"}]},
        {
            "success_criteria": [
                {"description": "a", "check": {"type": "file_exists", "path": "a"}},
                {"description": "judged by a model"},
            ]
        },
        {"success_criteria": []},
        {"budget_usd": 0},
        {"budget_usd": -1},
        {"budget_usd": MAX_BENCHMARK_BUDGET_USD + 0.01},
        {"budget_usd": math.inf},
        {"budget_usd": math.nan},
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"kind": "game"},
        {"id": "Bad_Id"},
        {"id": "../x"},
        {"id": "-x"},
        {"title": "  "},
        {"goal": ""},
        {"schema_version": 2},
        {"model": "gpt-x"},
        {"api_key": "sk-1"},
        {
            "success_criteria": [
                {"description": "x", "check": {"type": "file_exists", "path": "../etc/passwd"}}
            ]
        },
        {
            "success_criteria": [
                {"description": "x", "check": {"type": "file_exists", "path": "/etc/passwd"}}
            ]
        },
        {
            "success_criteria": [
                {"description": "x", "check": {"type": "command", "command": "a\nb"}}
            ]
        },
    ],
)
def test_model_rejects_invalid(over: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _bench(**over)


def test_model_is_frozen() -> None:
    b = _bench()
    with pytest.raises(ValueError):
        b.budget_usd = 100.0  # type: ignore[misc]


def test_outcome_consistency() -> None:
    ok = CheckResult(criterion=0, passed=True)
    bad = CheckResult(criterion=1, passed=False, detail="x")
    out = BenchmarkOutcome(benchmark_id="sample", passed=False, checks=[ok, bad])
    assert out.passed_count == 1
    assert out.score == 0.5
    with pytest.raises(ValueError):
        BenchmarkOutcome(benchmark_id="sample", passed=True, checks=[ok, bad])
    with pytest.raises(ValueError):
        BenchmarkOutcome(benchmark_id="sample", passed=False, checks=[bad])
    with pytest.raises(ValueError):
        BenchmarkOutcome(benchmark_id="sample", passed=True, checks=[])


# --- the loader --------------------------------------------------------------

GOOD_TOML = """
id = "{id}"
title = "Sample"
kind = "web"
goal = "Make a page."
budget_usd = 0.25

[[success_criteria]]
description = "index exists"
check = {{ type = "file_exists", path = "index.html" }}
"""


def _write(d: Path, bench_id: str, text: str | None = None) -> Path:
    p = d / f"{bench_id}.toml"
    p.write_text(text if text is not None else GOOD_TOML.format(id=bench_id), encoding="utf-8")
    return p


def test_load_dir_ignores_non_toml_entries(tmp_path: Path) -> None:
    _write(tmp_path, "b-two")
    _write(tmp_path, "a-one")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "c.toml").write_text("not even toml [", encoding="utf-8")
    suite = BenchmarkSuite.from_dir(tmp_path)
    assert suite.ids() == ["a-one", "b-two"]
    assert suite.get("a-one").kind is BenchmarkKind.WEB
    with pytest.raises(BenchmarkError, match="unknown benchmark"):
        suite.get("zzz")


def test_load_single_file(tmp_path: Path) -> None:
    assert load_benchmark(_write(tmp_path, "solo")).id == "solo"
    other = tmp_path / "solo.txt"
    other.write_text(GOOD_TOML.format(id="solo"), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="must end in"):
        load_benchmark(other)


def test_empty_suite_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="empty"):
        BenchmarkSuite.from_dir(tmp_path)


def test_missing_or_symlinked_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="does not exist"):
        BenchmarkSuite.from_dir(tmp_path / "nope")
    real = tmp_path / "real"
    real.mkdir()
    _write(real, "x")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(BenchmarkError, match="real directory"):
        BenchmarkSuite.from_dir(tmp_path / "link")


def test_one_bad_file_fails_the_whole_suite(tmp_path: Path) -> None:
    _write(tmp_path, "good")
    _write(tmp_path, "bad", "id = [")
    with pytest.raises(BenchmarkError, match="bad.toml: invalid TOML"):
        BenchmarkSuite.from_dir(tmp_path)


def test_id_must_match_file_name(tmp_path: Path) -> None:
    _write(tmp_path, "one", GOOD_TOML.format(id="two"))
    with pytest.raises(BenchmarkError, match="does not match file name"):
        BenchmarkSuite.from_dir(tmp_path)


def test_invalid_benchmark_reports_fields(tmp_path: Path) -> None:
    text = GOOD_TOML.format(id="x").replace("budget_usd = 0.25", "budget_usd = 50")
    with pytest.raises(BenchmarkError, match="budget_usd"):
        parse_benchmark(text, expected_id="x")


def test_symlinked_file_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.toml.real"
    target.write_text(GOOD_TOML.format(id="linked"), encoding="utf-8")
    d = tmp_path / "suite"
    d.mkdir()
    (d / "linked.toml").symlink_to(target)
    with pytest.raises(BenchmarkError, match="cannot open"):
        BenchmarkSuite.from_dir(d)


def test_directory_named_like_a_benchmark_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "x.toml").mkdir()
    with pytest.raises(BenchmarkError, match="not a regular file"):
        BenchmarkSuite.from_dir(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_is_rejected_without_hanging(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "x.toml")
    with pytest.raises(BenchmarkError, match="not a regular file"):
        BenchmarkSuite.from_dir(tmp_path)


def test_oversized_and_non_utf8_files_are_rejected(tmp_path: Path) -> None:
    big = GOOD_TOML.format(id="big") + "#" * (MAX_BENCHMARK_FILE_BYTES + 1)
    _write(tmp_path, "big", big)
    with pytest.raises(BenchmarkError, match="larger than"):
        BenchmarkSuite.from_dir(tmp_path)
    (tmp_path / "big.toml").unlink()
    (tmp_path / "bin.toml").write_bytes(b'id = "bin"\n\xff\xfe')
    with pytest.raises(BenchmarkError, match="UTF-8"):
        BenchmarkSuite.from_dir(tmp_path)


# --- structure ---------------------------------------------------------------


def test_benchmarks_package_never_touches_models_or_budget() -> None:
    """Benchmark success must not depend on a model; the package must not call one."""
    forbidden = ("automa.llm", "automa.budget", "automa.agents.worker", "automa.agents.executive")
    for py in SRC_PKG.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                assert not m.startswith(forbidden), f"{py.name} imports {m}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"complete", "run"} or py.name != "evaluate.py", (
                    f"{py.name} calls .{node.attr}"
                )
