"""The image benchmark (D104): ``benchmarks/landing-page-with-image.toml``,
``Benchmark.needs_images`` and image generation wired into ``run_benchmark`` /
``automa bench run``.

Local only: the shipped checks run on hand-made PNG/JPEG files; full runs use a
scripted ``FakeProvider`` and ``FakeImageProvider`` behind the real gateways,
BudgetGuard and CostLedger. No network, no real model or image calls.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tomllib
import zlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from automa import bench, cli
from automa.benchmarks import Benchmark, BenchmarkStop, BenchmarkSuite, evaluate_benchmark
from automa.budget import CostLedger
from automa.config import AutomaConfig, ConfigError
from automa.events import MemoryEventSink
from automa.llm import ChatResponse, FakeProvider, FinishReason, Message, ToolCall, Usage
from automa.llm import Role as MsgRole
from automa.llm.images import FakeImageProvider
from automa.plan.models import CommandCheck, shell_construct
from automa.project import ProjectStatus, ProjectStore
from automa.roles import RoleRegistry
from automa.tools import default_tools

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"
BID = "landing-page-with-image"

# Criterion indexes of the shipped benchmark.
EXISTS, PAGE, ONE_IMAGE, VALID_IMAGE, SHOWN, NO_EXTERNAL = range(6)


# --- image fixtures (kept local: tests are not a package, see the M11 CI fix) -------


def png(width: int = 512, height: int = 512) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def jpeg(width: int = 300, height: int = 300, *, sof: int = 0xC0, exif: bool = False) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app1 = b"\xff\xe1" + struct.pack(">H", 2 + 30) + b"Exif\x00\x00" + b"\x00" * 24
    frame = bytes([0xFF, sof]) + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + (app1 if exif else b"") + frame + b"\xff\xd9"


def jpeg_scan_before_frame() -> bytes:
    sos = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
    return b"\xff\xd8" + sos + jpeg()[2:]


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Lumen Kites</title>
<style>body { font-family: sans-serif; } img { max-width: 100%; }</style></head>
<body>
<h1>Lumen Kites</h1>
<img src="images/hero.png" alt="Colourful kites flying over a sunny beach">
<p>Handmade kites for every wind.</p>
</body>
</html>
"""


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return BenchmarkSuite.from_dir(BENCH_DIR).get(BID)


def project_files(tmp: Path, files: dict[str, str | bytes]) -> Path:
    root = tmp / "files"
    root.mkdir()
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    return root


def failed(benchmark: Benchmark, root: Path) -> set[int]:
    out = evaluate_benchmark(benchmark, root, MemoryEventSink())
    assert len(out.checks) == len(benchmark.success_criteria)
    return {c.criterion for c in out.checks if not c.passed}


# --- the shipped benchmark file ---------------------------------------------------


def test_benchmark_is_a_web_benchmark_that_needs_images(benchmark: Benchmark) -> None:
    assert benchmark.kind.value == "web"
    assert benchmark.needs_images is True
    assert len(benchmark.success_criteria) == 6
    assert 0 < benchmark.budget_usd <= 1.0
    assert "images/hero.png" in benchmark.goal and "images/hero.jpg" in benchmark.goal


def test_checks_are_plain_single_line_commands(benchmark: Benchmark) -> None:
    """Same rules as plan checks: one line, no shell syntax (D92), within the size cap."""
    commands = [c.check for c in benchmark.success_criteria if isinstance(c.check, CommandCheck)]
    assert len(commands) == 5
    for check in commands:
        assert shell_construct(check.command) is None, check.command
        assert check.command.startswith('python3 -c "exec(')
        assert "http" not in check.command.replace('\\"http:\\"', "").replace('\\"https:\\"', "")


def test_other_benchmarks_do_not_need_images() -> None:
    suite = BenchmarkSuite.from_dir(BENCH_DIR)
    assert [b.id for b in suite if b.needs_images] == [BID]


def test_needs_images_is_a_strict_bool(benchmark: Benchmark) -> None:
    data = benchmark.model_dump(mode="json")
    data["needs_images"] = "yes"
    with pytest.raises(ValidationError):
        Benchmark.model_validate(data)
    data.pop("needs_images")
    assert Benchmark.model_validate(data).needs_images is False


def test_readme_lists_the_benchmark() -> None:
    text = (BENCH_DIR / "README.md").read_text(encoding="utf-8")
    assert f"`{BID}`" in text and "needs_images" in text


# --- the deterministic checks -------------------------------------------------------


def test_reference_png_solution_passes(benchmark: Benchmark, tmp_path: Path) -> None:
    root = project_files(tmp_path, {"index.html": INDEX_HTML, "images/hero.png": png()})
    assert failed(benchmark, root) == set()


@pytest.mark.parametrize(
    "image",
    [jpeg(), jpeg(sof=0xC2), jpeg(exif=True), jpeg(256, 256)],
    ids=["baseline", "progressive", "exif", "minimum"],
)
def test_reference_jpeg_solution_passes(benchmark: Benchmark, tmp_path: Path, image: bytes) -> None:
    html = INDEX_HTML.replace("images/hero.png", "images/hero.jpg")
    root = project_files(tmp_path, {"index.html": html, "images/hero.jpg": image})
    assert failed(benchmark, root) == set()


@pytest.mark.parametrize(
    "tag",
    [
        "<IMG SRC='images/hero.png' ALT='Kites over a beach'>",
        '<img alt="Kites" src="./images/hero.png" width="512">',
        '<img class="hero" src = "images/hero.png" alt = "Kites" />',
    ],
)
def test_img_tag_variants_pass(benchmark: Benchmark, tmp_path: Path, tag: str) -> None:
    html = INDEX_HTML.replace(
        '<img src="images/hero.png" alt="Colourful kites flying over a sunny beach">', tag
    )
    root = project_files(tmp_path, {"index.html": html, "images/hero.png": png()})
    assert failed(benchmark, root) == set()


def test_nothing_built_fails_everything(benchmark: Benchmark, tmp_path: Path) -> None:
    assert failed(benchmark, project_files(tmp_path, {})) == set(range(6))


def test_page_without_image_fails_the_image_criteria(benchmark: Benchmark, tmp_path: Path) -> None:
    root = project_files(tmp_path, {"index.html": INDEX_HTML})
    assert failed(benchmark, root) == {ONE_IMAGE, VALID_IMAGE, SHOWN}


def test_two_hero_images_are_ambiguous(benchmark: Benchmark, tmp_path: Path) -> None:
    files: dict[str, str | bytes] = {
        "index.html": INDEX_HTML,
        "images/hero.png": png(),
        "images/hero.jpg": jpeg(),
    }
    assert failed(benchmark, project_files(tmp_path, files)) == {ONE_IMAGE, VALID_IMAGE, SHOWN}


def test_image_in_the_wrong_place_does_not_count(benchmark: Benchmark, tmp_path: Path) -> None:
    html = INDEX_HTML.replace("images/hero.png", "hero.png")
    root = project_files(tmp_path, {"index.html": html, "hero.png": png()})
    assert failed(benchmark, root) == {ONE_IMAGE, VALID_IMAGE, SHOWN}


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("images/hero.png", png(255, 512)),
        ("images/hero.png", png(512, 100)),
        ("images/hero.png", jpeg()),  # JPEG bytes behind a .png name
        ("images/hero.jpg", png()),  # PNG bytes behind a .jpg name
        ("images/hero.png", b"not an image at all, just text pretending" * 10),
        ("images/hero.png", b""),
        ("images/hero.png", png()[:20]),  # truncated header
        ("images/hero.jpg", jpeg(255, 300)),
        ("images/hero.jpg", jpeg()[:12]),  # no frame header
        ("images/hero.jpg", jpeg_scan_before_frame()),
        ("images/hero.jpg", b"\xff\xd8" + b"\x00" * 40),
    ],
    ids=[
        "png-narrow",
        "png-short",
        "jpeg-as-png",
        "png-as-jpg",
        "text",
        "empty",
        "png-truncated",
        "jpeg-narrow",
        "jpeg-truncated",
        "jpeg-scan-first",
        "jpeg-garbage",
    ],
)
def test_invalid_images_fail_only_the_validity_check(
    benchmark: Benchmark, tmp_path: Path, name: str, data: bytes
) -> None:
    html = INDEX_HTML.replace("images/hero.png", name)
    root = project_files(tmp_path, {"index.html": html, name: data})
    assert failed(benchmark, root) == {VALID_IMAGE}


def test_symlinked_image_does_not_count(benchmark: Benchmark, tmp_path: Path) -> None:
    root = project_files(tmp_path, {"index.html": INDEX_HTML})
    real = tmp_path / "outside.png"
    real.write_bytes(png())
    (root / "images").mkdir()
    os.symlink(real, root / "images" / "hero.png")
    assert failed(benchmark, root) == {ONE_IMAGE, VALID_IMAGE, SHOWN}


@pytest.mark.parametrize(
    ("old", "new", "want"),
    [
        ("<!DOCTYPE html>\n", "", {PAGE}),
        ('lang="en"', 'lang="de"', {PAGE}),
        ("<h1>Lumen Kites</h1>", "<h2>Lumen Kites</h2>", {PAGE}),
        ('<img src="images/hero.png" ', '<img src="images/other.png" ', {SHOWN}),
        ('alt="Colourful kites flying over a sunny beach"', 'alt=""', {SHOWN}),
        ('alt="Colourful kites flying over a sunny beach"', 'alt="   "', {SHOWN}),
        ('alt="Colourful kites flying over a sunny beach"', "", {SHOWN}),
        (
            '<img src="images/hero.png" alt="Colourful kites flying over a sunny beach">',
            '<div style="background: url(images/hero.png)">Kites</div>',
            {SHOWN},
        ),
        ('src="images/hero.png"', 'src="/images/hero.png"', {SHOWN}),
        ('src="images/hero.png"', 'src="https://cdn.example/hero.png"', {SHOWN, NO_EXTERNAL}),
        ('src="images/hero.png"', 'src="//cdn.example/hero.png"', {SHOWN, NO_EXTERNAL}),
        ('src="images/hero.png"', 'src="data:image/png;base64,AAAA"', {SHOWN, NO_EXTERNAL}),
        ("</body>", "<script>alert(1)</script></body>", {NO_EXTERNAL}),
        ("<p>Handmade", '<a href="javascript:void(0)">x</a><p>Handmade', {NO_EXTERNAL}),
        (
            "img { max-width",
            "@import url(http://fonts.example/x.css); img { max-width",
            {NO_EXTERNAL},
        ),
    ],
)
def test_broken_pages_fail_the_right_criterion(
    benchmark: Benchmark, tmp_path: Path, old: str, new: str, want: set[int]
) -> None:
    assert old in INDEX_HTML
    root = project_files(
        tmp_path, {"index.html": INDEX_HTML.replace(old, new), "images/hero.png": png()}
    )
    assert failed(benchmark, root) == want


def test_non_utf8_page_fails_the_page_checks(benchmark: Benchmark, tmp_path: Path) -> None:
    root = project_files(tmp_path, {"images/hero.png": png()})
    (root / "index.html").write_bytes(INDEX_HTML.encode("utf-8") + b"\xff\xfe")
    assert failed(benchmark, root) == {PAGE, SHOWN, NO_EXTERNAL}


# --- run_benchmark with images -------------------------------------------------------

TEXT = {"family": "anthropic", "capabilities": ["text"], "input_per_mtok": 1, "output_per_mtok": 2}
IMG = {
    "family": "openai",
    "capabilities": ["image_out"],
    "input_per_mtok": 0,
    "output_per_mtok": 0,
    "per_image_usd": 0.04,
}


def config(*, images: bool, tmp: Path) -> AutomaConfig:
    data: dict[str, Any] = {
        "workspace": str(tmp / "ws"),
        "tiers": {"small": "txt", "medium": "txt", "large": "txt"},
        "models": {"txt": TEXT, "img": IMG},
        "limits": {"project_usd": 5.0, "daily_usd": 2.0, "per_call_usd": 0.25},
    }
    if images:
        data["images"] = {"model": "img", "max_per_task": 2}
    return AutomaConfig.model_validate(data)


PLAN: dict[str, Any] = {
    "goal": "kite shop page",
    "tasks": [
        {
            "id": "page",
            "title": "Generate the hero image and write the page",
            "role": "designer",
            "acceptance_criteria": [
                {
                    "description": "hero image exists",
                    "check": {"type": "file_exists", "path": "images/hero.png"},
                },
                {
                    "description": "index.html exists",
                    "check": {"type": "file_exists", "path": "index.html"},
                },
            ],
        }
    ],
}


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=200, output_tokens=40),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def script(html: str = INDEX_HTML) -> list[Any]:
    return [
        json.dumps({"questions": []}),
        json.dumps(PLAN),
        calls(
            (
                "generate_image",
                {"prompt": "colourful kites over a beach", "path": "images/hero.png"},
            ),
            ("write_file", {"path": "index.html", "content": html}),
        ),
        "generated the hero image and wrote index.html",
    ]


class Rig:
    def __init__(self, tmp: Path, *, images: bool = True) -> None:
        self.cfg = config(images=images, tmp=tmp)
        self.store = ProjectStore.for_workspace(self.cfg.workspace)
        self.ledger = CostLedger(tmp / "ledger.sqlite3")
        self.provider = FakeProvider()
        self.image_provider = FakeImageProvider([png()])
        self.benchmark = BenchmarkSuite.from_dir(BENCH_DIR).get(BID)

    def run(self, *, image_provider: Any = "default", **kw: Any) -> Any:
        return bench.run_benchmark(
            kw.pop("benchmark", self.benchmark),
            config=self.cfg,
            provider=self.provider,
            store=self.store,
            ledger=self.ledger,
            roles=RoleRegistry.builtin(tools=default_tools()),
            image_provider=self.image_provider if image_provider == "default" else image_provider,
            **kw,
        )

    def projects(self) -> list[str]:
        root = self.cfg.workspace / "projects"
        return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def test_run_benchmark_generates_the_image_through_the_guard(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.provider.push(*script())
    rec = rig.run()
    assert rec.passed and rec.score == 1.0, rec.checks
    assert rec.stop is BenchmarkStop.COMPLETED
    assert rec.project_status is ProjectStatus.DONE
    (req,) = rig.image_provider.requests
    assert req.model == "img"
    rows = rig.ledger.entries(project_id=rec.project_id)
    assert [r.model for r in rows].count("img") == 1
    assert rec.model_calls == len(rows) == len(rig.provider.requests) + 1
    assert rec.cost_nusd == rig.ledger.spent_nusd(project_id=rec.project_id)
    assert (rig.store.files_dir(rec.project_id) / "images/hero.png").read_bytes() == png()


def test_run_benchmark_judges_the_page_not_the_plan(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.provider.push(
        *script(INDEX_HTML.replace(' alt="Colourful kites flying over a sunny beach"', ""))
    )
    rec = rig.run()
    assert rec.stop is BenchmarkStop.COMPLETED and not rec.passed
    assert [c.criterion for c in rec.checks if not c.passed] == [SHOWN]


def test_image_spending_is_capped_by_the_benchmark_budget(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    cheap = rig.benchmark.model_copy(update={"budget_usd": 0.01})
    rig.provider.push(*script())
    rec = rig.run(benchmark=cheap)
    assert not rec.passed
    assert rig.image_provider.requests == []
    assert rig.ledger.spent_nusd(project_id=rec.project_id) <= 0.01 * 1_000_000_000


def test_needs_images_without_a_provider_refuses_before_anything(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(ConfigError, match="needs image generation"):
        rig.run(image_provider=None)
    assert rig.projects() == [] and rig.ledger.entries() == []
    assert rig.provider.requests == []


def test_provider_with_images_off_is_refused(tmp_path: Path) -> None:
    rig = Rig(tmp_path, images=False)
    plain = rig.benchmark.model_copy(update={"needs_images": False})
    with pytest.raises(ConfigError, match=r"\[images\] model"):
        rig.run(benchmark=plain)
    assert rig.projects() == []


def test_benchmark_without_images_runs_without_a_provider(tmp_path: Path) -> None:
    """Old behaviour kept: no image provider, no image tool, the designer is not offered."""
    rig = Rig(tmp_path, images=False)
    plain = rig.benchmark.model_copy(update={"needs_images": False})
    rig.provider.push(*script())
    rec = rig.run(benchmark=plain, image_provider=None)
    # The plan names the designer, which is unknown to the Executive with images off.
    assert rec.stop is BenchmarkStop.PLANNING
    assert rig.image_provider.requests == []


# --- automa bench run / list -----------------------------------------------------------

TOML = """
workspace = "{ws}"
[tiers]
small = "txt"
medium = "txt"
large = "txt"
[models."txt"]
family = "anthropic"
capabilities = ["text"]
input_per_mtok = 1
output_per_mtok = 2
[models."img"]
family = "openai"
capabilities = ["image_out"]
input_per_mtok = 0
output_per_mtok = 0
per_image_usd = 0.04
[limits]
project_usd = 5.0
daily_usd = 2.0
per_call_usd = 0.25
{images}
"""


class Env:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, *, images: bool) -> None:
        self.ws = tmp / "ws"
        self.config = tmp / "automa.toml"
        section = '[images]\nmodel = "img"\nmax_per_task = 2\n' if images else ""
        self.config.write_text(TOML.format(ws=self.ws.as_posix(), images=section), "utf-8")
        self.suite = tmp / "suite"
        self.suite.mkdir()
        shutil.copy(BENCH_DIR / f"{BID}.toml", self.suite / f"{BID}.toml")
        self.provider = FakeProvider()
        self.image_provider = FakeImageProvider([png()])
        self.provider_calls = 0
        self.image_factory_calls = 0
        monkeypatch.chdir(tmp)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)

        def factory(config: AutomaConfig) -> FakeProvider:
            self.provider_calls += 1
            return self.provider

        def image_factory(config: AutomaConfig) -> FakeImageProvider:
            self.image_factory_calls += 1
            return self.image_provider

        monkeypatch.setattr(cli, "make_provider", factory)
        monkeypatch.setattr(cli, "make_image_provider", image_factory)

    def bench(self) -> Any:
        return CliRunner().invoke(
            cli.app, ["bench", "run", BID, "--suite", str(self.suite), "--config", str(self.config)]
        )

    def projects(self) -> list[str]:
        root = self.ws / "projects"
        return sorted(p.name for p in root.iterdir()) if root.is_dir() else []

    def results(self) -> list[Path]:
        d = self.suite / "results" / BID
        return sorted(d.glob("*.json")) if d.is_dir() else []


def test_cli_bench_run_with_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = Env(tmp_path, monkeypatch, images=True)
    env.provider.push(*script())
    r = env.bench()
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "images: img ($0.0400 per image, at most 2 per task attempt" in r.output
    assert "result: PASSED (score 1.00, 6/6 criteria)" in r.output
    assert env.image_factory_calls == 1 and len(env.image_provider.requests) == 1
    (path,) = env.results()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["passed"] is True and data["model_calls"] == 5  # 4 text calls + 1 image
    with CostLedger.for_workspace(env.ws) as led:
        assert [e.model for e in led.entries()].count("img") == 1


def test_cli_bench_run_refuses_without_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Env(tmp_path, monkeypatch, images=False)
    r = env.bench()
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "needs image generation" in r.output and "[images] model" in r.output
    assert env.provider_calls == 0 and env.image_factory_calls == 0
    assert env.projects() == [] and env.results() == []
    assert not (env.suite / "results").exists()


def test_cli_bench_run_image_provider_error_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Env(tmp_path, monkeypatch, images=True)

    def no_key(config: AutomaConfig) -> FakeImageProvider:
        raise ConfigError("no API key: set AUTOMA_API_KEY")

    monkeypatch.setattr(cli, "make_image_provider", no_key)
    r = env.bench()
    assert r.exit_code == cli.EXIT_CONFIG, r.output
    assert "AUTOMA_API_KEY" in r.output
    assert env.projects() == []


def test_cli_bench_run_other_benchmark_with_images_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain benchmark still runs when images are on (the provider is just available)."""
    env = Env(tmp_path, monkeypatch, images=True)
    shutil.copy(BENCH_DIR / "static-landing-page.toml", env.suite / "static-landing-page.toml")
    r = CliRunner().invoke(
        cli.app,
        [
            "bench",
            "run",
            "static-landing-page",
            "--suite",
            str(env.suite),
            "--config",
            str(env.config),
        ],
    )
    # FakeProvider has no script: the run stops at planning, but only after wiring.
    assert "images: img" in r.output
    assert env.image_provider.requests == []


def test_cli_bench_list_marks_image_benchmarks(tmp_path: Path) -> None:
    r = CliRunner().invoke(cli.app, ["bench", "list", "--suite", str(BENCH_DIR)])
    assert r.exit_code == 0, r.output
    lines = {line.split()[0]: line for line in r.output.splitlines() if line.strip()}
    assert lines[BID].endswith("[needs images]")
    assert "[needs images]" not in lines["static-landing-page"]


def test_example_toml_parses_like_the_suite() -> None:
    raw = tomllib.loads((BENCH_DIR / f"{BID}.toml").read_text(encoding="utf-8"))
    assert raw["needs_images"] is True
    assert Benchmark.model_validate(raw).id == BID
