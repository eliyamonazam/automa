"""Regenerate the pictures of the family guide (M15, D190).

Runs the real web UI (Persian, light theme, phone width) on a temporary workspace
with one made-up example project in each state the guide talks about, signs in
through the real login form and takes one picture per guide section with
Playwright. The example (a page for a tailoring shop) is sample text and every
picture is captioned as a sample in the guide. No model, no network, no real run:
the "planner" and the "start" button are stand-ins that are never called.

    python scripts/guide_screenshots.py      # writes src/automa/webui/static/guide/*.png

The names and captions come from :data:`automa.webui.guide.IMAGES`; a test checks
that every picture exists, is a small PNG and is served by the web UI.
"""

from __future__ import annotations

import io
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from automa.config import Tier
from automa.plan import Plan, TaskState, TaskStatus
from automa.plan.models import (
    CheckResult,
    NeedFromUser,
    ResultStatus,
    TaskResult,
    Verdict,
    VerdictOutcome,
)
from automa.project import ProjectStatus, ProjectStore
from automa.users import UserRole, UserStore
from automa.webui import server as ws
from automa.webui.guide import EXAMPLES, IMAGES
from automa.webui.ideas import clarify_path, project_path
from automa.webui.static import GUIDE_IMAGE_SCALE as SCALE

OUT = Path(__file__).resolve().parents[1] / "src" / "automa" / "webui" / "static" / "guide"
WIDTH, HEIGHT = 390, 844  # a common phone screen, in CSS pixels
NAME, PASSWORD = "sara", "sample-password-123"  # a throwaway account in a temp dir
NOW = datetime(2026, 11, 26, 9, 0, tzinfo=UTC)
GOAL = EXAMPLES[2]["fa"].split("«", 1)[1].rstrip("»").strip()
RANGE = "حدود ۲۰ تا ۸۰ هزار تومان"


class _SamplePlanner:
    """Stands in for the Executive: shows a fixed cost range, never plans."""

    def prepare(self) -> None:
        return None

    def clarify(self, project_id: str) -> Sequence[str]:
        raise RuntimeError("not used for pictures")

    def plan(self, project_id: str, clarifications: Sequence[object]) -> None:
        raise RuntimeError("not used for pictures")

    def cost_range(self, project_id: str) -> str:
        return RANGE


class _SampleRuns:
    """Stands in for ``automa run``: the button shows, nothing is ever started."""

    def needs_confirm(self, project_id: str) -> bool:
        return False

    def command(self, project_id: str, *, yes: bool) -> list[str]:
        raise RuntimeError("not used for pictures")


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "goal": GOAL,
            "tasks": [
                {
                    "id": "page",
                    "title": "ساختن صفحهٔ وب با اسم، عکس‌ها، ساعت کاری و دکمهٔ تماس",
                    "role": "developer",
                    "acceptance_criteria": [
                        {
                            "description": "فایل index.html ساخته شده",
                            "check": {"type": "file_exists", "path": "index.html"},
                        },
                        {"description": "روی گوشی درست و خوانا دیده می‌شه"},
                    ],
                },
                {
                    "id": "check",
                    "title": "بررسی صفحه روی گوشی و کامپیوتر",
                    "role": "reviewer",
                    "depends_on": ["page"],
                    "acceptance_criteria": [{"description": "همهٔ پیوندها و دکمه‌ها کار می‌کنن"}],
                },
            ],
            "deliverable": {
                "summary": "یک صفحهٔ وب یک‌فایلی برای معرفی کارگاه خیاطی.",
                "main_files": ["index.html"],
                "how_to_use": "روی «باز کردن» بزن تا صفحه رو ببینی؛ با «دانلود» فایل رو بگیر.",
            },
        }
    )


def _workspace(root: Path) -> dict[str, str]:
    """The sample projects: one with questions, one planned, one delivered."""
    UserStore(root).add("owner", PASSWORD + "-owner", UserRole.OWNER)
    UserStore(root).add(NAME, PASSWORD, UserRole.MEMBER)
    store = ProjectStore.for_workspace(root)
    ids: dict[str, str] = {}

    ids["questions"] = store.create(GOAL, owner=NAME).id
    store.save_questions(
        ids["questions"],
        ["چه رنگ‌هایی برای صفحه دوست داری؟", "ساعت کاری کارگاه چیه؟"],
        [
            NeedFromUser(
                what="چند عکس از کارگاه و کارهات",
                why="برای اینکه صفحه عکس‌های واقعی خودت رو نشون بده",
                required=False,
                where="از گالری گوشی انتخاب کن و همین‌جا بفرست",
            )
        ],
    )

    ids["start"] = store.create(GOAL, owner=NAME).id
    store.save_plan(ids["start"], _plan())

    ids["result"] = pid = store.create(GOAL, owner=NAME).id
    store.save_plan(pid, _plan())
    files = store.files_dir(pid)
    files.mkdir(parents=True, exist_ok=True)
    (files / "index.html").write_text("<!doctype html><title>نمونه</title>", encoding="utf-8")
    for tid in ("page", "check"):
        store.save_task_result(
            pid,
            TaskResult(
                task_id=tid,
                attempt=1,
                status=ResultStatus.COMPLETED,
                tier=Tier.MEDIUM,
                model="sample",
                started_at=NOW,
                finished_at=NOW,
            ),
        )
        criteria = len(_plan().task(tid).acceptance_criteria)
        store.save_verdict(
            pid,
            Verdict(
                task_id=tid,
                attempt=1,
                outcome=VerdictOutcome.PASS,
                checks=[CheckResult(criterion=i, passed=True) for i in range(criteria)],
                created_at=NOW,
            ),
        )
        store.save_task_state(
            pid,
            TaskState(task_id=tid, status=TaskStatus.PASSED, attempts=1, updated_at=NOW),
        )
    store.set_status(pid, ProjectStatus.RUNNING)
    store.set_status(pid, ProjectStatus.DONE)
    return ids


def _shrink(png: bytes) -> bytes:
    """Fewer colours (the pages are flat), so each picture stays small in the repo."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is optional
        return png
    img = Image.open(io.BytesIO(png)).convert("RGB")
    out = io.BytesIO()
    img.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(out, "PNG", optimize=True)
    return out.getvalue() if len(out.getvalue()) < len(png) else png


def main() -> int:
    from playwright.sync_api import Page, sync_playwright

    def fill_idea(pg: Page) -> None:
        pg.fill("textarea", GOAL)

    def to(
        selector: str, *, has_text: str | None = None, above: int = 64
    ) -> Callable[[Page], None]:
        """Scroll so the first match is ``above`` pixels below the top of the screen."""

        def scroll(pg: Page) -> None:
            el = pg.locator(selector, has_text=has_text).first
            top = el.evaluate("e => e.getBoundingClientRect().top + scrollY")
            pg.evaluate(f"window.scrollTo(0, {max(0, int(top) - above)})")

        return scroll

    def snap(pg: Page, until: str = "main") -> bytes:
        """The screen, cut just below ``until`` (no empty phone screen)."""
        bottom = pg.locator(until).first.evaluate("e => e.getBoundingClientRect().bottom")
        height = max(200, min(HEIGHT, int(bottom) + 16))
        return _shrink(pg.screenshot(clip={"x": 0, "y": 0, "width": WIDTH, "height": height}))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ids = _workspace(root)
        shots: dict[str, tuple[str, Callable[[Page], None] | None]] = {
            "login": ("/login", None),
            "idea": ("/new", fill_idea),
            "questions": (clarify_path(ids["questions"]), None),
            "start": (project_path(ids["start"]), to("p", has_text=RANGE, above=220)),
            "result": (project_path(ids["result"]), to("section.deliver")),
        }
        missing = set(IMAGES) ^ set(shots)
        if missing:
            raise RuntimeError(f"pictures and guide disagree: {sorted(missing)}")
        server = ws.create_server(
            root,
            root / "ledger.sqlite3",
            port=0,
            language="fa",
            allow_actions=True,
            planner=_SamplePlanner(),  # type: ignore[arg-type]
            runs=_SampleRuns(),
            toman_per_usd=1_000_000.0,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = ws.server_url(server).rstrip("/")
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                ctx = browser.new_context(
                    viewport={"width": WIDTH, "height": HEIGHT},
                    device_scale_factor=SCALE,
                    color_scheme="light",
                )
                pg = ctx.new_page()
                pg.goto(base + "/login")
                pg.evaluate("document.fonts.ready")
                (OUT / "login.png").write_bytes(snap(pg))
                pg.fill("input[name=name]", NAME)
                pg.fill("input[name=password]", PASSWORD)
                pg.click("button[type=submit]")
                pg.wait_for_load_state()
                for name, (path, prepare) in shots.items():
                    if name == "login":
                        continue
                    pg.goto(base + path)
                    pg.evaluate("document.fonts.ready")
                    if prepare is not None:
                        prepare(pg)
                    until = "section.deliver" if name == "result" else "main"
                    (OUT / f"{name}.png").write_bytes(snap(pg, until))
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    for name in IMAGES:
        print(f"{OUT / (name + '.png')}: {(OUT / (name + '.png')).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
