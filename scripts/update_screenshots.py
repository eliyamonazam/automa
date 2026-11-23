"""Regenerate the README pictures of the web UI's version bars (D178, D179).

Runs the real web UI on an empty temporary workspace with a fixed, made-up list of
"what's new" titles (sample text, labelled as such in the README) and takes two
screenshots with Playwright: the "new version is ready" bar and the one-time
"updated to the new version" bar. No model, no network, no real git.

    python scripts/update_screenshots.py            # writes docs/images/*.png
    pip install playwright && playwright install chromium   # if Playwright is missing
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

from automa.updatecheck import UpdateNotice
from automa.webui import server as ws

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
SAMPLE = UpdateNotice(
    commits=6,
    titles=(
        "نوار «نسخهٔ تازه آماده‌ست» در رابط وب",
        "بازطراحی ظاهر، بخش ۳ب",
        "پیش‌بینی هزینه از هزینهٔ واقعی",
    ),
)


class _Fixed:
    """A watcher that always says the same thing (screenshots only)."""

    def __init__(self, ready: UpdateNotice | None, done: UpdateNotice | None) -> None:
        self._ready, self._done = ready, done

    def notice(self) -> UpdateNotice | None:
        return self._ready

    def welcome(self) -> UpdateNotice | None:
        return self._done

    def welcome_shown(self) -> None:
        return None


def _shoot(watcher: _Fixed, out: Path, *, open_list: bool, width: int) -> None:
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        server = ws.create_server(
            root,
            root / "ledger.sqlite3",
            port=0,
            language="fa",
            updates=watcher,  # type: ignore[arg-type]
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                pg = browser.new_page(
                    viewport={"width": width, "height": 900}, device_scale_factor=2
                )
                pg.goto(ws.server_url(server))
                if open_list:
                    pg.click(".updatebar summary")
                pg.evaluate("document.fonts.ready")
                box = pg.locator(".updatebar").last.bounding_box()
                if box is None:
                    raise RuntimeError("the version bar is not on the page")
                pg.screenshot(
                    path=str(out),
                    clip={"x": 0, "y": 0, "width": width, "height": box["y"] + box["height"] + 16},
                )
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _shoot(_Fixed(SAMPLE, None), OUT / "update-ready.png", open_list=True, width=820)
    _shoot(_Fixed(None, SAMPLE), OUT / "update-done.png", open_list=False, width=820)
    print(f"wrote {OUT / 'update-ready.png'} and {OUT / 'update-done.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
