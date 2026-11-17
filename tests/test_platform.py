"""Automa refuses to start on non-POSIX platforms with a clear message (D90)."""

from __future__ import annotations

import pytest

from automa import cli


def test_posix_is_supported() -> None:
    assert cli.platform_supported("posix")


@pytest.mark.parametrize("name", ["nt", "java", ""])
def test_non_posix_is_not_supported(name: str) -> None:
    assert not cli.platform_supported(name)


def test_main_refuses_on_windows_with_wsl_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "platform_supported", lambda os_name=None: False)
    called = []
    monkeypatch.setattr(cli, "app", lambda: called.append(True))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == cli.EXIT_USER
    assert not called, "the CLI must not run any command on an unsupported platform"
    err = capsys.readouterr().err
    assert "WSL" in err
    assert "wsl --install" in err


def test_main_runs_app_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "platform_supported", lambda os_name=None: True)
    called = []
    monkeypatch.setattr(cli, "app", lambda: called.append(True))
    cli.main()
    assert called == [True]
