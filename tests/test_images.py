"""Image generation core (D100): catalog price, ``ImageGateway``, the OpenAI-compatible
image transport and the ``generate_image`` tool.

Local only: ``FakeImageProvider`` or ``httpx.MockTransport`` behind a real
BudgetGuard + CostLedger. No network, no real model calls.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from automa.budget import BudgetError, BudgetExceeded, BudgetGuard, CostLedger
from automa.config import AutomaConfig, ConfigError, ModelInfo, Tier
from automa.events import JsonlEventLog
from automa.llm.gateway import CallContext
from automa.llm.images import (
    MAX_IMAGE_BYTES,
    FakeImageProvider,
    ImageGateway,
    ImageRequest,
    ImageSize,
    OpenAICompatImageProvider,
    image_request_to_wire,
    image_response_from_wire,
)
from automa.llm.provider import ProviderError
from automa.tools import (
    GenerateImage,
    GenerateImageArgs,
    PermissionPolicy,
    RiskLevel,
    ToolError,
    ToolExecutor,
    Workspace,
    default_tools,
    image_info,
)
from automa.tools.images import image_format_for

CTX = CallContext(project_id="p1", task_id="t1")
TEXT = {"family": "anthropic", "capabilities": ["text"], "input_per_mtok": 1, "output_per_mtok": 2}


def png(width: int = 64, height: int = 32) -> bytes:
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


def jpeg(width: int = 48, height: int = 40) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def cfg_data(**image: Any) -> dict[str, Any]:
    return {
        "tiers": {"small": "txt", "medium": "txt"},
        "models": {
            "txt": TEXT,
            "img": {
                "family": "openai",
                "capabilities": ["image_out"],
                "input_per_mtok": 0,
                "output_per_mtok": 0,
                "per_image_usd": 0.04,
                **image,
            },
            "img-unpriced": {
                "family": "openai",
                "capabilities": ["image_out"],
                "input_per_mtok": 0,
                "output_per_mtok": 0,
            },
        },
        "limits": {"project_usd": 1.0, "daily_usd": 1.0, "per_call_usd": 0.25},
    }


def cfg(**image: Any) -> AutomaConfig:
    return AutomaConfig.model_validate(cfg_data(**image))


class Rig:
    def __init__(self, tmp: Path, images: list[bytes | Exception], **kw: Any) -> None:
        self.config = cfg()
        self.ledger = CostLedger(tmp / "ledger.db")
        self.guard = BudgetGuard(self.config.limits, self.ledger)
        self.provider = FakeImageProvider(images, **kw)
        self.gateway = ImageGateway(self.config, self.provider, guard=self.guard)
        root = tmp / "files"
        root.mkdir()
        self.workspace = Workspace(root)
        self.tool = GenerateImage(self.gateway, model="img", context=CTX)

    def run(self, **args: Any) -> str:
        return self.tool.run(GenerateImageArgs(**args), self.workspace)


# --- catalog price ------------------------------------------------------------


def test_per_image_price_is_optional_and_read() -> None:
    c = cfg()
    assert c.models["img"].per_image_usd == 0.04
    assert c.models["img-unpriced"].per_image_usd is None
    assert c.models["txt"].per_image_usd is None


@pytest.mark.parametrize("bad", ["0.04", True, -1, float("inf"), float("nan")])
def test_per_image_price_must_be_a_finite_non_negative_number(bad: Any) -> None:
    with pytest.raises(ValidationError):
        cfg(per_image_usd=bad)


def test_per_image_price_needs_image_out() -> None:
    with pytest.raises(ValidationError, match="image_out"):
        ModelInfo.model_validate({**TEXT, "per_image_usd": 0.01})


def test_image_only_model_cannot_serve_a_tier() -> None:
    data = cfg_data()
    data["tiers"]["large"] = "img"
    with pytest.raises(ValidationError, match="text"):
        AutomaConfig.model_validate(data)


def test_models_cli_shows_per_image_price(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from automa.cli import app

    toml = tmp_path / "automa.toml"
    toml.write_text(
        '[tiers]\nsmall = "txt"\n[models.txt]\nfamily = "anthropic"\ncapabilities = ["text"]\n'
        "input_per_mtok = 1\noutput_per_mtok = 2\n"
        '[models.img]\nfamily = "openai"\ncapabilities = ["image_out"]\n'
        "input_per_mtok = 0\noutput_per_mtok = 0\nper_image_usd = 0.04\n",
        encoding="utf-8",
    )
    r = CliRunner().invoke(app, ["models", "--json", "--config", str(toml)])
    assert r.exit_code == 0, r.output
    data = {e["id"]: e for e in json.loads(r.output)}
    assert data["img"]["per_image_usd"] == 0.04
    assert "per_image_usd" not in data["txt"]
    r = CliRunner().invoke(app, ["models", "--config", str(toml)])
    assert "$0.04 per image" in r.output


# --- ImageGateway ---------------------------------------------------------------


def test_gateway_requires_a_guard() -> None:
    with pytest.raises(TypeError):
        ImageGateway(cfg(), FakeImageProvider(), guard=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="missing"):
        ImageGateway(cfg(), FakeImageProvider(), guard=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model", "match"),
    [
        ("nope", "not in the"),
        ("txt", "image_out"),
        ("img-unpriced", "per_image_usd"),
        ("", "non-empty"),
    ],
)
def test_gateway_refuses_unpriced_or_non_image_models(
    tmp_path: Path, model: str, match: str
) -> None:
    rig = Rig(tmp_path, [png()])
    with pytest.raises(ConfigError, match=match):
        rig.gateway.price_per_image(model)
    if model:
        with pytest.raises(ConfigError):
            rig.gateway.generate(ImageRequest(model=model, prompt="x"), context=CTX)
    assert rig.provider.requests == []
    assert rig.ledger.entries() == []


def test_gateway_records_one_image_at_its_price(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])
    result = rig.gateway.generate(ImageRequest(model="img", prompt="a cat"), context=CTX)
    assert result.cost_usd == pytest.approx(0.04)
    [entry] = rig.ledger.entries()
    assert entry.model == "img"
    assert entry.project_id == "p1" and entry.task_id == "t1"
    assert (entry.input_tokens, entry.output_tokens) == (0, 1)
    assert entry.cost_usd == pytest.approx(0.04)
    assert entry.tier is Tier.MEDIUM
    assert rig.guard.pending_nusd() == 0


def test_gateway_takes_a_higher_reported_cost_never_a_lower_one(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()], reported_cost_usd=0.07)
    r = rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert r.cost_usd == pytest.approx(0.07)
    sub = tmp_path / "low"
    sub.mkdir()
    low = Rig(sub, [png()], reported_cost_usd=0.001)
    r = low.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert r.cost_usd == pytest.approx(0.04)


def test_gateway_releases_on_unbilled_failure(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("no connect", retryable=True)])
    with pytest.raises(ProviderError):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert rig.ledger.entries() == []
    assert rig.guard.pending_nusd() == 0


def test_gateway_records_worst_case_on_maybe_billed_failure(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("timeout", maybe_billed=True)])
    with pytest.raises(ProviderError):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    [entry] = rig.ledger.entries()
    assert entry.cost_usd == pytest.approx(0.04)
    assert rig.guard.pending_nusd() == 0


def test_gateway_releases_on_unexpected_exception(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert rig.ledger.entries() == []
    assert rig.guard.pending_nusd() == 0


def test_gateway_never_retries(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("busy", retryable=True, status=503), png()])
    with pytest.raises(ProviderError):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert len(rig.provider.requests) == 1


def test_gateway_needs_a_project_id(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])
    with pytest.raises(BudgetError, match="project_id"):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CallContext())
    assert rig.provider.requests == []


def test_budget_cap_blocks_before_sending(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()] * 30)
    for _ in range(25):  # 25 x $0.04 = $1.00 = the project cap
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    with pytest.raises(BudgetExceeded):
        rig.gateway.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert len(rig.provider.requests) == 25


def test_per_call_cap_applies_to_the_image_price(tmp_path: Path) -> None:
    config = cfg(per_image_usd=0.5)  # above per_call_usd = 0.25
    ledger = CostLedger(tmp_path / "l.db")
    provider = FakeImageProvider([png()])
    gw = ImageGateway(config, provider, guard=BudgetGuard(config.limits, ledger))
    with pytest.raises(BudgetExceeded):
        gw.generate(ImageRequest(model="img", prompt="x"), context=CTX)
    assert provider.requests == []


def test_image_request_bounds() -> None:
    with pytest.raises(ValidationError):
        ImageRequest(model="img", prompt="")
    with pytest.raises(ValidationError):
        ImageRequest(model="img", prompt="x" * 1001)
    with pytest.raises(ValidationError):
        ImageRequest(model="img", prompt="x", size="9999x9999")  # type: ignore[arg-type]


# --- OpenAI-compatible transport ---------------------------------------------------


def _provider(handler: Any) -> OpenAICompatImageProvider:
    return OpenAICompatImageProvider(
        "https://gw.example/v1", "sk-secret-key-123456", transport=httpx.MockTransport(handler)
    )


def test_wire_body_asks_for_one_base64_image() -> None:
    body = image_request_to_wire(ImageRequest(model="img", prompt="p", size=ImageSize.LANDSCAPE))
    assert body == {
        "model": "img",
        "prompt": "p",
        "n": 1,
        "size": "1536x1024",
        "response_format": "b64_json",
    }


def test_transport_posts_and_decodes() -> None:
    seen: list[httpx.Request] = []
    image = png()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "data": [{"b64_json": base64.b64encode(image).decode()}],
                "usage": {"cost": 0.05},
            },
        )

    resp = _provider(handler).generate(ImageRequest(model="img", prompt="a cat"))
    assert resp.data == image
    assert resp.reported_cost_usd == 0.05
    assert resp.model == "img"
    [req] = seen
    assert req.url == "https://gw.example/v1/images/generations"
    assert req.headers["Authorization"] == "Bearer sk-secret-key-123456"
    assert json.loads(req.content)["prompt"] == "a cat"


def test_transport_needs_a_key() -> None:
    with pytest.raises(ValueError):
        OpenAICompatImageProvider("https://gw.example/v1", "")


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (429, True), (503, True)]
)
def test_transport_http_errors_never_echo_the_body(status: int, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "bad key sk-secret-key-123456"}})

    with pytest.raises(ProviderError) as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert "secret" not in str(ei.value)
    assert ei.value.status == status
    assert ei.value.retryable is retryable
    assert ei.value.maybe_billed is False


def test_transport_connect_error_is_not_billed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ProviderError) as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert not ei.value.maybe_billed


def test_transport_read_timeout_may_be_billed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert ei.value.maybe_billed


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"data": []}).encode(),
        json.dumps({"data": [{"url": "https://x/y.png"}]}).encode(),
        json.dumps({"data": [{"b64_json": "!!!notbase64"}]}).encode(),
        json.dumps({"data": [{"b64_json": "QQ=="}, {"b64_json": "QQ=="}]}).encode(),
    ],
)
def test_transport_malformed_2xx_is_maybe_billed(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(ProviderError) as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert ei.value.maybe_billed


def test_transport_error_object_in_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "sk-secret-key-123456"}})

    with pytest.raises(ProviderError) as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert "secret" not in str(ei.value)


def test_transport_bounds_the_response_size() -> None:
    big = b"A" * (MAX_IMAGE_BYTES * 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"data":[{"b64_json":"' + big + b'"}]}')

    with pytest.raises(ProviderError, match="larger") as ei:
        _provider(handler).generate(ImageRequest(model="img", prompt="p"))
    assert ei.value.maybe_billed


def test_response_parser_ignores_bogus_costs() -> None:
    b64 = base64.b64encode(png()).decode()
    for cost in ("0.1", True, -1, None):
        r = image_response_from_wire(
            {"data": [{"b64_json": b64}], "usage": {"cost": cost}}, fallback_model="m"
        )
        assert r.reported_cost_usd is None
        assert r.model == "m"


def test_transport_ignores_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example:1")
    p = OpenAICompatImageProvider("https://gw.example/v1", "k")
    assert p._client._trust_env is False  # noqa: SLF001
    p.close()


# --- image inspection ---------------------------------------------------------------


def test_image_info_png_and_jpeg() -> None:
    assert image_info(png(64, 32)) == ("png", 64, 32)
    assert image_info(jpeg(48, 40)) == ("jpeg", 48, 40)


@pytest.mark.parametrize(
    "data",
    [
        b"GIF89a....",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 4,
        b"\xff\xd8\xff\xd9",
        b"\xff\xd8" + b"x" * 9,
    ],
)
def test_image_info_rejects_other_bytes(data: bytes) -> None:
    with pytest.raises(ToolError):
        image_info(data)


@pytest.mark.parametrize(
    ("path", "fmt"),
    [("a.png", "png"), ("x/b.JPG", "jpeg"), ("c.jpeg", "jpeg"), ("d.gif", None), (".png", None)],
)
def test_image_format_for(path: str, fmt: str | None) -> None:
    assert image_format_for(path) == fmt


# --- the tool -------------------------------------------------------------------------


def test_tool_is_write_risk_and_unconfigured_in_default_tools(tmp_path: Path) -> None:
    assert GenerateImage.risk is RiskLevel.WRITE
    assert not GenerateImage.irreversible
    found = [t for t in default_tools() if t.name == "generate_image"]
    assert len(found) == 1
    assert isinstance(found[0], GenerateImage) and not found[0].configured
    # The default instance refuses every call: nothing to spend through.
    ws = Workspace(tmp_path)
    args = GenerateImageArgs.model_validate({"prompt": "x", "path": "a.png"})
    with pytest.raises(ToolError, match="not configured"):
        found[0].run(args, ws)
    assert list(tmp_path.iterdir()) == []


def test_only_the_designer_role_lists_generate_image() -> None:
    from automa.roles import RoleRegistry

    roles = RoleRegistry.builtin(tools=default_tools())
    users = [r.name for r in roles if "generate_image" in r.tools]
    assert users == ["designer"]


def test_tool_writes_a_png(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png(64, 32)])
    out = rig.run(prompt="a red square", path="img/hero.png", size="1536x1024")
    assert "wrote img/hero.png: PNG 64x32" in out
    assert "$0.0400" in out
    assert (tmp_path / "files/img/hero.png").read_bytes() == png(64, 32)
    [req] = rig.provider.requests
    assert req.size is ImageSize.LANDSCAPE and req.model == "img"


def test_tool_writes_a_jpeg(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [jpeg()])
    assert "JPEG 48x40" in rig.run(prompt="x", path="a.jpg")
    assert (tmp_path / "files/a.jpg").exists()


def test_tool_refuses_a_format_mismatch_and_writes_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [jpeg()])
    with pytest.raises(ToolError, match="nothing written"):
        rig.run(prompt="x", path="a.png")
    assert not (tmp_path / "files/a.png").exists()
    assert len(rig.ledger.entries()) == 1  # the image was paid for anyway


@pytest.mark.parametrize("data", [b"<svg/>", png(8, 8), png(5000, 20)])
def test_tool_refuses_non_images_and_bad_dimensions(tmp_path: Path, data: bytes) -> None:
    rig = Rig(tmp_path, [data])
    with pytest.raises(ToolError, match="nothing written"):
        rig.run(prompt="x", path="a.png")
    assert list((tmp_path / "files").iterdir()) == []


def test_provider_error_text_never_reaches_the_model(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [ProviderError("echo sk-secret-key-123456", status=400)])
    with pytest.raises(ToolError) as ei:
        rig.run(prompt="x", path="a.png")
    assert str(ei.value) == "image generation failed (HTTP 400)"


def test_budget_refusal_propagates(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()] * 30)
    for i in range(25):
        rig.run(prompt="x", path=f"i{i}.png")
    with pytest.raises(BudgetExceeded):
        rig.run(prompt="x", path="last.png")
    assert not (tmp_path / "files/last.png").exists()


def test_unconfigured_tool_refuses(tmp_path: Path) -> None:
    root = tmp_path / "files"
    root.mkdir()
    with pytest.raises(ToolError, match="not configured"):
        GenerateImage().run(GenerateImageArgs(prompt="x", path="a.png"), Workspace(root))


def test_tool_construction_checks(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    with pytest.raises(ValueError):
        GenerateImage(rig.gateway, model="img")
    with pytest.raises(ValueError):
        GenerateImage(rig.gateway, context=CTX)
    with pytest.raises(ValueError, match="project_id"):
        GenerateImage(rig.gateway, model="img", context=CallContext())
    with pytest.raises(ConfigError):
        GenerateImage(rig.gateway, model="txt", context=CTX)
    with pytest.raises(ConfigError):
        GenerateImage(rig.gateway, model="img-unpriced", context=CTX)


@pytest.mark.parametrize(
    "args",
    [
        {"prompt": "", "path": "a.png"},
        {"prompt": "   ", "path": "a.png"},
        {"prompt": "x‮y", "path": "a.png"},
        {"prompt": "x\x00", "path": "a.png"},
        {"prompt": "line\nbreak", "path": "a.png"},
        {"prompt": "x" * 1001, "path": "a.png"},
        {"prompt": "x", "path": "a.gif"},
        {"prompt": "x", "path": "a.svg"},
        {"prompt": "x", "path": "../a.png"},
        {"prompt": "x", "path": "/abs/a.png"},
        {"prompt": "x", "path": "a.png", "size": "64x64"},
        {"prompt": "x", "path": "a.png", "n": 2},
    ],
)
def test_bad_arguments_are_rejected_before_anything_is_spent(args: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerateImageArgs(**args)


def test_path_is_normalized() -> None:
    assert GenerateImageArgs(prompt="x", path="./img//a.png").path == "img/a.png"


def test_writes_and_audit_report_no_prompt_text(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [])
    args = GenerateImageArgs(prompt="secret plan", path="a.png")
    assert rig.tool.writes(args) == ("a.png",)
    audit = rig.tool.audit(args)
    assert audit == {"path": "a.png", "prompt_chars": "11", "model": "img"}
    assert "secret" not in json.dumps(audit)


def _executor(rig: Rig, tmp_path: Path, policy: PermissionPolicy) -> ToolExecutor:
    return ToolExecutor(
        [rig.tool],
        policy=policy,
        workspace=rig.workspace,
        events=JsonlEventLog(tmp_path / "events.jsonl"),
        project_id="p1",
    )


def test_executor_denies_a_role_that_does_not_list_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])
    ex = _executor(rig, tmp_path, PermissionPolicy(allowed_tools={"write_file"}, max_risk="write"))
    result = ex.execute("generate_image", {"prompt": "x", "path": "a.png"})
    assert not result.ok and result.denied
    assert rig.provider.requests == []


def test_executor_denies_a_read_only_role(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])
    ex = _executor(rig, tmp_path, PermissionPolicy(allowed_tools={"generate_image"}))
    result = ex.execute("generate_image", {"prompt": "x", "path": "a.png"})
    assert not result.ok and result.denied
    assert rig.provider.requests == []


def test_executor_runs_it_for_a_listed_write_role(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [png()])
    ex = _executor(
        rig, tmp_path, PermissionPolicy(allowed_tools={"generate_image"}, max_risk="write")
    )
    result = ex.execute("generate_image", {"prompt": "x", "path": "a.png"})
    assert result.ok, result.error
    assert (tmp_path / "files/a.png").exists()
    log = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert '"prompt_chars"' in log


# --- Workspace.write_bytes ------------------------------------------------------------


def test_write_bytes_is_confined_and_bounded(tmp_path: Path) -> None:
    from automa.tools import WorkspaceError

    ws = Workspace(tmp_path)
    assert ws.write_bytes("d/x.bin", b"\x00\x01") == 2
    assert (tmp_path / "d/x.bin").read_bytes() == b"\x00\x01"
    with pytest.raises(WorkspaceError):
        ws.write_bytes("../x.bin", b"x")
    with pytest.raises(WorkspaceError):
        ws.write_bytes("y.bin", b"xx", max_bytes=1)
    with pytest.raises(WorkspaceError):
        ws.write_bytes("z.bin", "text")  # type: ignore[arg-type]
    (tmp_path / "link.png").symlink_to(tmp_path / "d/x.bin")
    with pytest.raises(WorkspaceError):
        ws.write_bytes("link.png", b"x")
    assert (tmp_path / "d/x.bin").read_bytes() == b"\x00\x01"


def test_write_text_still_rejects_nul(tmp_path: Path) -> None:
    from automa.tools import WorkspaceError

    with pytest.raises(WorkspaceError, match="NUL"):
        Workspace(tmp_path).write_text("a.txt", "x\x00")
