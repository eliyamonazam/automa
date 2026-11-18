"""Image generation wired into runs (D101): ``[images]``, the ``designer`` role, a
per-task ``generate_image`` in the Orchestrator, the cost estimate and ``automa run``.

Local only: ``FakeProvider`` and ``FakeImageProvider`` behind the real gateways,
BudgetGuard and CostLedger. No network, no real model or image calls.
"""

from __future__ import annotations

import ast
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import automa
from automa import cli
from automa.agents import VerifierAgent, WorkerAgent
from automa.budget import NANO_PER_USD, BudgetGuard, CostLedger, cap_to_nusd
from automa.config import AutomaConfig, ConfigError, Tier, load_config
from automa.finance import EstimateSettings, estimate_plan
from automa.llm import (
    ChatResponse,
    FakeProvider,
    FinishReason,
    LLMGateway,
    Message,
    ToolCall,
    Usage,
)
from automa.llm import Role as MsgRole
from automa.llm.gateway import CallContext
from automa.llm.images import FakeImageProvider, ImageGateway
from automa.orchestrator import (
    Orchestrator,
    OrchestratorSettings,
    StopReason,
    usable_roles,
    uses_images,
)
from automa.plan import Plan, TaskStatus
from automa.project import ProjectStatus, ProjectStore
from automa.quality import QualityStats
from automa.roles import RoleRegistry
from automa.tools import (
    GenerateImage,
    GenerateImageArgs,
    RiskLevel,
    ToolError,
    Workspace,
    default_tools,
)
from automa.tools.policy import Decision

PID = "proj-1"
PRICE = 0.04
TEXT = {"family": "anthropic", "capabilities": ["text"], "input_per_mtok": 1, "output_per_mtok": 2}
IMG = {
    "family": "openai",
    "capabilities": ["image_out"],
    "input_per_mtok": 0,
    "output_per_mtok": 0,
    "per_image_usd": PRICE,
}


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


def cfg_data(images: dict[str, Any] | None = None, **limits: float) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tiers": {"small": "txt", "medium": "txt", "large": "txt"},
        "models": {
            "txt": TEXT,
            "img": IMG,
            "img-unpriced": {**IMG, "per_image_usd": None},
            "img-dear": {**IMG, "per_image_usd": 0.30},
        },
        "limits": limits or {"project_usd": 2.0, "daily_usd": 2.0, "per_call_usd": 0.25},
    }
    data["models"]["img-unpriced"].pop("per_image_usd")
    if images is not None:
        data["images"] = images
    return data


def cfg(images: dict[str, Any] | None = None, **limits: float) -> AutomaConfig:
    return AutomaConfig.model_validate(cfg_data(images, **limits))


ON = {"model": "img"}


# --- [images] configuration --------------------------------------------------------


def test_images_off_by_default() -> None:
    c = cfg()
    assert not c.images.enabled
    assert c.images.model is None
    assert c.images.max_per_task == 4
    assert c.image_price() is None


def test_images_on_with_a_priced_image_model() -> None:
    c = cfg({"model": "img", "max_per_task": 2})
    assert c.images.enabled
    assert c.image_price() == PRICE
    assert c.images.max_per_task == 2


@pytest.mark.parametrize(
    ("images", "fragment"),
    [
        ({"model": "nope"}, "not in the [models] catalog"),
        ({"model": "txt"}, "no 'image_out' capability"),
        ({"model": "img-unpriced"}, "no per_image_usd price"),
        ({"model": "img-dear"}, "above [limits] per_call_usd"),
        ({"model": "bad id"}, "invalid model id"),
        ({"model": "img", "max_per_task": 0}, "max_per_task"),
        ({"model": "img", "max_per_task": 21}, "max_per_task"),
        ({"model": "img", "max_per_task": True}, "max_per_task"),
        ({"model": "img", "max_per_task": "2"}, "max_per_task"),
        ({"model": "img", "extra": 1}, "extra"),
    ],
)
def test_images_config_rejected(images: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError) as e:
        cfg(images)
    assert fragment in str(e.value)


def test_image_price_may_equal_the_per_call_cap() -> None:
    c = cfg({"model": "img"}, project_usd=1.0, daily_usd=1.0, per_call_usd=PRICE)
    assert c.image_price() == PRICE


def test_images_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text(
        """
[tiers]
medium = "txt"
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
[images]
model = "img"
max_per_task = 3
""",
        encoding="utf-8",
    )
    c = load_config(path, env_file=None, environ={})
    assert c.images.model == "img" and c.images.max_per_task == 3


def test_bad_images_toml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "automa.toml"
    path.write_text('[images]\nmodel = "missing"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="images"):
        load_config(path, env_file=None, environ={})


def test_example_config_documents_images() -> None:
    root = Path(automa.__file__).resolve().parents[2]
    text = (root / "automa.toml.example").read_text(encoding="utf-8")
    assert "[images]" in text and "max_per_task" in text and "designer" in text


# --- the tool's per-attempt limit ----------------------------------------------------


class ToolRig:
    def __init__(self, tmp: Path, images: list[bytes | Exception], **kw: Any) -> None:
        self.config = cfg(ON)
        self.ledger = CostLedger(tmp / "ledger.db")
        self.guard = BudgetGuard(self.config.limits, self.ledger)
        self.provider = FakeImageProvider(images)
        self.gateway = ImageGateway(self.config, self.provider, guard=self.guard)
        root = tmp / "files"
        root.mkdir()
        self.workspace = Workspace(root)
        self.tool = GenerateImage(
            self.gateway, model="img", context=CallContext(project_id="p", task_id="t"), **kw
        )

    def run(self, path: str) -> str:
        return self.tool.run(GenerateImageArgs(prompt="a cat", path=path), self.workspace)


def test_max_calls_limits_images(tmp_path: Path) -> None:
    rig = ToolRig(tmp_path, [png(), png(), png()], max_calls=2)
    rig.run("a.png")
    rig.run("b.png")
    with pytest.raises(ToolError, match="at most 2 image"):
        rig.run("c.png")
    assert len(rig.provider.requests) == 2
    assert rig.tool.calls == 2
    assert not (tmp_path / "files/c.png").exists()
    assert len(rig.ledger.entries()) == 2


def test_failed_call_counts_against_the_limit(tmp_path: Path) -> None:
    rig = ToolRig(tmp_path, [b"not an image", png()], max_calls=1)
    with pytest.raises(ToolError, match="nothing written"):
        rig.run("a.png")
    with pytest.raises(ToolError, match="at most 1 image"):
        rig.run("a.png")
    assert len(rig.provider.requests) == 1


def test_no_limit_by_default(tmp_path: Path) -> None:
    rig = ToolRig(tmp_path, [png()] * 5)
    for i in range(5):
        rig.run(f"{i}.png")
    assert rig.tool.max_calls is None and rig.tool.calls == 5


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_max_calls_must_be_a_positive_int(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_calls"):
        GenerateImage(max_calls=bad)


def test_unconfigured_tool_refuses_before_counting(tmp_path: Path) -> None:
    tool = GenerateImage(max_calls=1)
    ws = Workspace(tmp_path)
    for _ in range(2):
        with pytest.raises(ToolError, match="not configured"):
            tool.run(GenerateImageArgs(prompt="x", path="a.png"), ws)
    assert tool.calls == 0 and not tool.configured


# --- the designer role ---------------------------------------------------------------


def builtin() -> RoleRegistry:
    return RoleRegistry.builtin(tools=default_tools())


def test_designer_role_is_builtin_and_bounded() -> None:
    d = builtin().get("designer")
    assert uses_images(d)
    assert set(d.tools) == {"read_file", "list_dir", "write_file", "generate_image"}
    assert d.max_risk is RiskLevel.WRITE
    policy = d.policy()
    assert policy.decide("generate_image", RiskLevel.WRITE).decision is Decision.ALLOW
    for name, risk in (
        ("run_command", RiskLevel.EXECUTE),
        ("fetch_url", RiskLevel.EXTERNAL),
        ("delete_file", RiskLevel.WRITE),
    ):
        assert policy.decide(name, risk).decision is not Decision.ALLOW


def test_other_builtin_roles_cannot_call_generate_image() -> None:
    for role in builtin():
        if role.name == "designer":
            continue
        assert not uses_images(role)
        decision = role.policy().decide("generate_image", RiskLevel.WRITE).decision
        assert decision is not Decision.ALLOW


def test_usable_roles_hide_image_roles_when_images_are_off() -> None:
    roles = builtin()
    assert usable_roles(roles, cfg()) == roles.names() - {"designer"}
    assert usable_roles(roles, cfg(ON)) == roles.names()


# --- orchestrator --------------------------------------------------------------------


def calls(*items: tuple[str, dict[str, Any]]) -> ChatResponse:
    tcs = tuple(
        ToolCall(id=f"x{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(items)
    )
    return ChatResponse(
        model="fake",
        message=Message(role=MsgRole.ASSISTANT, content=None, tool_calls=tcs),
        usage=Usage(input_tokens=100, output_tokens=20),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def gen(path: str = "img/hero.png") -> tuple[str, dict[str, Any]]:
    return ("generate_image", {"prompt": "a blue hero banner", "path": path})


def task_dict(tid: str, *, role: str = "designer", path: str = "img/hero.png", **kw: Any) -> dict:
    d: dict[str, Any] = {
        "id": tid,
        "title": f"Make {tid}",
        "role": role,
        "depends_on": [],
        "acceptance_criteria": [
            {"description": f"{path} exists", "check": {"type": "file_exists", "path": path}}
        ],
    }
    d.update(kw)
    return d


def make_plan(*tasks: dict) -> Plan:
    return Plan.model_validate({"goal": "test goal", "tasks": list(tasks)})


class Rig:
    def __init__(
        self,
        tmp: Path,
        script: list[Any],
        images: list[bytes | Exception],
        plan: Plan,
        *,
        image_cfg: dict[str, Any] | None = ON,
        wire: bool = True,
        **limits: float,
    ) -> None:
        self.store = ProjectStore.for_workspace(tmp / "ws")
        self.store.create("test goal", project_id=PID)
        self.store.save_plan(PID, plan)
        self.cfg = cfg(image_cfg, **limits)
        self.ledger = CostLedger(tmp / "ledger.sqlite3")
        events = self.store.events(PID)
        self.guard = BudgetGuard(self.cfg.limits, self.ledger, events=events)
        self.provider = FakeProvider(script)
        self.gateway = LLMGateway(self.cfg, self.provider, guard=self.guard)
        self.image_provider = FakeImageProvider(images)
        self.images = (
            ImageGateway(self.cfg, self.image_provider, guard=self.guard)
            if wire and self.cfg.images.enabled
            else None
        )
        self.worker = WorkerAgent(self.gateway, events=events)
        self.verifier = VerifierAgent(self.gateway, events=events)
        self.roles = builtin()

    def orch(self, **settings: Any) -> Orchestrator:
        return Orchestrator(
            self.store,
            self.worker,
            self.verifier,
            self.roles,
            self.cfg,
            settings=OrchestratorSettings(**settings) if settings else None,
            images=self.images,
        )

    def files(self) -> Path:
        return self.store.files_dir(PID)

    def events(self, type_: str) -> list[dict[str, Any]]:
        return [e.data for e in self.store.events(PID).read() if e.type == type_]


def test_designer_task_generates_an_image_through_the_guard(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(gen()), "done"], [png()], make_plan(task_dict("a")))
    report = rig.orch().run(PID)
    assert report.status is ProjectStatus.DONE
    assert (rig.files() / "img/hero.png").read_bytes() == png()
    (req,) = rig.image_provider.requests
    assert req.model == "img" and req.prompt == "a blue hero banner"
    image_rows = [e for e in rig.ledger.entries() if e.model == "img"]
    assert len(image_rows) == 1
    row = image_rows[0]
    assert (row.project_id, row.task_id, row.tier) == (PID, "a", Tier.MEDIUM)
    assert row.cost_nusd == math.ceil(PRICE * NANO_PER_USD)
    assert row.input_tokens == 0 and row.output_tokens == 1


def test_image_call_uses_the_task_tier(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(gen()), "done"], [png()], make_plan(task_dict("a", tier="large")))
    assert rig.orch().run(PID).status is ProjectStatus.DONE
    (row,) = [e for e in rig.ledger.entries() if e.model == "img"]
    assert row.tier is Tier.LARGE


def test_designer_task_stops_before_any_call_when_images_are_off(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], [], make_plan(task_dict("a")), image_cfg=None)
    report = rig.orch().run(PID)
    assert report.reason is StopReason.CONFIG
    assert "[images] model" in report.detail
    assert rig.provider.requests == [] and rig.image_provider.requests == []
    assert rig.ledger.entries() == []
    assert rig.store.task_states(PID)["a"].status is TaskStatus.PENDING
    assert rig.store.task_history(PID, "a") == []


def test_designer_task_stops_when_images_are_on_but_not_wired(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], [], make_plan(task_dict("a")), wire=False)
    report = rig.orch().run(PID)
    assert report.reason is StopReason.CONFIG
    assert "not wired" in report.detail
    assert rig.provider.requests == [] and rig.ledger.entries() == []


def test_writer_task_runs_without_images_configured(tmp_path: Path) -> None:
    plan = make_plan(task_dict("w", role="writer", path="w.txt"))
    script = [calls(("write_file", {"path": "w.txt", "content": "ok"})), "done"]
    rig = Rig(tmp_path, script, [], plan, image_cfg=None)
    assert rig.orch().run(PID).status is ProjectStatus.DONE


def test_writer_cannot_generate_images_even_when_wired(tmp_path: Path) -> None:
    plan = make_plan(task_dict("w", role="writer", path="w.txt"))
    script = [
        calls(gen("w.png")),
        calls(("write_file", {"path": "w.txt", "content": "ok"})),
        "done",
    ]
    rig = Rig(tmp_path, script, [png()], plan)
    assert rig.orch().run(PID).status is ProjectStatus.DONE
    assert rig.image_provider.requests == []
    assert not (rig.files() / "w.png").exists()
    assert all(e.model != "img" for e in rig.ledger.entries())


def test_image_limit_per_attempt(tmp_path: Path) -> None:
    script = [calls(gen("a.png"), gen("b.png"), gen("img/hero.png")), "done"]
    rig = Rig(
        tmp_path,
        script,
        [png(), png(), png()],
        make_plan(task_dict("a", path="b.png")),
        image_cfg={"model": "img", "max_per_task": 2},
    )
    assert rig.orch().run(PID).status is ProjectStatus.DONE
    assert len(rig.image_provider.requests) == 2
    assert (rig.files() / "b.png").exists()
    assert not (rig.files() / "img/hero.png").exists()


def test_limit_resets_for_a_new_attempt(tmp_path: Path) -> None:
    # Attempt 1 writes the wrong file and fails its check; attempt 2 gets a fresh allowance.
    script = [calls(gen("wrong.png")), "done", calls(gen()), "done"]
    rig = Rig(
        tmp_path,
        script,
        [png(), png()],
        make_plan(task_dict("a")),
        image_cfg={"model": "img", "max_per_task": 1},
    )
    report = rig.orch(max_attempts=2).run(PID)
    assert report.status is ProjectStatus.DONE
    assert len(rig.image_provider.requests) == 2
    contexts = {(e.task_id, e.project_id) for e in rig.ledger.entries() if e.model == "img"}
    assert contexts == {("a", PID)}


def test_budget_cap_stops_image_spending(tmp_path: Path) -> None:
    # Project cap smaller than chat cost + one image: the image is refused by the guard.
    rig = Rig(
        tmp_path,
        [calls(gen()), "done"],
        [png()],
        make_plan(task_dict("a")),
        project_usd=PRICE,
        daily_usd=1.0,
        per_call_usd=PRICE,
    )
    report = rig.orch().run(PID)
    assert report.reason is StopReason.BUDGET
    assert rig.image_provider.requests == []
    assert not (rig.files() / "img/hero.png").exists()
    assert rig.ledger.spent_nusd(project_id=PID) <= cap_to_nusd(PRICE)


def test_parallel_designer_tasks_each_get_their_own_context(tmp_path: Path) -> None:
    plan = make_plan(task_dict("a", path="a.png"), task_dict("b", path="b.png"))

    class PerTask(FakeProvider):
        def complete(self, request: Any) -> ChatResponse:  # type: ignore[override]
            self.requests.append(request)
            text = json.dumps([m.model_dump(mode="json") for m in request.messages])
            tid = "a" if "Make a" in text else "b"
            done = any(m.role is MsgRole.TOOL for m in request.messages)
            if done:
                return ChatResponse(
                    model="fake",
                    message=Message.assistant("done"),
                    usage=Usage(input_tokens=10, output_tokens=2),
                    finish_reason=FinishReason.STOP,
                )
            return calls(gen(f"{tid}.png"))

    rig = Rig(tmp_path, [], [png(), png()], plan)
    rig.provider = PerTask([])
    rig.gateway = LLMGateway(rig.cfg, rig.provider, guard=rig.guard)
    rig.worker = WorkerAgent(rig.gateway, events=rig.store.events(PID))
    rig.verifier = VerifierAgent(rig.gateway, events=rig.store.events(PID))
    report = rig.orch(max_parallel=2).run(PID)
    assert report.status is ProjectStatus.DONE
    rows = sorted(e.task_id or "" for e in rig.ledger.entries() if e.model == "img")
    assert rows == ["a", "b"]


def test_constructor_checks_the_image_gateway(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], [], make_plan(task_dict("a")))
    base: dict[str, Any] = {}

    def build(images: Any, config: AutomaConfig | None = None) -> Orchestrator:
        return Orchestrator(
            rig.store, rig.worker, rig.verifier, rig.roles, config or rig.cfg, images=images, **base
        )

    with pytest.raises(TypeError, match="ImageGateway"):
        build(object())
    other_cfg = cfg({"model": "img", "max_per_task": 3})
    with pytest.raises(ValueError, match="different configuration"):
        build(ImageGateway(other_cfg, FakeImageProvider(), guard=rig.guard))
    # A gateway with its own guard would spend outside the worker's caps.
    other_guard = BudgetGuard(rig.cfg.limits, rig.ledger)
    with pytest.raises(ValueError, match="BudgetGuard"):
        build(ImageGateway(rig.cfg, FakeImageProvider(), guard=other_guard))
    assert build(rig.images) is not None


def test_image_gateway_needs_images_enabled(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [], [], make_plan(task_dict("a")), image_cfg=None)
    gw = ImageGateway(rig.cfg, FakeImageProvider(), guard=rig.guard)
    with pytest.raises(ValueError, match=r"\[images\] model"):
        Orchestrator(rig.store, rig.worker, rig.verifier, rig.roles, rig.cfg, images=gw)


def test_shares_guard_with() -> None:
    c = cfg(ON)
    led = CostLedger(":memory:")
    g = BudgetGuard(c.limits, led)
    llm = LLMGateway(c, FakeProvider([]), guard=g)
    assert ImageGateway(c, FakeImageProvider(), guard=g).shares_guard_with(llm)
    other = BudgetGuard(c.limits, led)
    assert not ImageGateway(c, FakeImageProvider(), guard=other).shares_guard_with(llm)
    assert not ImageGateway(c, FakeImageProvider(), guard=g).shares_guard_with(object())  # type: ignore[arg-type]


# --- cost estimate -------------------------------------------------------------------


def est_task(tid: str, role: str) -> dict[str, Any]:
    return task_dict(tid, role=role, path=f"{tid}.png")


def test_estimate_adds_images_to_the_worst_case() -> None:
    c = cfg({"model": "img", "max_per_task": 3})
    roles = builtin()
    p = make_plan(est_task("d", "designer"), est_task("w", "writer"))
    result = estimate_plan(p, c, roles, [])
    d, w = result.tasks
    per_image = math.ceil(PRICE * NANO_PER_USD)
    cap = cap_to_nusd(c.limits.per_call_usd)
    assert d.max_images == EstimateSettings().max_attempts * 3
    assert d.image_worst_nusd == d.max_images * per_image
    assert d.worst_nusd == d.max_calls * cap + d.image_worst_nusd
    assert w.max_images == 0 and w.image_worst_nusd == 0
    assert w.worst_nusd == w.max_calls * cap
    assert result.worst_nusd == d.worst_nusd + w.worst_nusd


def test_estimate_refuses_image_roles_when_images_are_off() -> None:
    with pytest.raises(ConfigError, match=r"\[images\] model"):
        estimate_plan(make_plan(est_task("d", "designer")), cfg(), builtin(), [])


def test_estimate_without_image_roles_is_unchanged_when_images_are_off() -> None:
    result = estimate_plan(make_plan(est_task("w", "writer")), cfg(), builtin(), [])
    assert result.tasks[0].max_images == 0


def test_expected_cost_includes_the_image_allowance() -> None:
    c = cfg({"model": "img", "max_per_task": 2})
    stats = [
        QualityStats(
            role="designer",
            model="txt",
            tiers=[Tier.MEDIUM],
            attempts=10,
            passed=10,
            cost_nusd=10_000_000,
        )
    ]
    (t,) = estimate_plan(make_plan(est_task("d", "designer")), c, builtin(), stats).tasks
    assert t.expected_attempts == pytest.approx(1.0)
    per_image = math.ceil(PRICE * NANO_PER_USD)
    assert t.expected_nusd == 1_000_000 + 2 * per_image


# --- automa run ----------------------------------------------------------------------

TOML = """
workspace = "{ws}"
[tiers]
small = "txt"
medium = "txt"
large = "txt"
[models."txt"]
family = "anthropic"
capabilities = ["text"]
input_per_mtok = 1.0
output_per_mtok = 2.0
[models."img"]
family = "openai"
capabilities = ["image_out"]
input_per_mtok = 0
output_per_mtok = 0
per_image_usd = 0.04
[limits]
project_usd = 2.0
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
        self.provider = FakeProvider()
        self.image_provider = FakeImageProvider()
        self.image_factory_calls = 0
        monkeypatch.chdir(tmp)
        for var in ("AUTOMA_API_KEY", "AUTOMA_CONFIG", "AUTOMA_WORKSPACE", "AUTOMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(cli, "make_provider", lambda config: self.provider)

        def image_factory(config: AutomaConfig) -> FakeImageProvider:
            self.image_factory_calls += 1
            return self.image_provider

        monkeypatch.setattr(cli, "make_image_provider", image_factory)
        self.store = ProjectStore.for_workspace(self.ws)

    def project(self, *tasks: dict[str, Any]) -> None:
        self.store.create("test goal", project_id=PID)
        self.store.save_plan(PID, make_plan(*tasks))

    def run(self) -> Any:
        return CliRunner().invoke(cli.app, ["run", PID, "--config", str(self.config)])


def test_cli_run_generates_an_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = Env(tmp_path, monkeypatch, images=True)
    env.project(task_dict("a"))
    env.provider.push(calls(gen()), "done")
    env.image_provider = FakeImageProvider([png()])
    r = env.run()
    assert r.exit_code == cli.EXIT_OK, r.output
    assert "images: img ($0.0400 per image, at most 2 per task attempt" in r.output
    assert env.image_factory_calls == 1
    assert (env.store.files_dir(PID) / "img/hero.png").exists()
    with CostLedger.for_workspace(env.ws) as led:
        assert [e.model for e in led.entries(project_id=PID)].count("img") == 1


def test_cli_run_with_images_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = Env(tmp_path, monkeypatch, images=False)
    env.project(task_dict("a"))
    r = env.run()
    assert r.exit_code == cli.RUN_EXIT_CODES[StopReason.CONFIG], r.output
    assert "images: off" in r.output
    assert env.image_factory_calls == 0
    assert env.provider.requests == []


def test_cli_real_image_factory_needs_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automa.llm.factory import image_provider_from_config

    env = Env(tmp_path, monkeypatch, images=True)
    env.project(task_dict("a"))
    monkeypatch.setattr(cli, "make_image_provider", image_provider_from_config)
    r = env.run()
    assert r.exit_code == cli.EXIT_CONFIG
    assert "AUTOMA_API_KEY" in r.output
    assert env.provider.requests == []


def test_executive_is_not_offered_the_designer_when_images_are_off() -> None:
    src = (Path(cli.__file__)).read_text(encoding="utf-8")
    assert "known_roles=roles.names()" not in src
    bench_src = (Path(automa.__file__).resolve().parent / "bench.py").read_text(encoding="utf-8")
    assert "known_roles=roles.names()" not in bench_src


# --- structure -----------------------------------------------------------------------


SRC = Path(automa.__file__).resolve().parent


def test_image_providers_are_not_imported_outside_llm_package() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("llm/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                if {"OpenAICompatImageProvider", "FakeImageProvider"} & names:
                    offenders.append(rel)
    assert offenders == []


def test_only_the_image_gateway_calls_an_image_provider() -> None:
    callers = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_provider"
            ):
                callers.append(rel)
    assert callers == ["llm/images.py"]


def test_orchestrator_binds_a_fresh_image_tool_per_attempt() -> None:
    tree = ast.parse((SRC / "orchestrator.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_with_image_tool"
    )
    built = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "GenerateImage"
    ]
    assert len(built) == 1
    kws = {k.arg for k in built[0].keywords}
    assert {"model", "context", "tier", "max_calls"} <= kws


# --- file_exists on binary files (verifier) ------------------------------------------


def _exists(tmp: Path, path: str) -> tuple[bool, str]:
    from automa.agents.verifier import _check_exists, verifier_policy
    from automa.events import JsonlEventLog
    from automa.plan.models import FileExistsCheck
    from automa.tools import ToolExecutor

    ws = Workspace(tmp / "files")
    executor = ToolExecutor(
        default_tools(), verifier_policy(), ws, JsonlEventLog(tmp / "ev.jsonl"), "p"
    )
    return _check_exists(executor, FileExistsCheck(path=path))


def test_file_exists_accepts_a_binary_file(tmp_path: Path) -> None:
    (tmp_path / "files/img").mkdir(parents=True)
    (tmp_path / "files/img/hero.png").write_bytes(png())
    (tmp_path / "files/top.png").write_bytes(png())
    assert _exists(tmp_path, "img/hero.png")[0]
    assert _exists(tmp_path, "top.png")[0]


def test_file_exists_still_fails_for_missing_or_symlinked_files(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir()
    (tmp_path / "outside.png").write_bytes(png())
    (files / "link.png").symlink_to(tmp_path / "outside.png")
    (files / "hero.png.txt").write_bytes(png())
    assert not _exists(tmp_path, "hero.png")[0]
    assert not _exists(tmp_path, "link.png")[0]
    assert not _exists(tmp_path, "nodir/hero.png")[0]


def test_generated_image_is_listed_among_the_worker_files(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [calls(gen()), "done"], [png()], make_plan(task_dict("a")))
    rig.orch().run(PID)
    (result, _), *_ = rig.store.task_history(PID, "a")
    assert "img/hero.png" in result.files
