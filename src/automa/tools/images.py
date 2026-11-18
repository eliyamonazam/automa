"""``generate_image``: create one PNG/JPEG file in the project workspace (risk ``write``, D100).

The image comes from ``ImageGateway`` (catalogued ``image_out`` model with a
``per_image_usd`` price; every call authorized and recorded by the BudgetGuard).
The tool:

* validates the prompt (length; no control or bidi characters) and the path (a
  safe workspace path ending in ``.png``, ``.jpg`` or ``.jpeg``) before anything
  is spent;
* writes only an image whose bytes really are PNG/JPEG of the format its
  extension names, with sane dimensions, atomically through ``Workspace``;
* never shows a provider's error text to the model (it may echo a key);
* lets a budget refusal propagate, so the run stops exactly as for a chat call.

``default_tools()`` holds an *unconfigured* instance (no gateway: it refuses
every call) so that role files listing it can be checked; the Orchestrator swaps
in a configured instance per task attempt, bound to that project/task's
``CallContext``, the task's tier and ``[images] max_per_task`` (D101). Only a
role that lists it explicitly (the built-in ``designer``) can call it.
"""

from __future__ import annotations

import struct
import unicodedata
from typing import ClassVar

from pydantic import Field, field_validator

from automa.config import Tier
from automa.llm.gateway import CallContext
from automa.llm.images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PROMPT_CHARS,
    ImageGateway,
    ImageRequest,
    ImageSize,
)
from automa.llm.provider import ProviderError
from automa.plan.models import MAX_PATH_CHARS
from automa.tools.base import RiskLevel, Tool, ToolArgs, ToolError
from automa.tools.workspace import Workspace, WorkspaceError, normalize

IMAGE_EXTENSIONS: dict[str, str] = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
MIN_IMAGE_SIDE = 16
MAX_IMAGE_SIDE = 4096

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers that carry the frame size (not DHT/JPG/DAC).
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _bad_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat in ("Cc", "Cs", "Co", "Cn") or cat == "Cf"


class GenerateImageArgs(ToolArgs):
    prompt: str = Field(
        min_length=1, max_length=MAX_IMAGE_PROMPT_CHARS, description="What the image shows"
    )
    path: str = Field(
        min_length=1,
        max_length=MAX_PATH_CHARS,
        description="Relative output path ending in .png, .jpg or .jpeg",
    )
    size: ImageSize = Field(default=ImageSize.SQUARE, description="Image size in pixels")

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str) -> str:
        if any(_bad_char(ch) for ch in v):
            raise ValueError("prompt must not contain control, format or bidi characters")
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        try:
            norm = normalize(v)
        except WorkspaceError as e:
            raise ValueError(str(e)) from None
        if image_format_for(norm) is None:
            raise ValueError("path must end in .png, .jpg or .jpeg")
        return norm


def image_format_for(path: str) -> str | None:
    """``png``/``jpeg`` from the path's extension (case-insensitive), else ``None``."""
    name = path.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    if dot <= 0:
        return None
    return IMAGE_EXTENSIONS.get(name[dot:].lower())


def image_info(data: bytes) -> tuple[str, int, int]:
    """``(format, width, height)`` of PNG/JPEG bytes; ``ToolError`` for anything else."""
    if data.startswith(_PNG_MAGIC):
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise ToolError("the generated image is not a valid PNG")
        width, height = struct.unpack(">II", data[16:24])
        return "png", width, height
    if data.startswith(b"\xff\xd8"):
        return ("jpeg", *_jpeg_size(data))
    raise ToolError("the generated file is not a PNG or JPEG image")


def _jpeg_size(data: bytes) -> tuple[int, int]:
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no length field
            i += 2
            continue
        if marker == 0xD9:  # end of image before a frame
            break
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        if length < 2:
            break
        if marker in _JPEG_SOF:
            if i + 9 > n:
                break
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + length
    raise ToolError("the generated image is not a valid JPEG")


class GenerateImage(Tool):
    name: ClassVar[str] = "generate_image"
    description: ClassVar[str] = (
        "Generate one image from a text prompt and save it as a PNG or JPEG file in the "
        "project workspace. Each call costs money (counted against the project budget)."
    )
    risk: ClassVar[RiskLevel] = RiskLevel.WRITE
    args_model: ClassVar[type[ToolArgs]] = GenerateImageArgs

    def __init__(
        self,
        gateway: ImageGateway | None = None,
        *,
        model: str | None = None,
        context: CallContext | None = None,
        tier: Tier = Tier.MEDIUM,
        max_calls: int | None = None,
    ) -> None:
        if max_calls is not None and (
            isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1
        ):
            raise ValueError("max_calls must be a positive integer")
        if gateway is not None:
            if model is None or context is None:
                raise ValueError("generate_image with a gateway needs a model and a context")
            if not context.project_id:
                raise ValueError("generate_image needs a context with a project_id")
            gateway.price_per_image(model)  # ConfigError now, not at the first call
        self._gateway = gateway
        self._model = model
        self._context = context
        self._tier = tier
        self._max_calls = max_calls
        self._calls = 0

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def configured(self) -> bool:
        return self._gateway is not None

    @property
    def context(self) -> CallContext | None:
        return self._context

    @property
    def tier(self) -> Tier:
        return self._tier

    @property
    def max_calls(self) -> int | None:
        return self._max_calls

    @property
    def calls(self) -> int:
        """Calls that reached the gateway (paid or refused) on this instance."""
        return self._calls

    def run(self, args: ToolArgs, workspace: Workspace) -> str:
        if not isinstance(args, GenerateImageArgs):
            raise TypeError("arguments were not validated by the executor")
        if self._gateway is None or self._model is None or self._context is None:
            raise ToolError("image generation is not configured")
        if self._max_calls is not None and self._calls >= self._max_calls:
            raise ToolError(
                f"image limit reached: at most {self._max_calls} image(s) per task attempt"
            )
        expected = image_format_for(args.path)
        request = ImageRequest(model=self._model, prompt=args.prompt, size=args.size)
        # Counted before the call: a failed or refused call still uses up the limit.
        self._calls += 1
        try:
            result = self._gateway.generate(request, context=self._context, tier=self._tier)
        except ProviderError as e:
            status = f" (HTTP {e.status})" if e.status is not None else ""
            raise ToolError(f"image generation failed{status}") from None
        spent = f"cost ${result.cost_usd:.4f}"
        if len(result.data) > MAX_IMAGE_BYTES:
            raise ToolError(f"the generated image is too large; nothing written ({spent})")
        try:
            fmt, width, height = image_info(result.data)
        except ToolError as e:
            raise ToolError(f"{e}; nothing written ({spent})") from None
        if fmt != expected:
            raise ToolError(
                f"the generated image is {fmt.upper()}, not {expected.upper() if expected else '?'}"
                f" as the path says; nothing written ({spent})"
            )
        if not (
            MIN_IMAGE_SIDE <= width <= MAX_IMAGE_SIDE and MIN_IMAGE_SIDE <= height <= MAX_IMAGE_SIDE
        ):
            raise ToolError(
                f"the generated image has unusable dimensions {width}x{height}; "
                f"nothing written ({spent})"
            )
        n = workspace.write_bytes(args.path, result.data, max_bytes=MAX_IMAGE_BYTES)
        return f"wrote {args.path}: {fmt.upper()} {width}x{height}, {n} bytes ({spent})"

    def writes(self, args: ToolArgs) -> tuple[str, ...]:
        path = getattr(args, "path", None)
        return (path,) if isinstance(path, str) else ()

    def audit(self, args: ToolArgs) -> dict[str, str]:
        out: dict[str, str] = {}
        path = getattr(args, "path", None)
        if isinstance(path, str):
            out["path"] = path
        prompt = getattr(args, "prompt", None)
        if isinstance(prompt, str):
            out["prompt_chars"] = str(len(prompt))
        if self._model is not None:
            out["model"] = self._model
        return out
