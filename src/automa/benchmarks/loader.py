"""Loading the benchmark suite from a directory of ``<id>.toml`` files.

Only top-level ``*.toml`` entries are benchmarks; everything else in the
directory (``README.md``, ``results/``) is ignored. Loading fails closed, like
roles (D19): a malformed, oversized, symlinked, non-regular or inconsistent
benchmark file is an error for the whole suite, never silently skipped, so a
benchmark run cannot quietly measure fewer goals than intended.
"""

from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from automa.benchmarks.models import Benchmark

BENCHMARK_SUFFIX = ".toml"
MAX_BENCHMARK_FILE_BYTES = 64 * 1024


class BenchmarkError(Exception):
    """A benchmark file or the benchmark suite is invalid."""


def _read_file(path: Path) -> str:
    # O_NONBLOCK: opening a FIFO must not hang; it is then rejected as non-regular.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise BenchmarkError(f"{path.name}: cannot open benchmark file ({e.strerror})") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BenchmarkError(f"{path.name}: not a regular file")
        if st.st_size > MAX_BENCHMARK_FILE_BYTES:
            raise BenchmarkError(f"{path.name}: larger than {MAX_BENCHMARK_FILE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = -1
            data = f.read(MAX_BENCHMARK_FILE_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > MAX_BENCHMARK_FILE_BYTES:
        raise BenchmarkError(f"{path.name}: larger than {MAX_BENCHMARK_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise BenchmarkError(f"{path.name}: not valid UTF-8") from None


def parse_benchmark(text: str, *, expected_id: str, source: str = "<benchmark>") -> Benchmark:
    """Parse one benchmark file's text; ``id`` must equal ``expected_id`` (the file stem)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise BenchmarkError(f"{source}: invalid TOML ({e})") from None
    try:
        bench = Benchmark.model_validate(data)
    except ValidationError as e:
        lines = [
            f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in e.errors()[:20]
        ]
        raise BenchmarkError(f"{source}: invalid benchmark\n" + "\n".join(lines)) from None
    if bench.id != expected_id:
        raise BenchmarkError(f"{source}: id {bench.id!r} does not match file name {expected_id!r}")
    return bench


def load_benchmark(path: Path) -> Benchmark:
    """Load a single benchmark file."""
    path = Path(path)
    if path.suffix != BENCHMARK_SUFFIX:
        raise BenchmarkError(f"{path.name}: benchmark files must end in {BENCHMARK_SUFFIX}")
    return parse_benchmark(_read_file(path), expected_id=path.stem, source=path.name)


class BenchmarkSuite:
    """An ordered (by id), immutable set of benchmarks."""

    def __init__(self, benchmarks: list[Benchmark]) -> None:
        if not benchmarks:
            raise BenchmarkError("benchmark suite is empty")
        ids = [b.id for b in benchmarks]
        if len(set(ids)) != len(ids):
            raise BenchmarkError("duplicate benchmark ids")
        self._by_id = {b.id: b for b in sorted(benchmarks, key=lambda b: b.id)}

    @classmethod
    def from_dir(cls, root: Path) -> BenchmarkSuite:
        root = Path(root)
        try:
            st = os.lstat(root)
        except OSError:
            raise BenchmarkError(f"benchmark directory does not exist: {root}") from None
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise BenchmarkError(f"benchmark directory must be a real directory: {root}")
        paths = sorted(p for p in root.iterdir() if p.name.endswith(BENCHMARK_SUFFIX))
        return cls([load_benchmark(p) for p in paths])

    def get(self, benchmark_id: str) -> Benchmark:
        try:
            return self._by_id[benchmark_id]
        except KeyError:
            raise BenchmarkError(f"unknown benchmark {benchmark_id!r}") from None

    def ids(self) -> list[str]:
        return list(self._by_id)

    def __iter__(self) -> Iterator[Benchmark]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
