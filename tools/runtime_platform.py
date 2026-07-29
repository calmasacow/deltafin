"""Small, dependency-free runtime portability helpers for Deltafin.

This module deliberately imports neither torch nor any model code.  Keeping the
platform decisions here makes them testable on one machine without loading
weights, opening the expert store, or loading an unvalidated native library.
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import os
import platform as host_platform
import re
import sys
from typing import Callable, Iterable


NATIVE_ABI_VERSION = 1
GIB = 1 << 30


class NativeLibraryError(ImportError):
    """A required Deltafin native library is absent or API-incompatible."""


def native_library_filename(stem: str, platform: str | None = None) -> str:
    """Return the native filename for a bare library stem.

    Deltafin currently supports Darwin Mach-O and Linux ELF builds.  Refusing
    unknown platforms is safer than guessing a suffix and later reporting an
    opaque dynamic-loader error.
    """
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return f"{stem}.dylib"
    if platform.startswith("linux"):
        return f"{stem}.so"
    raise NativeLibraryError(
        f"unsupported platform {platform!r}; Deltafin native kernels support "
        "macOS and Linux"
    )


def load_native_library(
    directory: str,
    stem: str,
    *,
    env_var: str,
    required_symbols: Iterable[str],
    abi_version: int = NATIVE_ABI_VERSION,
    platform: str | None = None,
    machine: str | None = None,
    cpuinfo_path: str = "/proc/cpuinfo",
    cdll_factory: Callable[[str], object] | None = None,
) -> tuple[object, str]:
    """Load and validate one Deltafin native library.

    The ABI handshake catches a stale build even when an older library happens
    to export some of the same function names.  The complete symbol manifest
    catches accidentally pointing an override at the other MXFP4 library.
    """
    default_name = native_library_filename(stem, platform)
    override = os.environ.get(env_var)
    path = os.path.abspath(os.path.expanduser(
        override if override else os.path.join(directory, default_name)
    ))
    if not os.path.isfile(path):
        raise NativeLibraryError(
            f"required native library not found: {path}. Rebuild {default_name} "
            f"for this machine, or set {env_var} to a compatible build"
        )

    missing_features = missing_native_cpu_features(
        platform=platform,
        machine=machine,
        cpuinfo_path=cpuinfo_path,
    )
    if missing_features:
        raise NativeLibraryError(
            "this Linux x86-64 CPU is missing native-kernel requirements "
            f"{', '.join(missing_features)}. The current library targets "
            "x86-64-v3; use compatible hardware or rebuild a safe kernel for "
            "this CPU"
        )

    factory = ctypes.CDLL if cdll_factory is None else cdll_factory
    try:
        lib = factory(path)
    except OSError as exc:
        raise NativeLibraryError(
            f"could not load {path}: {exc}. Rebuild it for the current OS and "
            "CPU architecture"
        ) from exc

    symbols = ("mxfp4_abi_version", *tuple(required_symbols))
    missing = [name for name in symbols if not hasattr(lib, name)]
    if missing:
        raise NativeLibraryError(
            f"incompatible or stale native library {path}: missing "
            f"{', '.join(missing)}; rebuild it from the current sources"
        )

    abi = getattr(lib, "mxfp4_abi_version")
    try:
        abi.argtypes = []
        abi.restype = ctypes.c_uint32
        observed = int(abi())
    except Exception as exc:
        raise NativeLibraryError(
            f"could not query the ABI of {path}: {exc}; rebuild it from the "
            "current sources"
        ) from exc
    if observed != abi_version:
        raise NativeLibraryError(
            f"incompatible native library ABI at {path}: found {observed}, "
            f"need {abi_version}; rebuild it from the current sources"
        )
    return lib, path


def import_when_enabled(
    enabled: bool,
    module_name: str,
    importer: Callable[[str], object] = importlib.import_module,
) -> object | None:
    """Import an optional acceleration module only when it is actually enabled."""
    return importer(module_name) if enabled else None


_CUDA_SPEC = re.compile(r"cuda(?::(\d+))?\Z")


def choose_device_spec(
    requested: str | None,
    *,
    mps_available: bool,
    cuda_available: bool,
    cuda_device_count: int = 0,
) -> str:
    """Resolve K3_DEV, validating explicit requests before torch sees them."""
    if requested is None or not requested.strip():
        if mps_available:
            return "mps"
        if cuda_available and cuda_device_count > 0:
            return "cuda"
        return "cpu"

    value = requested.strip().lower()
    if value == "cpu":
        return value
    if value == "mps":
        if not mps_available:
            raise RuntimeError(
                "K3_DEV=mps was requested, but PyTorch reports MPS unavailable"
            )
        return value
    match = _CUDA_SPEC.fullmatch(value)
    if match:
        if not cuda_available or cuda_device_count <= 0:
            raise RuntimeError(
                f"K3_DEV={value} was requested, but PyTorch reports CUDA unavailable"
            )
        index = int(match.group(1) or 0)
        if index >= cuda_device_count:
            raise ValueError(
                f"K3_DEV={value} selects CUDA device {index}, but only "
                f"{cuda_device_count} device(s) are visible"
            )
        return value
    raise ValueError(
        "K3_DEV must be cpu, mps, cuda, or cuda:N "
        f"(got {requested!r})"
    )


def synchronize_device(torch_module, device) -> None:
    """Wait for work on the selected asynchronous device, if any."""
    if device.type == "mps":
        torch_module.mps.synchronize()
    elif device.type == "cuda":
        torch_module.cuda.synchronize(device)


def choose_moe_backend(requested: str | None, device_type: str) -> str:
    """Use Metal only by default on MPS; retain explicit Metal bring-up."""
    value = (
        requested if requested is not None
        else ("metal" if device_type == "mps" else "cpu")
    ).strip().lower()
    if value not in ("cpu", "metal"):
        raise ValueError("K3_MOE must be cpu or metal")
    return value


def darwin_file_hints_enabled(
    requested: bool, platform: str | None = None
) -> bool:
    """Darwin fcntl command numbers must never be sent to another kernel."""
    platform = sys.platform if platform is None else platform
    return bool(requested and platform == "darwin")


# GCC/Clang's -march=x86-64-v3 contract. Linux calls SSE3 "pni" on many
# kernels and exposes LZCNT as either "abm" or "lzcnt", so those two features
# use aliases. AVX is only advertised when the OS has enabled the XSAVE state.
_X86_64_V3_FEATURES = {
    "avx": frozenset(("avx",)),
    "avx2": frozenset(("avx2",)),
    "bmi1": frozenset(("bmi1",)),
    "bmi2": frozenset(("bmi2",)),
    "cx16": frozenset(("cx16",)),
    "f16c": frozenset(("f16c",)),
    "fma": frozenset(("fma",)),
    "lahf_lm": frozenset(("lahf_lm",)),
    "lzcnt": frozenset(("abm", "lzcnt")),
    "movbe": frozenset(("movbe",)),
    "popcnt": frozenset(("popcnt",)),
    "sse3": frozenset(("pni", "sse3")),
    "sse4_1": frozenset(("sse4_1",)),
    "sse4_2": frozenset(("sse4_2",)),
    "ssse3": frozenset(("ssse3",)),
    "xsave": frozenset(("xsave",)),
}


def linux_cpu_flags(cpuinfo_path: str = "/proc/cpuinfo") -> frozenset[str]:
    """Return flags shared by every logical x86 CPU in /proc/cpuinfo."""
    text = _read_text(cpuinfo_path)
    per_cpu = []
    for line in (text or "").splitlines():
        name, sep, values = line.partition(":")
        if sep and name.strip().lower() in ("flags", "features"):
            per_cpu.append(set(values.strip().lower().split()))
    if not per_cpu:
        return frozenset()
    common = per_cpu[0]
    for flags in per_cpu[1:]:
        common.intersection_update(flags)
    return frozenset(common)


def missing_native_cpu_features(
    *,
    platform: str | None = None,
    machine: str | None = None,
    cpuinfo_path: str = "/proc/cpuinfo",
) -> tuple[str, ...]:
    """Return required x86-64-v3 kernel features absent on this Linux host."""
    platform = sys.platform if platform is None else platform
    machine = host_platform.machine() if machine is None else machine
    if (not platform.startswith("linux")
            or machine.strip().lower() not in ("x86_64", "amd64")):
        return ()
    flags = linux_cpu_flags(cpuinfo_path)
    return tuple(sorted(
        name for name, aliases in _X86_64_V3_FEATURES.items()
        if flags.isdisjoint(aliases)
    ))


def _positive_int(text: str | None) -> int | None:
    try:
        value = int(text) if text is not None else 0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _nonnegative_int(text: str | None) -> int | None:
    try:
        value = int(text) if text is not None else -1
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _read_text(path: str) -> str | None:
    _status, value = _read_text_status(path)
    return value


def _read_text_status(path: str) -> tuple[str, str | None]:
    """Return (ok|missing|error, text), preserving fail-closed distinctions."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return "ok", handle.read().strip()
    except FileNotFoundError:
        return "missing", None
    except OSError:
        return "error", None


def _safe_cgroup_path(root: str, group: str) -> str | None:
    root = os.path.abspath(root)
    candidate = os.path.abspath(os.path.join(root, group.lstrip("/")))
    if candidate == root or candidate.startswith(root + os.sep):
        return candidate
    return None


def _cgroup_memory_paths(
    cgroup_text: str | None, cgroup_root: str
) -> list[tuple[str, str]]:
    """Return leaf-to-root candidate (limit,current) cgroup paths."""
    def ancestors(base: str, root: str) -> list[str]:
        base = os.path.abspath(base)
        root = os.path.abspath(root)
        if base != root and not base.startswith(root + os.sep):
            return []
        out = []
        current = base
        while True:
            out.append(current)
            if current == root:
                break
            parent = os.path.dirname(current)
            if parent == current or (
                parent != root and not parent.startswith(root + os.sep)
            ):
                break
            current = parent
        return out

    out: list[tuple[str, str]] = []
    declared = False
    for line in (cgroup_text or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, group = parts
        base = _safe_cgroup_path(cgroup_root, group)
        if base is None:
            continue
        if hierarchy == "0" and controllers == "":
            declared = True
            for candidate in ancestors(base, cgroup_root):
                out.append((
                    os.path.join(candidate, "memory.max"),
                    os.path.join(candidate, "memory.current"),
                ))
        elif "memory" in controllers.split(","):
            declared = True
            v1_root = os.path.join(cgroup_root, "memory")
            v1_base = _safe_cgroup_path(v1_root, group)
            if v1_base is not None:
                for candidate in ancestors(v1_base, v1_root):
                    out.append((
                        os.path.join(candidate, "memory.limit_in_bytes"),
                        os.path.join(candidate, "memory.usage_in_bytes"),
                    ))
    if not declared:
        # Fallback for cgroup namespaces that omit a useful /proc path.
        out.extend([
            (
                os.path.join(cgroup_root, "memory.max"),
                os.path.join(cgroup_root, "memory.current"),
            ),
            (
                os.path.join(cgroup_root, "memory", "memory.limit_in_bytes"),
                os.path.join(cgroup_root, "memory", "memory.usage_in_bytes"),
            ),
        ])
    return list(dict.fromkeys(out))


def _declares_memory_cgroup(cgroup_text: str | None) -> bool:
    for line in (cgroup_text or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, _group = parts
        if (hierarchy == "0" and controllers == "") or (
            "memory" in controllers.split(",")
        ):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class LinuxMemory:
    host_total_bytes: int | None
    host_available_bytes: int | None
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_available_bytes: int | None = None

    @property
    def effective_total_bytes(self) -> int:
        values = []
        if self.host_total_bytes is not None and self.host_total_bytes > 0:
            values.append(self.host_total_bytes)
        if self.cgroup_limit_bytes is not None:
            values.append(max(0, self.cgroup_limit_bytes))
        return min(values) if values else 0

    @property
    def effective_available_bytes(self) -> int:
        values: list[int] = []
        if self.host_available_bytes is not None:
            values.append(max(0, self.host_available_bytes))
        if self.cgroup_available_bytes is not None:
            values.append(max(0, self.cgroup_available_bytes))
        elif self.cgroup_limit_bytes is not None:
            # A finite limit with unreadable usage has unknown headroom. Fail
            # closed instead of treating the entire limit as currently free.
            values.append(
                max(0, self.cgroup_limit_bytes - self.cgroup_current_bytes)
                if self.cgroup_current_bytes is not None else 0
            )
        if not values:
            return 0
        total = self.effective_total_bytes
        if total:
            values.append(total)
        return min(values)


def linux_memory_limits(
    *,
    meminfo_path: str = "/proc/meminfo",
    self_cgroup_path: str = "/proc/self/cgroup",
    cgroup_root: str = "/sys/fs/cgroup",
) -> LinuxMemory:
    """Read host and current-cgroup memory limits without external commands."""
    fields: dict[str, int] = {}
    for line in (_read_text(meminfo_path) or "").splitlines():
        name, sep, rest = line.partition(":")
        if not sep:
            continue
        token = rest.strip().split()
        value = _nonnegative_int(token[0] if token else None)
        if value is not None:
            if name == "MemTotal" and value == 0:
                continue
            fields[name] = value * 1024  # /proc/meminfo is KiB

    candidates: list[tuple[int, int | None]] = []
    cgroup_status, cgroup_text = _read_text_status(self_cgroup_path)
    declared = _declares_memory_cgroup(cgroup_text)
    saw_limit_file = False
    cgroup_read_error = cgroup_status == "error"
    for limit_path, current_path in _cgroup_memory_paths(
        cgroup_text, cgroup_root
    ):
        limit_status, raw_limit = _read_text_status(limit_path)
        if limit_status == "missing":
            continue
        if limit_status == "error":
            cgroup_read_error = True
            continue
        saw_limit_file = True
        current_status, raw_current = _read_text_status(current_path)
        candidate_limit = (
            None if raw_limit == "max"
            else _nonnegative_int(raw_limit)
        )
        if raw_limit in (None, ""):
            cgroup_read_error = True
            continue
        if candidate_limit is None and raw_limit != "max":
            cgroup_read_error = True
            continue
        candidate_current = _nonnegative_int(raw_current)
        if candidate_limit is not None:
            if current_status != "ok":
                candidate_current = None
            candidates.append((candidate_limit, candidate_current))

    # A declared hierarchy whose control state cannot be read is an unknown
    # capacity constraint, not proof that the host's full RAM is available.
    if cgroup_read_error or (declared and not saw_limit_file):
        candidates.append((0, 0))

    host_total = fields.get("MemTotal")
    # Some cgroup-v1 hosts publish a sentinel close to 2**63 rather than
    # "unlimited".  It is not a meaningful cap if it exceeds host RAM.
    candidates = [
        pair for pair in candidates
        if pair[0] < (1 << 60)
        and (host_total is None or pair[0] < host_total * 4)
    ]

    limit = min((pair[0] for pair in candidates), default=None)
    limiting_pair = (
        min(candidates, key=lambda pair: pair[0]) if candidates else None
    )
    current = limiting_pair[1] if limiting_pair is not None else None
    cgroup_available = (
        min(
            max(0, candidate_limit - candidate_current)
            if candidate_current is not None else 0
            for candidate_limit, candidate_current in candidates
        )
        if candidates else None
    )
    return LinuxMemory(
        host_total_bytes=host_total,
        host_available_bytes=fields.get("MemAvailable"),
        cgroup_limit_bytes=limit,
        cgroup_current_bytes=current,
        cgroup_available_bytes=cgroup_available,
    )


def safe_linux_host_budget(memory: LinuxMemory, reserve_bytes: int) -> int:
    """Conservative process budget bounded by both capacity and headroom."""
    reserve_bytes = max(0, int(reserve_bytes))
    total = max(0, memory.effective_total_bytes - reserve_bytes)
    available = max(0, memory.effective_available_bytes - reserve_bytes)
    return min(total, available) if total and available else 0


def cuda_free_memory_budget(
    free_bytes: int,
    total_bytes: int,
    *,
    reserve_bytes: int | None = None,
    utilization: float = 0.85,
) -> int:
    """A conservative allocation ceiling derived from current free VRAM."""
    if not 0 < utilization <= 1:
        raise ValueError("CUDA utilization must be in (0, 1]")
    free_bytes = max(0, int(free_bytes))
    total_bytes = max(0, int(total_bytes))
    if reserve_bytes is None:
        reserve_bytes = max(2 * GIB, int(0.10 * total_bytes))
    return int(max(0, free_bytes - max(0, reserve_bytes)) * utilization)
