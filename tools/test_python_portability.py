#!/usr/bin/env python3
"""Weight-free portability gates for native loading, I/O hints, and devices."""

import ast
import importlib.util
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime_platform as rp  # noqa: E402
import spine_cache  # noqa: E402
import spine_io  # noqa: E402

V3_FLAGS = {
    next(iter(aliases))
    for aliases in rp._X86_64_V3_FEATURES.values()
}


class FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.result


class FakeLibrary:
    def __init__(self, symbols=(), abi=rp.NATIVE_ABI_VERSION):
        self.mxfp4_abi_version = FakeFunction(abi)
        for name in symbols:
            setattr(self, name, FakeFunction())


def _native_loader_checks():
    assert rp.native_library_filename("libx", "darwin") == "libx.dylib"
    assert rp.native_library_filename("libx", "linux") == "libx.so"
    assert rp.native_library_filename("libx", "linux-musl") == "libx.so"
    try:
        rp.native_library_filename("libx", "win32")
    except rp.NativeLibraryError:
        pass
    else:
        raise AssertionError("unknown native platform was accepted")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "libdemo.so")
        pathlib.Path(path).touch()
        with mock.patch.dict(os.environ, {"TEST_NATIVE_LIB": path}):
            lib, got = rp.load_native_library(
                td,
                "libdemo",
                env_var="TEST_NATIVE_LIB",
                required_symbols=("run",),
                platform="linux",
                machine="aarch64",
                cdll_factory=lambda _path: FakeLibrary(("run",)),
            )
            assert got == path and hasattr(lib, "run")

            for fake, expected in (
                (FakeLibrary(()), "missing"),
                (FakeLibrary(("run",), abi=99), "found 99"),
            ):
                try:
                    rp.load_native_library(
                        td,
                        "libdemo",
                        env_var="TEST_NATIVE_LIB",
                        required_symbols=("run",),
                        platform="linux",
                        machine="aarch64",
                        cdll_factory=lambda _path, fake=fake: fake,
                    )
                except rp.NativeLibraryError as exc:
                    assert expected in str(exc)
                else:
                    raise AssertionError("incompatible native library was accepted")

        cpuinfo = os.path.join(td, "cpuinfo")
        pathlib.Path(cpuinfo).write_text(
            "processor: 0\nflags: " + " ".join(sorted(V3_FLAGS)) + "\n"
            "processor: 1\nflags: " + " ".join(sorted(V3_FLAGS | {"sha_ni"})) + "\n",
            encoding="utf-8",
        )
        assert rp.missing_native_cpu_features(
            platform="linux", machine="x86_64", cpuinfo_path=cpuinfo
        ) == ()
        pathlib.Path(cpuinfo).write_text(
            "processor: 0\nflags: "
            + " ".join(sorted(V3_FLAGS - {"fma"})) + "\n",
            encoding="utf-8",
        )
        assert rp.missing_native_cpu_features(
            platform="linux", machine="x86_64", cpuinfo_path=cpuinfo
        ) == ("fma",)
        with mock.patch.dict(os.environ, {"TEST_NATIVE_LIB": path}):
            try:
                rp.load_native_library(
                    td,
                    "libdemo",
                    env_var="TEST_NATIVE_LIB",
                    required_symbols=("run",),
                    platform="linux",
                    machine="x86_64",
                    cpuinfo_path=cpuinfo,
                    cdll_factory=lambda _path: FakeLibrary(("run",)),
                )
            except rp.NativeLibraryError as exc:
                assert "x86-64-v3" in str(exc) and "fma" in str(exc)
            else:
                raise AssertionError("unsafe x86 native library was loaded")


def _load_module_with_fake_native(filename):
    calls = []

    def fake_loader(directory, stem, **kwargs):
        calls.append((directory, stem, kwargs))
        return FakeLibrary(kwargs["required_symbols"]), f"/fake/{stem}"

    name = f"_portability_{filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(rp, "load_native_library", fake_loader):
        spec.loader.exec_module(module)
    return module, calls


def _native_module_manifest_checks():
    fast, calls = _load_module_with_fake_native("fast_moe.py")
    assert len(calls) == 1
    assert calls[0][1] == "libmxfp4gemv"
    assert calls[0][2]["env_var"] == "K3_GEMV_LIB"
    assert tuple(calls[0][2]["required_symbols"]) == ("mxfp4_gemv_mt",)
    assert fast._LIB_PATH == "/fake/libmxfp4gemv"

    batch, calls = _load_module_with_fake_native("fast_moe_batch.py")
    assert len(calls) == 1
    assert calls[0][1] == "libmxfp4batch"
    assert calls[0][2]["env_var"] == "K3_BATCH_LIB"
    assert {
        "mxfp4_gemv_batch",
        "mxfp4_moe_expert_set",
        "mxfp4_pool_init",
        "mxfp4_pool_shutdown",
    }.issubset(calls[0][2]["required_symbols"])
    assert batch._LIB_PATH == "/fake/libmxfp4batch"

    imported = []
    assert rp.import_when_enabled(
        False, "fast_moe", importer=lambda name: imported.append(name)
    ) is None
    assert imported == []

    # Wiring gate: the heavyweight runner is parsed, never imported.
    source = (HERE / "kimi_run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct = [
        node for node in tree.body
        if (isinstance(node, ast.Import)
            and any(alias.name == "fast_moe" for alias in node.names))
    ]
    assert not direct
    assert 'import_when_enabled(FAST_MOE, "fast_moe")' in source
    assert "torch.mps.synchronize" not in source
    assert source.count("_device_synchronize()") == 3  # definition + two gates


def _device_checks():
    assert rp.choose_device_spec(
        None, mps_available=True, cuda_available=True, cuda_device_count=2
    ) == "mps"
    assert rp.choose_device_spec(
        None, mps_available=False, cuda_available=True, cuda_device_count=2
    ) == "cuda"
    assert rp.choose_device_spec(
        None, mps_available=False, cuda_available=False
    ) == "cpu"
    assert rp.choose_device_spec(
        "cuda:1", mps_available=False, cuda_available=True,
        cuda_device_count=2,
    ) == "cuda:1"
    for requested, error in (
        ("mps", RuntimeError),
        ("cuda", RuntimeError),
        ("cuda:2", ValueError),
        ("xpu", ValueError),
    ):
        try:
            rp.choose_device_spec(
                requested,
                mps_available=False,
                cuda_available=requested.startswith("cuda"),
                cuda_device_count=2 if requested == "cuda:2" else 0,
            )
        except error:
            pass
        else:
            raise AssertionError(f"invalid explicit device {requested} accepted")

    calls = []
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(synchronize=lambda: calls.append(("mps", None))),
        cuda=SimpleNamespace(
            synchronize=lambda device: calls.append(("cuda", device))
        ),
    )
    mps = SimpleNamespace(type="mps")
    cuda = SimpleNamespace(type="cuda")
    cpu = SimpleNamespace(type="cpu")
    rp.synchronize_device(fake_torch, mps)
    rp.synchronize_device(fake_torch, cuda)
    rp.synchronize_device(fake_torch, cpu)
    assert calls == [("mps", None), ("cuda", cuda)]
    assert rp.choose_moe_backend(None, "mps") == "metal"
    assert rp.choose_moe_backend(None, "cuda") == "cpu"
    assert rp.choose_moe_backend(None, "cpu") == "cpu"
    assert rp.choose_moe_backend("metal", "cuda") == "metal"
    try:
        rp.choose_moe_backend("cuda", "cuda")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported MoE backend was accepted")


def _memory_checks():
    gib = rp.GIB
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        proc = root / "proc"
        cg = root / "cgroup"
        (proc / "self").mkdir(parents=True)
        (cg / "demo").mkdir(parents=True)
        (proc / "meminfo").write_text(
            "MemTotal: 131072000 kB\nMemAvailable: 104857600 kB\n",
            encoding="utf-8",
        )
        (proc / "self" / "cgroup").write_text(
            "0::/demo\n", encoding="utf-8"
        )
        (cg / "demo" / "memory.max").write_text(
            str(64 * gib), encoding="utf-8"
        )
        (cg / "demo" / "memory.current").write_text(
            str(20 * gib), encoding="utf-8"
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 64 * gib
        assert memory.effective_available_bytes == 44 * gib
        assert rp.safe_linux_host_budget(memory, 10 * gib) == 34 * gib

        # A constrained parent must win over an unlimited leaf.
        parent = cg / "parent"
        leaf = parent / "leaf"
        leaf.mkdir(parents=True)
        (proc / "self" / "cgroup").write_text(
            "0::/parent/leaf\n", encoding="utf-8"
        )
        (leaf / "memory.max").write_text("max", encoding="utf-8")
        (leaf / "memory.current").write_text(
            str(2 * gib), encoding="utf-8"
        )
        (parent / "memory.max").write_text(
            str(48 * gib), encoding="utf-8"
        )
        (parent / "memory.current").write_text(
            str(30 * gib), encoding="utf-8"
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 48 * gib
        assert memory.effective_available_bytes == 18 * gib

        # Finite capacity with unreadable usage is not treated as free RAM.
        (parent / "memory.current").unlink()
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_available_bytes == 0

        (parent / "memory.max").write_text("0", encoding="utf-8")
        (parent / "memory.current").write_text("0", encoding="utf-8")
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 0
        assert memory.effective_available_bytes == 0

        # A declared but unreadable controller must not fall back to host RAM.
        (parent / "memory.max").unlink()
        (parent / "memory.max").mkdir()
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 0
        assert memory.effective_available_bytes == 0

        # Host exhaustion is meaningful even when a cgroup has room.
        (parent / "memory.max").rmdir()
        (parent / "memory.max").write_text(
            str(48 * gib), encoding="utf-8"
        )
        (proc / "meminfo").write_text(
            "MemTotal: 131072000 kB\nMemAvailable: 0 kB\n",
            encoding="utf-8",
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_available_bytes == 0

    cap = rp.cuda_free_memory_budget(20 * gib, 24 * gib)
    assert 14 * gib < cap < 16 * gib


def _file_hint_checks():
    assert rp.darwin_file_hints_enabled(True, "darwin")
    assert not rp.darwin_file_hints_enabled(True, "linux")

    with mock.patch.object(spine_io, "_IS_DARWIN", False), \
            mock.patch.object(spine_io, "NOCACHE", True), \
            mock.patch.object(spine_io.fcntl, "fcntl") as call:
        assert spine_io._apply_nocache(7, True) is False
        call.assert_not_called()
    with mock.patch.object(spine_io, "_IS_DARWIN", True), \
            mock.patch.object(spine_io.fcntl, "fcntl", return_value=0) as call:
        assert spine_io._apply_nocache(7, True) is True
        call.assert_called_once_with(7, spine_io.F_NOCACHE, 1)
    with mock.patch.object(spine_io, "_IS_LINUX", True), \
            mock.patch.object(
                spine_io.os, "POSIX_FADV_DONTNEED", 4, create=True
            ), mock.patch.object(
                spine_io.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        assert spine_io._drop_read_cache(7, 1024, 4096, True) is True
        call.assert_called_once_with(7, 1024, 4096, 4)
    with tempfile.TemporaryDirectory() as td, \
            mock.patch.object(spine_io, "_IS_DARWIN", False), \
            mock.patch.object(spine_io, "_IS_LINUX", True), \
            mock.patch.object(
                spine_io.os, "POSIX_FADV_WILLNEED", 3, create=True
            ), mock.patch.object(
                spine_io.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        path = os.path.join(td, "layer.bin")
        pathlib.Path(path).write_bytes(b"x" * 32)
        assert spine_io.rdadvise(path, 4, 12) is True
        call.assert_called_once_with(mock.ANY, 4, 12, 3)

    with tempfile.TemporaryDirectory() as td, \
            mock.patch.dict(os.environ, {"DELTAFIN_ROOT": td}):
        import fetch_v2
    assert fetch_v2._pread_nocache_default("darwin") == "1"
    assert fetch_v2._pread_nocache_default("linux") == "0"
    with mock.patch.object(fetch_v2, "_IS_DARWIN", False), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(fetch_v2.fcntl, "fcntl") as call:
        assert fetch_v2._apply_pread_nocache(9) is False
        call.assert_not_called()
    with mock.patch.object(fetch_v2, "_IS_DARWIN", True), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(fetch_v2.fcntl, "fcntl", return_value=0) as call:
        assert fetch_v2._apply_pread_nocache(9) is True
        call.assert_called_once_with(9, fetch_v2.F_NOCACHE, 1)
    with mock.patch.object(fetch_v2, "_IS_DARWIN", False), \
            mock.patch.object(fetch_v2, "_IS_LINUX", True), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(
                fetch_v2.os, "POSIX_FADV_DONTNEED", 4, create=True
            ), mock.patch.object(
                fetch_v2.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        assert fetch_v2._apply_pread_nocache(9) is False
        assert fetch_v2._drop_linux_pread_cache(9) is True
        call.assert_called_once_with(9, 0, 0, 4)

    # The Darwin hint must precede preadv; Linux eviction must follow the last
    # successful read. Exercise the actual _Slot.read control flow with 4 bytes.
    events = []
    slot = object.__new__(fetch_v2._Slot)
    slot.mv = memoryview(bytearray(4))

    def fake_preadv(_fd, views, _offset):
        events.append("read")
        return len(views[0])

    with mock.patch.object(fetch_v2, "EXPERT_SPAN", 4), \
            mock.patch.object(fetch_v2.os, "open", return_value=11), \
            mock.patch.object(fetch_v2.os, "close",
                              side_effect=lambda _fd: events.append("close")), \
            mock.patch.object(fetch_v2.os, "preadv",
                              side_effect=fake_preadv), \
            mock.patch.object(
                fetch_v2, "_apply_pread_nocache",
                side_effect=lambda _fd: events.append("darwin-before"),
            ), mock.patch.object(
                fetch_v2, "_drop_linux_pread_cache",
                side_effect=lambda _fd: events.append("linux-after"),
            ):
        slot.read("unused")
    assert events == ["darwin-before", "read", "linux-after", "close"]


def _spine_cache_linux_checks():
    assert spine_cache.PAGE == os.sysconf("SC_PAGE_SIZE")
    with tempfile.TemporaryDirectory() as td:
        meminfo = pathlib.Path(td) / "meminfo"
        vmstat = pathlib.Path(td) / "vmstat"
        meminfo.write_text(
            "MemFree: 1024 kB\nCached: 2048 kB\n"
            "SReclaimable: 512 kB\nShmem: 256 kB\n",
            encoding="utf-8",
        )
        vmstat.write_text("pswpout 7\n", encoding="utf-8")
        snap = spine_cache._linux_vm_snapshot(
            str(meminfo), str(vmstat), available_cap_bytes=1 << 40
        )
        assert snap["free"] * spine_cache.PAGE == 1024 * 1024
        assert snap["external"] * spine_cache.PAGE == 2304 * 1024
        assert snap["swapouts"] == 7
        assert snap["compressions"] == 0
        capped = spine_cache._linux_vm_snapshot(
            str(meminfo), str(vmstat), available_cap_bytes=0
        )
        assert capped["free"] == capped["external"] == 0

    with mock.patch.object(spine_cache, "_IS_DARWIN", False), \
            mock.patch.object(spine_cache, "_lc",
                              side_effect=AssertionError("Mach called")):
        assert spine_cache.pressure_level() == 1

    memory = rp.LinuxMemory(
        host_total_bytes=128 * rp.GIB,
        host_available_bytes=100 * rp.GIB,
        cgroup_limit_bytes=64 * rp.GIB,
        cgroup_current_bytes=10 * rp.GIB,
    )

    class FakeSpineFast:
        @staticmethod
        def pin(*_args):
            pass

        @staticmethod
        def unpin(*_args):
            pass

    fake_snapshot = dict.fromkeys(spine_cache._VM_KEYS, 0)
    fake_snapshot["free"] = 32 * rp.GIB // spine_cache.PAGE
    with mock.patch.dict(
        os.environ, {"K3_SPINE_CACHE_AUTO": "1"}, clear=True
    ), mock.patch.object(spine_cache, "_IS_LINUX", True), \
            mock.patch.object(
                spine_cache.runtime_platform,
                "linux_memory_limits",
                return_value=memory,
            ), mock.patch.object(
                spine_cache, "vm_snapshot", return_value=fake_snapshot
            ):
        cache = spine_cache.SpineCache(FakeSpineFast())
        expected = rp.safe_linux_host_budget(
            memory, int(cache.floor + cache.file_reserve)
        )
        assert cache.enabled and cache.ceiling == expected

    messages = []
    with mock.patch.dict(
        os.environ, {"K3_SPINE_CACHE_GB": "100"}, clear=True
    ), mock.patch.object(spine_cache, "_IS_LINUX", True), \
            mock.patch.object(
                spine_cache.runtime_platform,
                "linux_memory_limits",
                return_value=memory,
            ), mock.patch.object(
                spine_cache, "vm_snapshot", return_value=fake_snapshot
            ):
        cache = spine_cache.SpineCache(
            FakeSpineFast(), log=messages.append
        )
        expected = rp.safe_linux_host_budget(
            memory, int(cache.floor + cache.file_reserve)
        )
        assert cache.ceiling == expected < 100e9
        assert any("clamped requested" in message for message in messages)
        no_room = dict.fromkeys(spine_cache._VM_KEYS, 0)
        cache.wd.mark = cache.wd.last = no_room
        with mock.patch.object(
            spine_cache, "vm_snapshot", return_value=no_room
        ):
            assert not cache.admit(
                "layer.0.",
                {"q": bytearray(1), "sc": None, "other": None},
                1,
            )


def main():
    _native_loader_checks()
    _native_module_manifest_checks()
    _device_checks()
    _memory_checks()
    _file_hint_checks()
    _spine_cache_linux_checks()
    print("PASS Python portability (no model weights or native execution)")


if __name__ == "__main__":
    main()
