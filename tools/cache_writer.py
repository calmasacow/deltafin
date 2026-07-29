"""Bounded asynchronous publication for downloaded cache objects.

This module deliberately knows nothing about K3 expert shapes. Its contract is:

* publication is an atomic rename in the destination directory;
* an accepted job owns immutable bytes until publication finishes;
* ``max_pending`` bounds active plus queued jobs (not just queue entries);
* producers block when that bound is reached, providing backpressure;
* failures are retained for inspection and can be raised by ``flush``/``shutdown``.

The default runtime remains synchronous. ``fetch_v2`` creates this writer only
when K3_ASYNC_CACHE_WRITE=1.
"""
from __future__ import annotations

import collections
import itertools
import os
import queue
import threading
import time


_TMP_IDS = itertools.count()
_STOP = object()


class CacheWriteFailure(RuntimeError):
    """One or more accepted asynchronous cache publications failed."""

    def __init__(self, snapshot):
        failures = snapshot.get("recent_failures", ())
        detail = failures[-1] if failures else {}
        message = (
            f"{snapshot.get('failures', 0)} asynchronous cache write(s) failed"
        )
        if detail:
            message += (
                f"; last={detail.get('stage')}:{detail.get('error_type')}:"
                f" {detail.get('message')} ({detail.get('path')})"
            )
        super().__init__(message)
        self.snapshot = snapshot


def atomic_publish(path, payload, *, before_replace=None):
    """Write *payload* beside *path*, then atomically replace *path*.

    ``before_replace`` is a test/diagnostic hook invoked after the temporary
    file is closed and before it becomes visible at the final name. No fsync is
    issued: this preserves the old cache writer's durability contract while
    making name visibility atomic.
    """
    path = os.fsdecode(os.fspath(path))
    view = memoryview(payload).cast("B")
    tmp = (
        f"{path}.tmp{os.getpid()}-{threading.get_ident()}-"
        f"{next(_TMP_IDS)}"
    )
    fd = None
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        written = 0
        while written < len(view):
            n = os.write(fd, view[written:])
            if n <= 0:
                raise IOError(f"short cache write {written}/{len(view)} to {tmp}")
            written += n
        os.close(fd)
        fd = None
        if before_replace is not None:
            before_replace(tmp, path)
        os.replace(tmp, path)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    finally:
        view.release()


class AsyncCacheWriter:
    """A bounded background cache publisher with explicit drain semantics."""

    def __init__(
        self,
        max_pending=4,
        workers=1,
        *,
        publisher=atomic_publish,
        name="k3-cache-write",
    ):
        max_pending = int(max_pending)
        workers = int(workers)
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if workers > max_pending:
            raise ValueError("workers cannot exceed max_pending")

        self.max_pending = max_pending
        self.workers = workers
        self._publisher = publisher
        self._queue = queue.Queue()
        # This semaphore, rather than Queue.maxsize, bounds queued + active
        # payloads exactly. A blocked producer has not copied a mutable payload.
        self._slots = threading.BoundedSemaphore(max_pending)
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._shutdown_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._accepting = True
        self._stopped = False
        self._submitters = 0
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "publish_failures": 0,
            "callback_failures": 0,
            "failures": 0,
            "bytes": 0,
            "pending": 0,
            "pending_peak": 0,
            "backpressure_events": 0,
            "backpressure_s": 0.0,
        }
        self._failures = collections.deque(maxlen=32)
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"{name}-{i}",
                daemon=True,
            )
            for i in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _own(payload):
        # Exact bytes are immutable, so retaining the same object is sufficient.
        # Everything else (bytearray, mmap, numpy, memoryview) is snapshotted.
        return payload if type(payload) is bytes else bytes(payload)

    def submit(self, path, payload, *, on_published=None, on_error=None):
        """Accept a publication, blocking when the pending bound is full.

        Returns ``False`` after shutdown has begun. Callers can then take their
        synchronous fallback without losing the cache object.
        """
        path = os.fsdecode(os.fspath(path))
        # An in-flight count, instead of holding the state lock while blocked,
        # keeps status inspection live under backpressure. Shutdown first stops
        # admission, then waits until every admitted submitter has enqueued.
        with self._state_changed:
            if not self._accepting:
                return False
            self._submitters += 1
        acquired = False
        counted = False
        try:
            if self._slots.acquire(blocking=False):
                waited = 0.0
                pressured = False
            else:
                pressured = True
                t0 = time.perf_counter()
                self._slots.acquire()
                waited = time.perf_counter() - t0
            acquired = True
            try:
                owned = self._own(payload)
            except BaseException:
                self._slots.release()
                acquired = False
                raise
            with self._stats_lock:
                self._stats["submitted"] += 1
                self._stats["pending"] += 1
                counted = True
                self._stats["pending_peak"] = max(
                    self._stats["pending_peak"], self._stats["pending"]
                )
                if pressured:
                    self._stats["backpressure_events"] += 1
                    self._stats["backpressure_s"] += waited
            self._queue.put(
                (path, owned, on_published, on_error),
                block=False,
            )
            acquired = False  # the worker now owns and releases this slot
            counted = False   # the worker now retires the pending count
            return True
        finally:
            if counted:
                with self._stats_lock:
                    self._stats["submitted"] -= 1
                    self._stats["pending"] -= 1
            if acquired:
                self._slots.release()
            with self._state_changed:
                self._submitters -= 1
                self._state_changed.notify_all()

    def _record_failure(self, path, stage, exc):
        record = {
            "path": path,
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        with self._stats_lock:
            self._stats["failures"] += 1
            self._stats[f"{stage}_failures"] += 1
            self._failures.append(record)

    def _notify_error(self, callback, path, stage, exc):
        if callback is None:
            return
        try:
            callback(path, stage, exc)
        except Exception as callback_exc:
            self._record_failure(path, "callback", callback_exc)

    def _run(self):
        while True:
            job = self._queue.get()
            if job is _STOP:
                self._queue.task_done()
                return
            path, payload, on_published, on_error = job
            try:
                self._publisher(path, payload)
                with self._stats_lock:
                    self._stats["completed"] += 1
                    self._stats["bytes"] += len(payload)
                if on_published is not None:
                    try:
                        on_published(path, len(payload))
                    except Exception as exc:
                        self._record_failure(path, "callback", exc)
                        self._notify_error(on_error, path, "callback", exc)
            except Exception as exc:
                self._record_failure(path, "publish", exc)
                self._notify_error(on_error, path, "publish", exc)
            finally:
                # Drop the last payload reference before waking a producer.
                del payload
                with self._stats_lock:
                    self._stats["pending"] -= 1
                self._slots.release()
                self._queue.task_done()

    def snapshot(self):
        with self._state_changed:
            accepting = self._accepting
            stopped = self._stopped
            submitters = self._submitters
        with self._stats_lock:
            out = dict(self._stats)
            out["recent_failures"] = list(self._failures)
        out["accepting"] = accepting
        out["stopped"] = stopped
        out["submitters"] = submitters
        out["max_pending"] = self.max_pending
        out["workers"] = self.workers
        out["alive_workers"] = sum(t.is_alive() for t in self._threads)
        return out

    def _raise_if_failed(self):
        snapshot = self.snapshot()
        if snapshot["failures"]:
            raise CacheWriteFailure(snapshot)
        return snapshot

    def flush(self, *, raise_errors=True):
        """Wait for all currently accepted work."""
        self._queue.join()
        if raise_errors:
            return self._raise_if_failed()
        return self.snapshot()

    def shutdown(self, *, raise_errors=True):
        """Stop accepting work, drain accepted jobs, and join every worker."""
        with self._shutdown_lock:
            with self._state_changed:
                already_stopped = self._stopped
                self._accepting = False
                while self._submitters:
                    self._state_changed.wait()
            if not already_stopped:
                self._queue.join()
                for _ in self._threads:
                    self._queue.put(_STOP)
                self._queue.join()
                for thread in self._threads:
                    thread.join()
                with self._state_changed:
                    self._stopped = True
                    self._state_changed.notify_all()
        if raise_errors:
            return self._raise_if_failed()
        return self.snapshot()
