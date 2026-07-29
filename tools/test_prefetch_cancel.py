#!/usr/bin/env python3
"""Regression test for fetch_v2's two-phase wrong-prefetch cancellation.

No files are read.  The first fake future represents a pread already running;
its ``result()`` asserts that every queued loser was offered cancellation before
the reader waited.  The historical one-future-at-a-time drain fails this test
because it blocks on the running future first.
"""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2  # noqa: E402


class Future:
    def __init__(self, name, running, events, queued):
        self.name = name
        self.running = running
        self.events = events
        self.queued = queued

    def cancel(self):
        self.events.append(("cancel", self.name))
        return not self.running

    def result(self):
        self.events.append(("wait", self.name))
        assert all(("cancel", name) in self.events for name in self.queued), (
            "waited for active speculation before cancelling every queued loser")


def main():
    events = []
    names = ["queued-1", "queued-2", "queued-3"]
    jobs = {
        0: (object(), Future("active", True, events, names)),
        1: (object(), Future(names[0], False, events, names)),
        2: (object(), Future(names[1], False, events, names)),
        3: (object(), Future(names[2], False, events, names)),
    }
    reader = object.__new__(fetch_v2._PreadReader)
    returned = []
    reader._give = lambda slots: returned.extend(slots)
    before = fetch_v2.stats["prefetch_wasted"]
    reader._drain(jobs)
    assert len(returned) == 4
    assert fetch_v2.stats["prefetch_wasted"] - before == 4
    assert events == [
        ("cancel", "active"),
        ("cancel", "queued-1"),
        ("cancel", "queued-2"),
        ("cancel", "queued-3"),
        ("wait", "active"),
    ], events
    print("PASS: all queued speculative reads cancel before active-read wait")


if __name__ == "__main__":
    main()
