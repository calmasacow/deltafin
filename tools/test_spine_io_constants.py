#!/usr/bin/env python3
"""Focused checks for Darwin constants used by the spine reader."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import spine_io


def main():
    # Values from the current macOS SDK:
    #   <sys/resource.h> IOPOL_DEFAULT=0, IOPOL_IMPORTANT=1
    #   <sys/fcntl.h>    F_RDADVISE=44, F_NOCACHE=48
    assert spine_io.IOPOL_IMPORTANT == 1
    assert spine_io.F_RDADVISE == 44
    assert spine_io.F_NOCACHE == 48
    assert spine_io.IOPOL_IMPORTANT != 0

    class FakeLibC:
        def __init__(self):
            self.iopol_args = None
            self.qos_args = None

        def setiopolicy_np(self, *args):
            self.iopol_args = args
            return 0

        def pthread_set_qos_class_self_np(self, *args):
            self.qos_args = args
            return 0

    fake = FakeLibC()
    old_iopol, old_libc = spine_io.IOPOL, spine_io._libc
    try:
        spine_io.IOPOL = True
        spine_io._libc = fake
        assert spine_io.set_process_io_policy() is True
        assert fake.iopol_args == (
            spine_io.IOPOL_TYPE_DISK,
            spine_io.IOPOL_SCOPE_PROCESS,
            1,
        )
        assert spine_io.set_thread_qos() is True
        assert fake.qos_args == (spine_io.QOS_CLASS_USER_INITIATED, 0)
    finally:
        spine_io.IOPOL, spine_io._libc = old_iopol, old_libc
    print("PASS spine_io Darwin constants")


if __name__ == "__main__":
    main()
