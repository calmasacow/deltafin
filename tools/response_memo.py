"""Bounded exact-response memoization for Deltafin's deterministic server.

The OpenAI-compatible server is greedy and keeps one immutable model instance
for its lifetime. Therefore the same mode, prompt-token sequence, and output
limit must produce the same token sequence. This cache exploits only strict
equality; it never guesses a continuation for a merely similar prefix.
"""
from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MemoizedResponse:
    token_ids: tuple[int, ...]
    text: str
    finish_reason: str


class DeterministicResponseMemo:
    def __init__(self, max_entries: int):
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = max_entries
        self._items: collections.OrderedDict[
            tuple[str, tuple[int, ...], int], MemoizedResponse
        ] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
            mode: str, prompt_ids: Iterable[int], max_new: int,
    ) -> tuple[str, tuple[int, ...], int]:
        if max_new <= 0:
            raise ValueError("max_new must be positive")
        return str(mode), tuple(int(value) for value in prompt_ids), int(max_new)

    def get(
            self, mode: str, prompt_ids: Iterable[int], max_new: int,
    ) -> MemoizedResponse | None:
        if self.max_entries == 0:
            return None
        key = self._key(mode, prompt_ids, max_new)
        with self._lock:
            value = self._items.pop(key, None)
            if value is None:
                self.misses += 1
                return None
            self._items[key] = value
            self.hits += 1
            return value

    def put(
            self, mode: str, prompt_ids: Iterable[int], max_new: int,
            token_ids: Iterable[int], text: str, finish_reason: str,
    ) -> None:
        if self.max_entries == 0:
            return
        key = self._key(mode, prompt_ids, max_new)
        value = MemoizedResponse(
            tuple(int(token) for token in token_ids),
            str(text),
            str(finish_reason),
        )
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = value
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
