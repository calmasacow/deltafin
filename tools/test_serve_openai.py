#!/usr/bin/env python3
"""Focused generation-boundary tests for the OpenAI-compatible server."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_openai as server


class _Tokenizer:
    def decode(self, ids):
        return ",".join(str(token) for token in ids)


class ServerGenerationBoundaryTests(unittest.TestCase):
    def test_nonstreaming_request_installs_no_token_callback(self):
        observed = {}

        def generate(
            layers,
            cache,
            embed,
            ids,
            max_new,
            on_token=None,
            universal_drafter=None,
        ):
            observed["on_token"] = on_token
            observed["universal_drafter"] = universal_drafter
            return [7, 8, server.kr.EOS_ID]

        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(server.kr, "generate", side_effect=generate),
            mock.patch.object(
                server.kr, "IncrementalTokenDecoder"
            ) as decoder,
        ):
            out, text, finish = server._gen([1, 2], 3)
        self.assertIsNone(observed["on_token"])
        self.assertIsNone(observed["universal_drafter"])
        decoder.assert_not_called()
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "stop")

    def test_streaming_preserves_delta_order_tail_and_eos_filter(self):
        deltas = []

        class Decoder:
            def append(self, token):
                return {7: "", 8: "eight"}[token]

            def finish(self):
                return "tail"

        def generate(
            layers,
            cache,
            embed,
            ids,
            max_new,
            on_token=None,
            universal_drafter=None,
        ):
            self.assertIsNotNone(on_token)
            self.assertIsNone(universal_drafter)
            on_token(7)
            on_token(8)
            on_token(server.kr.EOS_ID)
            return [7, 8, server.kr.EOS_ID]

        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(server.kr, "generate", side_effect=generate),
            mock.patch.object(
                server.kr, "IncrementalTokenDecoder", return_value=Decoder()
            ),
        ):
            out, text, finish = server._gen(
                [1, 2], 3, on_delta=deltas.append
            )
        self.assertEqual(deltas, ["eight", "tail"])
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "stop")

    def test_length_finish_keeps_last_non_eos_token(self):
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(
                server.kr, "generate", return_value=[7, 8]
            ),
        ):
            out, text, finish = server._gen([1, 2], 2)
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "length")


if __name__ == "__main__":
    unittest.main()
