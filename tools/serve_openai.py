#!/usr/bin/env python3
"""Deltafin OpenAI-compatible API server.

Exposes /v1/chat/completions, /v1/completions, and /v1/models over HTTP so any
OpenAI-SDK client, chat UI, or coding agent can talk to Kimi K3 running locally:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=none  <your tool>

Design notes, honestly stated:
  * Decoding is greedy and reproducible; temperature/top_p are accepted and
    ignored. max_tokens defaults to 32 and is clamped to 256 — at roughly a
    token per minute, larger values are a footgun.
  * One generation at a time (a global lock). Concurrency would be meaningless
    at this speed.
  * Chat mode renders K3's template, which includes a thinking section; the
    response splits it into `reasoning_content` and `content` (DeepSeek-style).
  * Each request uses a fresh KV/state cache; nothing is shared across calls.
  * A chat request's prompt is ~60+ tokens; on a COLD expert cache the prefill
    can take hours because most experts get fetched. Warm up with short
    completions first, or let the cache grow across sessions.

Usage:  K3_DEV=mps K3_SPINE=int8 python tools/serve_openai.py [--port 8000]

No dependencies beyond the standard library + what kimi_run already needs.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402

MODEL_ID = "deltafin-kimi-k3"
MAX_TOKENS_CAP = int(os.environ.get("K3_SERVER_MAX_TOKENS", "256"))
RESPONSE_MARKER = "<|open|>response<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"

_lock = threading.Lock()
_tok = None
_layers = None
_embed = None


def _boot():
    global _tok, _layers, _embed
    from transformers import AutoTokenizer
    print("[serve] loading tokenizer + layer skeletons...", flush=True)
    _tok = AutoTokenizer.from_pretrained(
        os.path.join(kr.ROOT, "k3-meta"), trust_remote_code=True)
    kr.check_expert_pool()
    _layers = kr.build_layers()
    _embed = kr.LazyEmbed()
    print("[serve] ready", flush=True)


def _split_reasoning(text):
    """K3 chat output: <think...><|close|>think<|sep|><|open|>response<|sep|><answer>."""
    if RESPONSE_MARKER in text:
        pre, ans = text.rsplit(RESPONSE_MARKER, 1)
        reasoning = pre.replace(THINK_CLOSE, "").replace("<|open|>think<|sep|>", "").strip()
        return reasoning or None, ans
    return None, text


def _gen(ids, max_new, on_delta=None):
    """Run one generation under the global lock; stream decoded-text deltas."""
    cache = kr.ml.KimiDynamicCache(kr.config)
    toks, decoded = [], ""

    def on_token(t):
        nonlocal decoded
        if t == kr.EOS_ID:
            return
        toks.append(t)
        full = _tok.decode(toks)
        delta, decoded = full[len(decoded):], full
        if on_delta and delta:
            on_delta(delta)

    out = kr.generate(_layers, cache, _embed, ids, max_new, on_token=on_token)
    if kr.EOS_ID in out:
        out = out[:out.index(kr.EOS_ID)]
        finish = "stop"
    else:
        finish = "length"
    return out, _tok.decode(out), finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "deltafin"}]})
        else:
            self._err(404, f"no route {self.path}")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        except json.JSONDecodeError:
            return self._err(400, "invalid JSON body")
        chat = self.path in ("/v1/chat/completions", "/chat/completions")
        comp = self.path in ("/v1/completions", "/completions")
        if not (chat or comp):
            return self._err(404, f"no route {self.path}")

        if chat:
            messages = body.get("messages")
            if not messages:
                return self._err(400, "messages required")
            ids = _tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                return self._err(400, "prompt (string) required")
            ids = _tok.encode(prompt)
        max_new = min(int(body.get("max_tokens") or 32), MAX_TOKENS_CAP)
        stream = bool(body.get("stream"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:20]
        created = int(time.time())

        if not _lock.acquire(timeout=5):
            return self._err(429, "a generation is already running (Deltafin serves one at a time)")
        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                def on_delta(delta):
                    if chat:
                        sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "delta": {"content": delta},
                                          "finish_reason": None}]})
                    else:
                        sse({"id": rid, "object": "text_completion", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "text": delta, "finish_reason": None}]})

                _, _, finish = _gen(ids, max_new, on_delta=on_delta)
                key = "delta" if chat else "text"
                sse({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                     "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, key: {} if chat else "", "finish_reason": finish}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            out, text, finish = _gen(ids, max_new)
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                     "total_tokens": len(ids) + len(out)}
            if chat:
                reasoning, content = _split_reasoning(text)
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self._json(200, {"id": rid, "object": "chat.completion", "created": created,
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "message": msg,
                                              "finish_reason": finish}]})
            else:
                self._json(200, {"id": rid, "object": "text_completion", "created": created,
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "text": text,
                                              "finish_reason": finish}]})
        except BrokenPipeError:
            print("[serve] client disconnected mid-generation", flush=True)
        except Exception as e:
            print(f"[serve] error: {e!r}", flush=True)
            try:
                self._err(500, str(e))
            except Exception:
                pass
        finally:
            _lock.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    _boot()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] Deltafin OpenAI-compatible API on http://{args.host}:{args.port}/v1",
          flush=True)
    print("[serve] note: ~1 token/min; first chat request on a cold expert cache "
          "is very slow (large prefill fetch). Warm up with short completions.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
