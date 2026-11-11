"""Smoke test for the Qwen3.5-9B vLLM server (stdlib only: runs with any Python 3.10+).

    python inference/qwen35-9b/smoke_test.py                 # http://localhost:8000/v1, model qwen3.5-9b
    python inference/qwen35-9b/smoke_test.py --base-url http://host:8001/v1 --model qwen3.5-9b

Checks, in order (exit code 1 if any fails):
  1. GET /v1/models lists the served model name
  2. one chat completion in Vietnamese (thinking off)
  3. one tool-call request (thinking on, as in DataMate tool mode): the model must call `revenue` with dates
  4. determinism: the same greedy request twice gives the same text (temperature 0, seed 42)
Env defaults: LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL (same variables as Zalo-Track-2 llm/config.py).
Adapted from the workspace bench_lfm3.py smoke.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

NO_THINK = {"enable_thinking": False}
THINK = {"enable_thinking": True}
TOOLS = [{"type": "function", "function": {
    "name": "revenue",
    "description": "Tổng doanh thu (VND) của các đơn hợp lệ trong khoảng ngày [start, end].",
    "parameters": {"type": "object", "properties": {
        "start": {"type": "string", "description": "YYYY-MM-DD"},
        "end": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["start", "end"]}}}]


def call(base: str, path: str, body: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base: str, model: str, messages: list, max_tokens: int, template: dict, **extra) -> tuple[dict, float]:
    body = {"model": model, "messages": messages, "temperature": 0, "seed": 42, "max_tokens": max_tokens,
            "chat_template_kwargs": template, **extra}
    t = time.perf_counter()
    out = call(base, "/chat/completions", body)
    return out, (time.perf_counter() - t) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("LOCAL_LLM_MODEL", "qwen3.5-9b"))
    a = ap.parse_args()
    ok = True

    def report(name: str, passed: bool, msg: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<11} {msg}")

    try:
        ids = [m["id"] for m in call(a.base_url, "/models", timeout=10)["data"]]
    except (urllib.error.URLError, OSError) as e:
        report("models", False, f"{a.base_url}/models unreachable: {e}")
        return 1
    report("models", a.model in ids, f"served: {ids}")
    if a.model not in ids:
        return 1

    r, ms = chat(a.base_url, a.model, [{"role": "user", "content": "Giải thích ngắn gọn 'doanh thu thuần' là gì."}],
                 96, NO_THINK)
    text = r["choices"][0]["message"].get("content") or ""
    report("chat", bool(text.strip()), f"{ms:6.0f} ms  {r['usage']['completion_tokens']} tok  {text[:100]!r}")

    r, ms = chat(a.base_url, a.model,
                 [{"role": "user", "content": "Hôm nay là 2026-02-10. Doanh thu hôm qua bao nhiêu?"}],
                 2048, THINK, tools=TOOLS, tool_choice="auto")
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    names = [c["function"]["name"] for c in calls]
    shown = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls)
    report("tool call", names[:1] == ["revenue"],
           f"{ms:6.0f} ms  finish={r['choices'][0]['finish_reason']}  "
           + (shown or f"NO tool call: {(msg.get('content') or '')[:120]!r}")
           + ("" if not calls else "  (expected 2026-02-09 .. 2026-02-09)"))

    req = [{"role": "user", "content": "Liệt kê 3 mặt hàng tạp hoá bán chạy ở Việt Nam."}]
    outs = [chat(a.base_url, a.model, req, 48, NO_THINK)[0]["choices"][0]["message"].get("content") for _ in range(2)]
    report("determinism", outs[0] == outs[1], f"identical={outs[0] == outs[1]}")

    print("OK" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
