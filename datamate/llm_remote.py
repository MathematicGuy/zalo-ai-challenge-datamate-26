"""Engines that call a model over HTTP, behind the same interface as llm.Engine (chat / params / n_tokens / SamplingParams),
so pipeline.py runs unchanged on any of them:

    OpenAIEngine   an OpenAI-compatible server: the tuned Qwen3.5-9B vLLM server in inference/qwen35-9b (LoRA served as "sql")
    ClaudeEngine   Anthropic Messages API        (llm_api.py)
    GeminiEngine   Google Gemini API, key rotation (llm_api.py)

chat() returns one object per conversation shaped like vLLM's RequestOutput: `.outputs[i].text` for the n samples. Requests run
on a thread pool. A request that still fails after its retries yields empty texts (the pipeline treats them as failed samples)
instead of killing a multi-hour run; the error is logged. Every call is appended to `log_path` (JSONL: tokens, latency, cost).
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


def _out(texts):
    return SimpleNamespace(outputs=[SimpleNamespace(text=t) for t in texts])


class RemoteEngine:
    """Shared plumbing: sampling params as dicts, a thread pool, the call log."""
    backend = "remote"

    def __init__(self, model, *, workers=8, log_path=None, seed=0):
        self.model, self.workers, self.seed = model, workers, seed
        self.log_path = log_path
        self._lock = threading.Lock()
        self.SamplingParams = self.params  # debts.draft builds its params through engine.SamplingParams(...)

    @staticmethod
    def params(**kw):
        return dict(kw)

    def log(self, rec):
        if not self.log_path:
            return
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": self.backend, "model": self.model, **rec}
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _map(self, fn, items, workers=None):
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(workers or self.workers, len(items))) as ex:
            return list(ex.map(fn, items))

    def n_tokens(self, chat):  # rough count (Vietnamese text runs ~3 characters per token); API contexts are large
        return sum(len(m["content"]) for m in chat) // 3 + 8 * len(chat)


class OpenAIEngine(RemoteEngine):
    """An OpenAI-compatible vLLM server. lora=True asks for the served adapter (`lora_name`), else the base model.
    Start the server with REASONING_PARSER= so the text comes back raw, as the in-process engine returns it."""
    backend = "openai"

    def __init__(self, model="qwen3.5-9b", *, base_url="http://localhost:8000/v1", lora_name="sql", workers=16, timeout=1800, **kw):
        super().__init__(model, workers=workers, **kw)
        from openai import OpenAI
        import requests
        self.client = OpenAI(base_url=base_url, api_key="local", timeout=timeout, max_retries=3)
        self.root = base_url.rsplit("/v1", 1)[0]
        self.http = requests.Session()
        served = {m["id"]: m for m in self.http.get(self.root + "/v1/models", timeout=30).json()["data"]}
        if model not in served:
            raise RuntimeError(f"{model} is not served at {base_url} (served: {sorted(served)})")
        self.lora_name = lora_name if lora_name in served else None
        self.max_len = served[model].get("max_model_len") or 32768

    def n_tokens(self, chat):
        r = self.http.post(self.root + "/tokenize", json={"model": self.model, "messages": chat, "add_generation_prompt": True}, timeout=60)
        r.raise_for_status()
        return r.json()["count"]

    def _one(self, job):
        chat, p, thinking, model, prefill = job
        extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if "top_k" in p:
            extra["top_k"] = p["top_k"]
        if prefill:
            extra.update(add_generation_prompt=False, continue_final_message=True)
        kw = dict(model=model, messages=chat, n=p.get("n", 1), temperature=p.get("temperature", 1.0), top_p=p.get("top_p", 1.0),
                  max_tokens=p.get("max_tokens", 1024), extra_body=extra)
        if p.get("seed") is not None:
            kw["seed"] = p["seed"]
        t0 = time.time()
        for attempt in range(2):
            try:
                r = self.client.chat.completions.create(**kw)
                texts = [c.message.content or "" for c in sorted(r.choices, key=lambda c: c.index)]
                self.log({"lora": model != self.model, "thinking": thinking, "n": kw["n"], "prompt_tokens": r.usage.prompt_tokens,
                          "output_tokens": r.usage.completion_tokens, "latency_s": round(time.time() - t0, 2),
                          "finish": [c.finish_reason for c in r.choices]})
                return _out(texts)
            except Exception as e:  # prompt + max_tokens over the context: shrink max_tokens to what is left, once
                msg = str(e).lower()
                if attempt == 0 and ("context length" in msg or "maximum context" in msg or "max_tokens" in msg):
                    left = self.max_len - self.n_tokens(chat) - 8
                    if left > 16:
                        kw["max_tokens"] = min(kw["max_tokens"], left)
                        continue
                self.log({"error": str(e)[:300], "thinking": thinking, "latency_s": round(time.time() - t0, 2)})
                return _out([""] * kw["n"])

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        model = self.lora_name if lora and self.lora_name else self.model
        return self._map(self._one, [(c, params, thinking, model, prefill) for c in chats])
