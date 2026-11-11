"""Hosted-API engines (Claude, Gemini) behind llm.Engine's interface, for comparing models on the same pipeline.

What differs from the local engine, and how it is bridged:
- n samples: one request per sample (Gemini first tries candidate_count=n in one request).
- prefill (continue the last assistant turn): neither API accepts it (Claude Haiku 5.5 returns 400). The trailing assistant text
  becomes an instruction to output only its continuation; an echoed copy of it is stripped from the reply.
- thinking: the API's own thinking mode. The reply is wrapped as "<think>\\n</think>\\n" + text so the pipeline's parser,
  which keys on </think> to spot a cut-off thought, reads it like a Qwen thinking sample.
- lora: ignored. Sampling seed/temperature: Gemini takes them; Claude Haiku 5.5 rejects non-default values, so it gets none.
- Prompt caching (Claude): breakpoints after the system prompt and after the shop context (the part of the user message before
  DATE ANCHORS, shared by every question of the shop). The first request of each shared prefix runs before the rest so the
  others read the cache instead of all writing it at once.

Keys come from the environment or the repo's `env` file (DATAMATE_ENV_FILE overrides the path). Only the names this module
needs are read from that file: ANTHROPIC_API_KEY (or CLAUDE_API), and every GEMINI_API_KEY* for rotation.
"""
import json
import os
import random
import re
import threading
import time
from pathlib import Path

from .llm_remote import RemoteEngine, _out

REPO = Path(__file__).resolve().parents[1]
CONTEXT_MARK = "\n\nDATE ANCHORS:"
CONTINUE = ("\n\n---\nYour reply has already been started; its beginning is between the <reply_so_far> tags. "
            "Output ONLY the text that continues it, without repeating any of it.\n<reply_so_far>\n{p}\n</reply_so_far>")


def load_env(names=("ANTHROPIC_API_KEY", "CLAUDE_API"), prefixes=("GEMINI_API_KEY",)):
    """Set the wanted variables from the env file when not already in the environment. Other lines are skipped unread."""
    path = Path(os.environ.get("DATAMATE_ENV_FILE", REPO / "env"))
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if not sep or not (name in names or name.startswith(prefixes)):
            continue
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(name):
            os.environ[name] = value


def unprefill(chat):
    """[..., user, assistant(P)] -> ([..., user + instruction to continue P], P). Chats without a trailing assistant pass through."""
    if not chat or chat[-1]["role"] != "assistant":
        return chat, ""
    p, body = chat[-1]["content"], list(chat[:-1])
    if body and body[-1]["role"] == "user":
        body[-1] = {"role": "user", "content": body[-1]["content"] + CONTINUE.format(p=p)}
    else:
        body.append({"role": "user", "content": CONTINUE.format(p=p).lstrip()})
    return body, p


def strip_echo(text, p):
    if not p:
        return text
    if text.startswith(p):
        return text[len(p):]
    t, q = text.lstrip(), p.strip()
    return t[len(q):] if q and t.startswith(q) else text


def think_wrap(text, cut):
    """A thinking-mode reply in the Qwen shape. cut: the token cap hit before any answer (a cut-off thought, no query)."""
    return "<think>\n" if cut and not text.strip() else "<think>\n</think>\n" + text


class APIEngine(RemoteEngine):
    min_tokens = 32  # a 4-token TYPE fill is too tight once an API adds its own framing
    cache_full = False  # also warm one request per distinct chat (explicit prompt caching with a breakpoint on the last turn)

    def request(self, chat, p, thinking, n):
        """-> (texts, cut flags) for n samples of one chat. Implemented per API."""
        raise NotImplementedError

    def prefix_key(self, chat):
        sys_ = "".join(m["content"] for m in chat if m["role"] == "system")
        user = next((m["content"] for m in chat if m["role"] == "user"), "")
        return hash((sys_, user.split(CONTEXT_MARK)[0]))

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        p = dict(params, max_tokens=max(params.get("max_tokens", 1024), self.min_tokens))
        n = p.get("n", 1)
        convs = [unprefill(c) if prefill else (c, "") for c in chats]

        def run(i):
            body, pre = convs[i]
            try:
                texts, cuts = self.request(body, p, thinking, n)
            except Exception as e:  # retries are inside request(); a hard failure counts as failed samples
                self.log({"error": f"{type(e).__name__}: {str(e)[:300]}", "thinking": thinking})
                texts, cuts = [""] * n, [False] * n
            texts = [strip_echo(t, pre) for t in texts]
            return _out([think_wrap(t, c) if thinking else t for t, c in zip(texts, cuts)])

        # prompt-cache warm-up in waves: one request per shared shop prefix, then (cache_full) one per distinct chat so the n
        # samples of a question read its tail instead of all writing it, then the rest in parallel
        keys = [self.prefix_key] + ([lambda c: hash(repr(c))] if self.cache_full else [])
        res, todo = {}, list(range(len(convs)))
        for key in keys:
            wave, seen = [], set()
            for i in todo:
                k = key(convs[i][0])
                if k not in seen:
                    seen.add(k)
                    wave.append(i)
            if len(wave) == len(todo):
                break
            res.update(zip(wave, self._map(run, wave)))
            todo = [i for i in todo if i not in res]
        res.update(zip(todo, self._map(run, todo)))
        return [res[i] for i in range(len(convs))]


# ---- Claude -----------------------------------------------------------------------------------------------------------
HAIKU_PRICE = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}  # $/MTok, prompts <= 100K tokens


class ClaudeEngine(APIEngine):
    """Prompt caching as in Zalo-Track-2 llm/client.py ClaudeLLM (breakpoint after the shared prefix, warmed once before the
    parallel calls), plus breakpoints after the shop context and on the last turn: the n samples of a question share it all."""
    backend = "claude"
    cache_full = True

    def __init__(self, model="claude-haiku-5-5", *, workers=10, effort="medium", think_effort="high", client=None, **kw):
        super().__init__(model, workers=workers, **kw)
        if client is None:
            import anthropic
            load_env()
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not found in the environment or the env file")
            client = anthropic.Anthropic(api_key=key, max_retries=8, timeout=600)
        self.client, self.effort, self.think_effort = client, effort, think_effort

    @staticmethod
    def build(chat):
        """(system blocks, messages) with cache breakpoints after the system prompt and the shop context."""
        sys_ = "\n\n".join(m["content"] for m in chat if m["role"] == "system")
        msgs, first = [], True
        for m in chat:
            if m["role"] == "system":
                continue
            content = m["content"]
            if first and m["role"] == "user":
                first = False
                ctx, mark, rest = content.partition(CONTEXT_MARK)
                blocks = [{"type": "text", "text": ctx, "cache_control": {"type": "ephemeral"}}]
                if mark:
                    blocks.append({"type": "text", "text": mark.lstrip("\n") + rest})
                content = blocks
            msgs.append({"role": m["role"], "content": content})
        if msgs:  # breakpoint 3: the whole chat, read by the other samples of the same question
            last = msgs[-1]
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            last["content"][-1] = dict(last["content"][-1], cache_control={"type": "ephemeral"})
        system = [{"type": "text", "text": sys_, "cache_control": {"type": "ephemeral"}}] if sys_ else []
        return system, msgs

    def _call(self, system, msgs, p, thinking):
        kw = dict(model=self.model, max_tokens=p["max_tokens"], messages=msgs,
                  thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
                  output_config={"effort": self.think_effort if thinking else self.effort})
        if system:
            kw["system"] = system
        t0 = time.time()
        r = self.client.messages.create(**kw)
        u = r.usage
        tok = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
               "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0, "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        cost = (tok["input_tokens"] * HAIKU_PRICE["input"] + tok["output_tokens"] * HAIKU_PRICE["output"]
                + tok["cache_read_tokens"] * HAIKU_PRICE["cache_read"] + tok["cache_write_tokens"] * HAIKU_PRICE["cache_write"]) / 1e6
        self.log({"thinking": thinking, **tok, "cost_usd": round(cost, 6), "latency_s": round(time.time() - t0, 2), "finish": r.stop_reason})
        text = "".join(b.text for b in r.content if b.type == "text")
        return ("" if r.stop_reason == "refusal" else text), r.stop_reason == "max_tokens"

    def request(self, chat, p, thinking, n):
        system, msgs = self.build(chat)
        res = [self._call(system, msgs, p, thinking) for _ in range(n)]  # n>1: chat() already runs chats in parallel
        return [t for t, _ in res], [c for _, c in res]

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        n = params.get("n", 1)
        if n == 1:
            return super().chat(chats, params, thinking=thinking, prefill=prefill)
        # spread the n samples over the pool too: one job per (chat, sample), regrouped afterwards
        flat = super().chat([c for c in chats for _ in range(n)], dict(params, n=1), thinking=thinking, prefill=prefill)
        return [_out([o.outputs[0].text for o in flat[i * n:(i + 1) * n]]) for i in range(len(chats))]


# ---- Gemini -----------------------------------------------------------------------------------------------------------
class GeminiEngine(APIEngine):
    """Gemini over google-genai with one client per GEMINI_API_KEY*. Requests take the keys round-robin; a 429 or 5xx puts that
    key on cooldown (the server's retryDelay when it sends one, longer for a per-day quota) and the request moves to the next
    free key, waiting for the first one to cool down when all are busy. A request gives up only after `patience` seconds, so a
    per-minute quota wall slows the run instead of turning samples into empty answers."""
    backend = "gemini"
    patience, cooldown, day_cooldown = 1800.0, 30.0, 3600.0

    def __init__(self, model="gemini-3.5-flash-lite", *, workers=None, clients=None, think_level="HIGH", fast_level="MINIMAL", **kw):
        from google import genai
        from google.genai import errors, types
        if clients is None:
            load_env()
            keys = [v for k, v in sorted(os.environ.items()) if k.startswith("GEMINI_API_KEY") and v]
            if not keys:
                raise RuntimeError("no GEMINI_API_KEY* found in the environment or the env file")
            clients = [genai.Client(api_key=k) for k in keys]
        super().__init__(model, workers=workers or 2 * len(clients), **kw)
        self.clients, self.errors, self.types = clients, errors, types
        self.think_level, self.fast_level = think_level, fast_level
        self._next, self._klock = 0, threading.Lock()
        self._cool = [0.0] * len(clients)  # key -> time it may be used again
        self.multi = True  # candidate_count=n in one request until the model refuses it

    def _pick(self):
        """(key, client, 0) for the next free key in round-robin order, or (None, None, seconds until one is free)."""
        with self._klock:
            now, m = time.time(), len(self.clients)
            for j in range(m):
                k = (self._next + j) % m
                if self._cool[k] <= now:
                    self._next = (k + 1) % m
                    return k, self.clients[k], 0.0
            return None, None, min(self._cool) - now

    def _config(self, system, p, thinking, n, seed):
        t = self.types
        return t.GenerateContentConfig(
            system_instruction=system or None, temperature=p.get("temperature"), top_p=p.get("top_p"), top_k=p.get("top_k"),
            seed=seed, max_output_tokens=p["max_tokens"], candidate_count=n if n > 1 else None,
            thinking_config=t.ThinkingConfig(thinking_level=self.think_level if thinking else self.fast_level))

    def _delay(self, e):
        """Cooldown for a rate-limited key: retryDelay from the error details, an hour for a per-day quota."""
        details = ((e.details or {}).get("error") or {}).get("details") or [] if isinstance(getattr(e, "details", None), dict) else []
        delay = self.cooldown if e.code == 429 else 5.0
        for d in details:
            if "PerDay" in json.dumps(d.get("violations", [])):
                return self.day_cooldown
            m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                delay = float(m.group(1)) + 1
        return delay

    def _generate(self, contents, cfg):
        last, deadline = None, time.time() + self.patience
        while time.time() < deadline:
            k, client, wait = self._pick()
            if client is None:
                time.sleep(min(wait, 60.0) + random.uniform(0, 1))
                continue
            try:
                return k, client.models.generate_content(model=self.model, contents=contents, config=cfg)
            except self.errors.APIError as e:
                last = e
                if e.code != 429 and not 500 <= (e.code or 0) < 600:
                    raise
                with self._klock:
                    self._cool[k] = max(self._cool[k], time.time() + self._delay(e))
        raise RuntimeError(f"Gemini failed for {self.patience:.0f}s over {len(self.clients)} keys: {last}")

    def _one(self, system, contents, p, thinking, n, seed):
        t0 = time.time()
        k, r = self._generate(contents, self._config(system, p, thinking, n, seed))
        u = r.usage_metadata
        texts, cuts = [], []
        for c in (r.candidates or [])[:n]:
            parts = (c.content.parts if c.content else None) or []
            texts.append("".join(x.text for x in parts if x.text and not x.thought))
            cuts.append(str(c.finish_reason).endswith("MAX_TOKENS"))
        texts += [""] * (n - len(texts)); cuts += [False] * (n - len(cuts))
        self.log({"thinking": thinking, "n": n, "key": k, "input_tokens": getattr(u, "prompt_token_count", None),
                  "output_tokens": getattr(u, "candidates_token_count", None), "thought_tokens": getattr(u, "thoughts_token_count", None),
                  "cache_read_tokens": getattr(u, "cached_content_token_count", None), "latency_s": round(time.time() - t0, 2),
                  "finish": [str(c.finish_reason) for c in (r.candidates or [])]})
        return texts, cuts

    def request(self, chat, p, thinking, n):
        system = "\n\n".join(m["content"] for m in chat if m["role"] == "system")
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in chat if m["role"] != "system"]
        seed = p.get("seed")
        if n > 1 and self.multi:
            try:
                return self._one(system, contents, p, thinking, n, seed)
            except self.errors.APIError as e:
                if e.code != 400 or not re.search(r"candidate", str(e), re.I):
                    raise
                self.multi = False
        texts, cuts = [], []
        for i in range(n):
            t, c = self._one(system, contents, p, thinking, 1, None if seed is None else seed + i)
            texts += t; cuts += c
        return texts, cuts
