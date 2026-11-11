"""The hosted-API engines (llm_api.py) with fake clients: prefill emulation, thinking shape, n fan-out, caching, key rotation."""
from types import SimpleNamespace

import time
import pytest

from datamate import llm_api
from datamate.llm_api import CONTEXT_MARK, ClaudeEngine, GeminiEngine, strip_echo, think_wrap, unprefill
from datamate.prompts import NOTES

CHAT = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "README: shop" + CONTEXT_MARK + "\n2026-01-01\n\nQUESTION: q?"}]


def test_unprefill_moves_assistant_start_into_instruction():
    body, p = unprefill(CHAT + [{"role": "assistant", "content": NOTES}])
    assert p == NOTES and body[-1]["role"] == "user" and NOTES in body[-1]["content"] and len(body) == 2
    assert unprefill(CHAT) == (CHAT, "")


def test_strip_echo():
    assert strip_echo(NOTES + " a\nSELECT 1", NOTES) == " a\nSELECT 1"
    assert strip_echo("  Notes:\n- a", NOTES) == " a"
    assert strip_echo("number", "x\nTYPE:") == "number"


def test_think_wrap():
    assert think_wrap("SQL", False) == "<think>\n</think>\nSQL"
    assert think_wrap("", True) == "<think>\n"  # cut-off thought: no </think>, so the pipeline sees no query


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        prefix = NOTES if "reply_so_far" in str(kw["messages"]) else ""
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=100, cache_creation_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=prefix + " out")], stop_reason="end_turn", usage=usage)


def test_claude_fanout_prefill_cache_and_no_sampling_params(tmp_path):
    fake = FakeMessages()
    eng = ClaudeEngine(client=SimpleNamespace(messages=fake), log_path=tmp_path / "calls.jsonl", workers=4)
    outs = eng.chat([CHAT + [{"role": "assistant", "content": NOTES}]] * 2, eng.params(n=3, temperature=0.7, seed=0, max_tokens=4), prefill=True)
    assert [len(o.outputs) for o in outs] == [3, 3]
    assert all(x.text == " out" for o in outs for x in o.outputs)  # echoed prefill stripped
    kw = fake.calls[0]
    assert "temperature" not in kw and "seed" not in kw and kw["max_tokens"] >= 32
    assert kw["system"][0]["cache_control"] and kw["messages"][0]["content"][0]["cache_control"]
    assert kw["messages"][0]["content"][0]["text"] == "README: shop"  # breakpoint right before DATE ANCHORS
    assert kw["thinking"] == {"type": "disabled"}
    assert len(fake.calls) == 6 and (tmp_path / "calls.jsonl").read_text().count("\n") == 6


def test_claude_thinking_shape():
    eng = ClaudeEngine(client=SimpleNamespace(messages=FakeMessages()))
    out = eng.chat([CHAT], eng.params(n=1, max_tokens=100), thinking=True)[0].outputs[0].text
    assert out == "<think>\n</think>\n out"


class FakeAPIError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg or str(code))
        self.code = code


class FakeModels:
    def __init__(self, name, fail_first, log):
        self.name, self.fail_first, self.log = name, fail_first, log

    def generate_content(self, model, contents, config):
        self.log.append(self.name)
        if self.fail_first and self.log.count(self.name) == 1:
            raise FakeAPIError(429)
        n = config.candidate_count or 1
        cand = SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=f"{self.name}", thought=False)]), finish_reason="STOP")
        return SimpleNamespace(candidates=[cand] * n, usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1))


def test_gemini_rotates_keys_on_429(monkeypatch):
    log = []
    clients = [SimpleNamespace(models=FakeModels(f"k{i}", fail_first=(i < 2), log=log)) for i in range(3)]
    eng = GeminiEngine(clients=clients, workers=1)
    monkeypatch.setattr(eng.errors, "APIError", FakeAPIError)
    eng.cooldown = 0
    out = eng.chat([CHAT], eng.params(n=4, temperature=0.7, top_p=0.95, seed=0, max_tokens=64))[0]
    assert log == ["k0", "k1", "k2"]  # k0 and k1 rate-limited, k2 answered
    assert [x.text for x in out.outputs] == ["k2"] * 4  # n samples via candidate_count in one request
    assert GeminiEngine(clients=clients).workers == 6  # default: 2 x keys


def test_load_env_reads_only_wanted_names(tmp_path, monkeypatch):
    f = tmp_path / "env"
    f.write_text("OTHER_SECRET=nope\nGEMINI_API_KEY_1=a\nGEMINI_API_KEY_2='b'\nANTHROPIC_API_KEY=c\n")
    for k in ("OTHER_SECRET", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATAMATE_ENV_FILE", str(f))
    llm_api.load_env()
    import os
    assert os.environ["GEMINI_API_KEY_2"] == "b" and os.environ["ANTHROPIC_API_KEY"] == "c" and "OTHER_SECRET" not in os.environ


@pytest.mark.parametrize("backend", ["openai", "claude", "gemini"])
def test_predict_accepts_backend(backend, tmp_path):
    from datamate.predict import parse_args
    a = parse_args(["--data", str(tmp_path), "--output", str(tmp_path), "--model", "m", "--backend", backend])
    assert a.backend == backend


def test_gemini_cooldown_from_retry_delay_and_daily_quota():
    eng = GeminiEngine(clients=[SimpleNamespace(models=None)] * 2)
    err = lambda details: SimpleNamespace(code=429, details={"error": {"details": details}})
    assert eng._delay(err([{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}])) == 42
    assert eng._delay(err([{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])) == eng.day_cooldown
    assert eng._delay(SimpleNamespace(code=503, details=None)) == 5.0
    eng._cool = [time.time() + 100, 0.0]  # key 0 cooling: the next pick skips it
    assert eng._pick()[0] == 1


def test_claude_caches_last_turn_and_warms_each_question_once(tmp_path):
    fake = FakeMessages()
    eng = ClaudeEngine(client=SimpleNamespace(messages=fake), workers=4)
    q2 = [CHAT[0], {"role": "user", "content": CHAT[1]["content"].replace("q?", "other?")}]
    eng.chat([CHAT, q2], eng.params(n=3, max_tokens=8))
    msgs = [c["messages"] for c in fake.calls]
    assert all(m[-1]["content"][-1]["cache_control"] for m in msgs)  # breakpoint 3 on the last turn
    q = [m[-1]["content"][-1]["text"].rsplit(" ", 1)[-1] for m in msgs]
    assert len(q) == 6 and q[0] == "q?"  # wave 1: the shop prefix once
    assert sorted(q[1:3]) == ["other?", "q?"]  # wave 2: one per question, before the remaining samples
