"""The agent pipeline end to end with a scripted ChatModel: no GPU, real tables and sandbox."""
import json

import pytest

from datamate.agent import Agent, AgentConfig
from datamate.io import split_dir
from datamate.predict import run
from datamate.prompts import PROMPTS
from datamate.tables import cache_path, load_business_cached

BIZ = "biz_110"
GOOD = "```python\nANSWER = {'type': 'number', 'answer': 42.0}\n```"
OTHER = "```python\nANSWER = {'type': 'number', 'answer': 7}\n```"
BROKEN = "```python\n1/0\n```"


class ScriptedChat:
    """Returns the scripted replies in order, one list of n per conversation; records every call."""

    def __init__(self, *rounds):
        self.rounds, self.calls = list(rounds), []

    def chat(self, conversations, n=1, temperature=0.0, thinking=False):
        self.calls.append({"n": n, "temperature": temperature, "conversations": conversations})
        replies = self.rounds.pop(0)
        return [replies[i % len(replies)][:n] for i in range(len(conversations))]


@pytest.fixture(scope="module")
def biz():
    return load_business_cached(split_dir("train"), BIZ)


def _agent(llm, **cfg):
    data = split_dir("train")
    return Agent(llm, AgentConfig(**cfg), (data / "rulebook.md").read_text(encoding="utf-8"), data / "calendar.csv")


QS = [{"id": "q1", "business_id": BIZ, "question": "dthu hqua la bn", "as_of_date": "2026-07-03"},
      {"id": "q2", "business_id": BIZ, "question": "tỉ lệ đơn huỷ tháng 6/2026 là bao nhiêu %",
       "as_of_date": "2026-07-03"}]


def test_vote_threshold_and_percent_guard(biz):
    llm = ScriptedChat([[GOOD, GOOD, OTHER]])
    run_ = _agent(llm, k=3, tau=0.6).answer_business(biz, cache_path(split_dir("train"), BIZ), QS)
    rows = [r.row for r in run_.results]
    assert rows[0] == {"id": "q1", "type": "number", "answer": 42}          # 2/3 agree >= 0.6
    assert rows[1] == {"id": "q2", "type": "percent", "answer": 42.0}       # '%' question: number -> percent
    assert llm.calls[0]["n"] == 3 and llm.calls[0]["temperature"] == 0.7
    assert len(run_.results[0].log["samples"]) == 3 and run_.results[0].log["support"] == pytest.approx(2 / 3)


def test_abstains_below_threshold(biz):
    run_ = _agent(ScriptedChat([[GOOD, OTHER, BROKEN]]), k=3, tau=0.6).answer_business(
        biz, cache_path(split_dir("train"), BIZ), QS[:1])
    assert run_.results[0].row == {"id": "q1", "type": "abstain"}


def test_repair_only_when_every_sample_failed(biz):
    llm = ScriptedChat([[BROKEN, BROKEN]], [[GOOD]])
    run_ = _agent(llm, k=2, tau=0.5).answer_business(biz, cache_path(split_dir("train"), BIZ), QS[:1])
    assert len(llm.calls) == 2 and llm.calls[1]["temperature"] == 0.0
    repair_turn = llm.calls[1]["conversations"][0][-1]["content"]
    assert "ZeroDivisionError" in repair_turn
    assert run_.results[0].row == {"id": "q1", "type": "number", "answer": 42}   # 1 of k=2 support = 0.5
    assert run_.results[0].log["samples"][-1]["repair"] is True


def test_prompt_wiring(biz):
    llm = ScriptedChat([[GOOD]])
    _agent(llm, k=1, prompt="v2").answer_business(biz, cache_path(split_dir("train"), BIZ), QS[:1])
    system, user = (m["content"] for m in llm.calls[0]["conversations"][0])
    assert "# RULEBOOK" in system and "README.md of biz_110" in system and "{rulebook}" not in system
    assert "to_number(x)" in system                      # helper docs injected
    assert "doanh thu hôm qua" in user                   # abbreviations expanded for v2
    assert llm.calls[0]["temperature"] == 0.0            # k=1 decodes greedily


def test_registry_has_every_version():
    assert set(PROMPTS) == {"v0", "v1", "v2"} and PROMPTS["v0"].uses_context is False


def test_run_writes_submission_logs_and_resumes(tmp_path):
    data = split_dir("train")
    ids = {json.loads(l)["id"] for l in (data / "questions.jsonl").open() if f'"{BIZ}"' in l}
    ids = set(sorted(ids)[:2])
    run(data, tmp_path, AgentConfig(k=1), ids, llm=ScriptedChat([[GOOD]]))
    lines = (tmp_path / "submission.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(l)["answer"] == 42 for l in lines)
    assert len(list((tmp_path / "logs").glob("*.json"))) == 2
    run(data, tmp_path, AgentConfig(k=1), ids, llm=ScriptedChat())   # nothing left: no model call at all
    assert len((tmp_path / "submission.jsonl").read_text().splitlines()) == 2
