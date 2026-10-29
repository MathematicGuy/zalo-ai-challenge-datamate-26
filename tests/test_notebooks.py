"""The notebooks' library cells (01_data, 02_agent) under test: normalization, data layer, voting,
answer handling and the agent end to end with a scripted model (no GPU)."""
import json
import os

import pandas as pd
import pytest

from datamate.nblib import run_lib_cells

from .conftest import REPO, TRAIN


@pytest.fixture(scope="session")
def nb():
    os.chdir(REPO / "notebooks")
    ns = {}
    run_lib_cells("02_agent.ipynb", ns)   # also loads 01_data's library cells
    return ns


# ---------------------------------------------------------------- question normalization

@pytest.mark.parametrize("raw, want", [
    ("dthu hqua cua cua hang la bn z", "doanh thu hôm qua cua cua hang la bao nhiêu vậy"),
    ("Dthu th1/2026 đc bnhiu z?", "doanh thu tháng 1/2026 được bao nhiêu vậy?"),
    ("T10/2025 dthu hơn hay kém", "tháng 10/2025 doanh thu hơn hay kém"),
    ("3/26 thợ mô làm nhiều nhất rứa?", "3/26 thợ nào làm nhiều nhất vậy?"),
    ("bình quân mỗi o đờ", "bình quân mỗi đơn"),
])
def test_expands(nb, raw, want):
    assert nb["expand_abbreviations"](raw) == want


@pytest.mark.parametrize("text", ["check giùm", "hơn hay kém", "biển số 59-K1", "giá 55k", "mua nhiu đơn nhất",
                                  "Ống PVC Bình Minh D50 cây 4m", "Topping Thạch cà phê", "Size XL"])
def test_leaves_alone(nb, text):
    assert nb["expand_abbreviations"](text) == text


# ---------------------------------------------------------------- data layer

def test_normalization_on_a_hand_kept_ledger(nb):
    biz = nb["load_business_cached"]("public_test", "dev_004")
    key = "ALL SoBanHang.xlsx::T*"
    assert biz.combined[key][0] == "SoBanHang.xlsx::T01-2025" and len(biz.combined[key]) == 18
    df = biz.frames[key]
    assert df["Thời gian__date"].dtype.kind == "M" and df["Thời gian__date"].notna().all()  # '1/1' + sheet year
    assert df["Amount__num"].notna().mean() > 0.99                                           # '5tr520', '90k'
    assert biz.filled[key] == "Ma don" and df["Ma don"].notna().all()
    blank_lines = df.groupby("Ma don")["Khách hàng (tên)"].apply(lambda s: s.isna().any() and s.notna().any())
    assert not blank_lines.any()                                                              # filled down


def test_description_points_to_parsed_columns(nb):
    biz = nb["load_business_cached"]("public_test", "dev_004")
    d = nb["describe_business"](biz)
    assert "use 'Thời gian__date' (datetime64" in d and "'ALL SoBanHang.xlsx::T*'" in d
    assert d.count("### tables['SoBanHang.xlsx::T0") == 0            # parts described once, via the ALL view


def test_parse_plan_never_reads_plain_ints_as_dates(nb):
    df = pd.DataFrame({"tien": [45000, 46000, 45500] * 10, "ngay": ["01/04/2025", "02/04/2025", "1 thg 5 2025"] * 10})
    plan = nb["plan_parsing"](df.astype(object), "x")
    assert "tien" not in plan and plan["ngay"][0] == "date"


# ---------------------------------------------------------------- answers and voting

N = lambda v: {"type": "number", "answer": v}


def test_vote_majority_support_counts_failures(nb):
    v = nb["vote"]([N(10), N(10), N(12), None, N(10.0)], 5)
    assert v["answer"] == N(10) and v["support"] == 0.6 and [c["size"] for c in v["clusters"]] == [3, 1]


def test_vote_clusters_like_the_scorer(nb):
    a, b = {"type": "string", "answer": "Sting dâu 330ml"}, {"type": "string", "answer": "STING DÂU 330ML"}
    assert nb["answer_key"](a) == nb["answer_key"](b)
    t1 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["X", 1], ["Y", 2]]}}
    t2 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["y", 2], ["x", 1]]}}
    assert nb["answer_key"](t1) == nb["answer_key"](t2)
    assert nb["answer_key"](N(5)) != nb["answer_key"]({"type": "percent", "answer": 5})


def test_decide(nb):
    v = nb["vote"]([N(1), N(2), None], 3)
    assert nb["decide"](v, 0.5) == {"type": "abstain"} and nb["decide"](v, 0.3) == N(1)
    u = {"type": "unanswerable", "reason": "x"}
    assert nb["decide"](nb["vote"]([u, u, N(3)], 3), 0.6) == u


@pytest.mark.parametrize("raw, want", [
    ({"type": "number", "answer": 12.0}, N(12)),
    ({"type": "percent", "answer": -12.4567}, {"type": "percent", "answer": -12.46}),
    ({"type": "list", "answer": [3.0, " Bạc xỉu "]}, {"type": "list", "answer": [3, "Bạc xỉu"]}),
    ({"type": "unanswerable"}, {"type": "unanswerable", "reason": ""}),
])
def test_normalize_answer(nb, raw, want):
    assert nb["normalize_answer"](raw) == want


@pytest.mark.parametrize("raw", [N("48250000"), {"type": "date", "answer": "15/08/2026"},
                                 {"type": "boolean", "answer": 1}, {"answer": 3}, {"type": "foo", "answer": 1}])
def test_normalize_answer_rejects(nb, raw):
    with pytest.raises(ValueError):
        nb["normalize_answer"](raw)


def test_percent_guard(nb):
    assert nb["postprocess"]("tỉ lệ đơn huỷ là bao nhiêu %", N(12.5)) == {"type": "percent", "answer": 12.5}
    assert nb["postprocess"]("doanh thu tháng 5", N(12)) == N(12)


def test_prompt_v2_is_unchanged_and_v3_differs(nb):
    assert "Parse every raw cell with the helpers" in nb["SYSTEM_V2"]
    assert "__date" in nb["SYSTEM_V3"] and "Parse every raw cell with the helpers" not in nb["SYSTEM_V3"]


# ---------------------------------------------------------------- agent end to end

GOOD = "```python\nANSWER = {'type': 'number', 'answer': 42.0}\n```"
OTHER = "```python\nANSWER = {'type': 'number', 'answer': 7}\n```"
BROKEN = "```python\n1/0\n```"
QS = [{"id": "q1", "business_id": "biz_110", "question": "dthu hqua la bn", "as_of_date": "2026-07-03"},
      {"id": "q2", "business_id": "biz_110", "question": "tỉ lệ đơn huỷ tháng 6/2026 là bao nhiêu %",
       "as_of_date": "2026-07-03"}]


def _agent(nb, llm, **cfg):
    return nb["Agent"](llm, nb["AgentConfig"](**cfg), (TRAIN / "rulebook.md").read_text(encoding="utf-8"),
                       TRAIN / "calendar.csv")


def _answer(nb, agent, qs):
    biz = nb["load_business_cached"]("train", "biz_110")
    return agent.answer_business(biz, nb["sandbox_tables"]("train", biz), qs)


def test_vote_threshold_and_percent_guard(nb):
    llm = nb["ScriptedChat"]([[GOOD, GOOD, OTHER]])
    run = _answer(nb, _agent(nb, llm, k=3, tau=0.6), QS)
    assert run.rows == [{"id": "q1", "type": "number", "answer": 42}, {"id": "q2", "type": "percent", "answer": 42.0}]
    assert llm.calls[0]["n"] == 3 and llm.calls[0]["temperature"] == 0.7
    assert run.logs[0]["support"] == pytest.approx(2 / 3) and len(run.logs[0]["samples"]) == 3


def test_abstains_below_threshold(nb):
    run = _answer(nb, _agent(nb, nb["ScriptedChat"]([[GOOD, OTHER, BROKEN]]), k=3, tau=0.6), QS[:1])
    assert run.rows == [{"id": "q1", "type": "abstain"}]


def test_repair_only_when_every_sample_failed(nb):
    llm = nb["ScriptedChat"]([[BROKEN, BROKEN]], [GOOD])
    run = _answer(nb, _agent(nb, llm, k=2, tau=0.5), QS[:1])
    assert len(llm.calls) == 2 and llm.calls[1]["temperature"] == 0.0
    assert "ZeroDivisionError" in llm.calls[1]["conversations"][0][-1]["content"]
    assert run.rows == [{"id": "q1", "type": "number", "answer": 42}] and run.logs[0]["samples"][-1]["repair"]


def test_prompt_wiring(nb):
    llm = nb["ScriptedChat"]([GOOD])
    _answer(nb, _agent(nb, llm, k=1, prompt="v3"), QS[:1])
    system, user = (m["content"] for m in llm.calls[0]["conversations"][0])
    assert "# RULEBOOK" in system and "README.md of biz_110" in system and "{rulebook}" not in system
    assert "parse_number(x)" in system and "doanh thu hôm qua" in user and llm.calls[0]["temperature"] == 0.0


def test_run_split_writes_resumes_and_shards(nb, tmp_path):
    ids = sorted(json.loads(line)["id"] for line in (TRAIN / "questions.jsonl").open()
                 if '"biz_110"' in line or '"biz_114"' in line)[:4]
    cfg = nb["AgentConfig"](k=1)
    for i in range(2):
        nb["run_split"]("train", tmp_path, cfg, set(ids), nb["ScriptedChat"]([GOOD]), shard=f"{i}/2")
    lines = (tmp_path / "submission.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["id"] for x in lines) == ids and len(list((tmp_path / "logs").glob("*.json"))) == 4
    nb["run_split"]("train", tmp_path, cfg, set(ids), nb["ScriptedChat"]([BROKEN]), shard="0/2")  # resumes: no new work
    assert all(json.loads(x)["answer"] == 42 for x in (tmp_path / "submission.jsonl").read_text().splitlines())


@pytest.mark.parametrize("name", ["01_data.ipynb", "02_agent.ipynb", "03_evaluate.ipynb"])
def test_committed_notebooks_carry_no_outputs(name):
    """Outputs hold competition data (customer names, questions): regulations §4.3. Clear them before committing."""
    import nbformat
    cells = nbformat.read(REPO / "notebooks" / name, as_version=4).cells
    assert not any(c.get("outputs") or c.get("execution_count") for c in cells if c.cell_type == "code")
