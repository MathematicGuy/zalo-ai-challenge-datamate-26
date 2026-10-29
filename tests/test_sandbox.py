import pytest

from datamate.answers import normalize_answer
from datamate.io import split_dir
from datamate.sandbox import Executor
from datamate.tables import CACHE, load_business_cached


@pytest.fixture(scope="module")
def ex():
    load_business_cached(split_dir("train"), "biz_101")
    with Executor(CACHE / "biz_train_biz_101.pkl", split_dir("train") / "calendar.csv", timeout=5) as e:
        yield e


def test_answer_and_numpy_coercion(ex):
    r = ex.run("df = tables['ban_hang.csv']; print(df.shape); ANSWER = {'type': 'number', 'answer': np.int64(len(df))}",
               "2026-08-24")
    assert r.ok and normalize_answer(r.answer) == {"type": "number", "answer": r.answer["answer"]}
    assert "(" in r.stdout


def test_error_is_reported(ex):
    r = ex.run("1/0", "2026-08-24")
    assert not r.ok and "ZeroDivisionError" in r.error


def test_timeout(ex):
    r = ex.run("while True: pass", "2026-08-24")
    assert not r.ok and "Timed out" in r.error


def test_jobs_are_isolated(ex):
    assert ex.run("tables.clear(); ANSWER = {'type': 'boolean', 'answer': True}", "2026-08-24").ok
    r = ex.run("ANSWER = {'type': 'date', 'answer': AS_OF}; print(len(tables))", "2026-08-24")
    assert r.answer == {"type": "date", "answer": "2026-08-24"} and r.stdout.strip() == "1"


@pytest.mark.parametrize("raw, want", [
    ({"type": "number", "answer": 12.0}, {"type": "number", "answer": 12}),
    ({"type": "percent", "answer": -12.4567}, {"type": "percent", "answer": -12.46}),
    ({"type": "list", "answer": [3.0, " Bạc xỉu "]}, {"type": "list", "answer": [3, "Bạc xỉu"]}),
    ({"type": "unanswerable"}, {"type": "unanswerable", "reason": ""}),
])
def test_normalize(raw, want):
    assert normalize_answer(raw) == want


@pytest.mark.parametrize("raw", [{"type": "number", "answer": "48250000"}, {"type": "date", "answer": "15/08/2026"},
                                 {"type": "boolean", "answer": 1}, {"answer": 3}, {"type": "foo", "answer": 1}])
def test_normalize_rejects(raw):
    with pytest.raises(ValueError):
        normalize_answer(raw)


def test_run_many_keeps_order_and_isolates_timeouts(ex):
    jobs = [("ANSWER = {'type': 'number', 'answer': 1}", "2026-08-24"),
            ("while True: pass", "2026-08-24"),
            ("ANSWER = {'type': 'number', 'answer': to_number('2,825tr')}", "2026-08-24"),
            ("1/0", "2026-08-24")]
    r = ex.run_many(jobs)
    assert [x.ok for x in r] == [True, False, True, False]
    assert r[0].answer["answer"] == 1 and r[2].answer["answer"] == 2825000.0 and "Timed out" in r[1].error


def test_execute_aligns_replies_with_and_without_code(ex):
    from datamate.agent import execute_replies
    out = execute_replies(ex, [("no code here", "2026-08-24"),
                        ("```python\nANSWER = {'type': 'number', 'answer': 7.0}\n```", "2026-08-24"),
                        ("```python\nANSWER = {'type': 'number', 'answer': '7'}\n```", "2026-08-24")])
    assert out[0][0] is None and "No python code" in out[0][1]["error"]
    assert out[1][0] == {"type": "number", "answer": 7}
    assert out[2][0] is None and "Invalid ANSWER" in out[2][1]["error"]
