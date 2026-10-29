import numpy as np
import pandas as pd
import pytest

from datamate.sandbox import Executor, save_tables

from .conftest import TRAIN


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    frames = {"sales": pd.DataFrame({"id": ["A", "B", "C"], "amount": ["55k", "2,825tr", "1.500đ"]})}
    path = save_tables(tmp_path_factory.mktemp("tables") / "t.pkl", frames, "README text")
    with Executor(path, TRAIN / "calendar.csv", timeout=5) as e:
        yield e


def test_answer_numpy_coercion_and_helpers(ex):
    r = ex.run("df = tables['sales']; print(df.shape)\n"
               "ANSWER = {'type': 'number', 'answer': np.int64(parse_money(df.amount).sum())}", "2026-08-24")
    assert r.ok and r.answer == {"type": "number", "answer": 55000 + 2825000 + 1500} and "(3, 2)" in r.stdout


def test_environment(ex):
    r = ex.run("ANSWER = {'type': 'date', 'answer': AS_OF}; print(README, len(calendar) > 0)", "2026-08-24")
    assert r.answer == {"type": "date", "answer": "2026-08-24"} and r.stdout.strip() == "README text True"


def test_error_is_reported(ex):
    r = ex.run("1/0", "2026-08-24")
    assert not r.ok and "ZeroDivisionError" in r.error


def test_missing_answer(ex):
    r = ex.run("x = 1", "2026-08-24")
    assert not r.ok and "without setting ANSWER" in r.error


def test_timeout(ex):
    r = ex.run("while True: pass", "2026-08-24")
    assert not r.ok and "Timed out" in r.error


def test_jobs_are_isolated(ex):
    assert ex.run("tables.clear(); ANSWER = {'type': 'boolean', 'answer': True}", "2026-08-24").ok
    r = ex.run("print(len(tables)); ANSWER = {'type': 'boolean', 'answer': True}", "2026-08-24")
    assert r.stdout.strip() == "1"


def test_run_many_keeps_order_and_isolates_timeouts(ex):
    jobs = [("ANSWER = {'type': 'number', 'answer': 1}", "2026-08-24"),
            ("while True: pass", "2026-08-24"),
            ("ANSWER = {'type': 'number', 'answer': parse_number('2,825tr')}", "2026-08-24"),
            ("1/0", "2026-08-24")]
    r = ex.run_many(jobs)
    assert [x.ok for x in r] == [True, False, True, False]
    assert r[0].answer["answer"] == 1 and r[2].answer["answer"] == 2825000.0 and "Timed out" in r[1].error


def test_numpy_values_become_json(ex):
    r = ex.run("ANSWER = {'type': 'list', 'answer': np.array([1.5, 2.0])}", "2026-08-24")
    assert r.answer == {"type": "list", "answer": [1.5, 2.0]} and np is not None
