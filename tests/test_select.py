from datamate.select import answer_key, decide, vote

N = lambda v: {"type": "number", "answer": v}


def test_majority_wins_and_support_counts_failures():
    v = vote([N(10), N(10), N(12), None, N(10.0)], k=5)
    assert v["answer"] == N(10) and v["support"] == 0.6 and [c["size"] for c in v["clusters"]] == [3, 1]


def test_strings_cluster_by_scorer_normalization():
    a = {"type": "string", "answer": "Sting dâu 330ml"}
    b = {"type": "string", "answer": "STING DÂU 330ML"}
    assert answer_key(a) == answer_key(b)
    assert vote([a, b, a], k=3)["support"] == 1.0


def test_tables_cluster_ignoring_row_order():
    t1 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["X", 1], ["Y", 2]]}}
    t2 = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["y", 2], ["x", 1]]}}
    assert answer_key(t1) == answer_key(t2)


def test_type_matters():
    assert answer_key(N(5)) != answer_key({"type": "percent", "answer": 5})


def test_decide_threshold_and_all_failed():
    v = vote([N(1), N(2), None], k=3)
    assert decide(v, 0.5) == {"type": "abstain"} and decide(v, 0.3) == N(1)
    assert decide(vote([None, None], k=2), 0.0) == {"type": "abstain"}
    u = {"type": "unanswerable", "reason": "x"}
    assert decide(vote([u, u, N(3)], k=3), 0.6) == u
