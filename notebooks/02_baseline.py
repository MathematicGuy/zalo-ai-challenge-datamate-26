# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: datamate (py3.12)
#     language: python
#     name: datamate
# ---

# %% [markdown]
# # 02 — Đánh giá một run trên val300
#
# Chạy trước (ngoài notebook, vì vLLM chiếm GPU):
# ```
# python -m datamate.predict --data data/train --out runs/<run> --ids splits/val300.txt --prompt v2
# ```
# Notebook này chỉ đọc `runs/<run>/submission.jsonl` và `runs/<run>/logs/`.

# %%
import json

import pandas as pd

from datamate.evaluate import evaluate, summarize
from datamate.io import REPO, load_answers

pd.set_option("display.max_colwidth", 120, "display.width", 200)
RUN = REPO / "runs" / "v0_val300"
IDS = set((REPO / "splits" / "val300.txt").read_text().split())
res, df = evaluate(RUN, IDS)
REFS = {r["id"]: r for r in load_answers("train").to_dict("records")}
logs = {p.stem: json.loads(p.read_text()) for p in (RUN / "logs").glob("*.json")}


def samples(lg):  # v0 logs call them "attempts"
    return lg.get("samples", lg.get("attempts", []))
res

# %%
summarize(df, "ref_type")

# %%
pd.crosstab(df.ref_type, df.pred_type, margins=True)

# %%
summarize(df, "business_id").sort_values("PA")

# %% [markdown]
# ## Precision khi trả lời và tác dụng của abstain
# Nếu chỉ trả lời câu nào chạy được ngay lần đầu (không cần sửa), PA đổi ra sao?

# %%
df["attempts"] = df.id.map(lambda i: len(samples(logs[i])) if i in logs else 0)
df["first_try_ok"] = df.id.map(lambda i: i in logs and samples(logs[i])[0].get("error") is None)
print(pd.crosstab(df.attempts, df.score, margins=True))
only_first = df.score.where(df.first_try_ok, 0)
print("PA nếu bỏ qua các câu phải sửa:", round(only_first.mean(), 4), "| PA hiện tại:", round(df.score.mean(), 4))

# %% [markdown]
# ## Câu sai: xem code và output

# %%
def show(qid, max_code=2500):
    lg = logs[qid]
    ref = REFS[qid]
    print(lg["business_id"], lg["as_of_date"], "|", lg["question"])
    print("REF :", ref["type"], json.dumps(ref.get("answer"), ensure_ascii=False)[:200])
    print("PRED:", json.dumps(lg["final"], ensure_ascii=False)[:200])
    for i, a in enumerate(samples(lg)):
        print(f"--- attempt {i}: error={str(a.get('error'))[-300:]}")
        print((a.get("code") or "")[:max_code])
        print(">>> stdout:", (a.get("stdout") or "")[-600:])


wrong = df[df.score == -1]
print(len(wrong), "wrong")
wrong[["id", "business_id", "ref_type", "pred_type", "question"]].head(30)

# %%
for qid in wrong.id.head(3):
    show(qid)
    print("=" * 100)
