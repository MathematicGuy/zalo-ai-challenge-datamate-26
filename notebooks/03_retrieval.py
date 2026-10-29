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
# # 03 — Rule retrieval: lexical vs bi-encoder vs hybrid
#
# Labels: `splits/rule_labels_train.json`, 48 val300 questions hand-labelled (by Claude) with the train
# rulebook rows that change how each is computed (§9 format rows excluded). Small and single-annotator:
# read differences under ~5 points as noise.
#
# Metrics per question, averaged: Recall@k (share of its labelled rules in the top k) and
# Precision@k (share of the top k that are labelled; low precision = distracting context).

# %%
import json

import pandas as pd

from datamate.io import REPO, load_questions, split_dir
from datamate.retrieval import DenseRuleRetriever, LexicalRuleRetriever, rrf

rulebook = (split_dir("train") / "rulebook.md").read_text(encoding="utf-8")
labels = json.loads((REPO / "splits" / "rule_labels_train.json").read_text())["labels"]
Q = load_questions("train").set_index("id").loc[list(labels)]
lex = LexicalRuleRetriever(rulebook)
names = [r.name for r in lex.rules]
gold = {qid: {names.index(n) for n in ns} for qid, ns in labels.items()}
questions = Q.question.tolist()
print(len(names), "rules,", len(gold), "labelled questions")

# %%
rankings = {"lexical": [lex.rank(q) for q in questions]}
for model in ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3"]:
    idx = DenseRuleRetriever(rulebook, model)
    short = model.split("/")[1]
    rankings[short] = [idx.rank(q) for q in questions]
    rankings[f"hybrid lexical+{short}"] = [rrf([a, b]) for a, b in zip(rankings["lexical"], rankings[short])]
rankings["hybrid lexical+both"] = [rrf([a, b, c]) for a, b, c in
                                   zip(rankings["lexical"], rankings["Qwen3-Embedding-0.6B"], rankings["bge-m3"])]


def metrics(ranks, ks=(1, 3, 5, 8)):
    out = {}
    for k in ks:
        rec = [len(set(r[:k]) & gold[q]) / len(gold[q]) for q, r in zip(gold, ranks)]
        prec = [len(set(r[:k]) & gold[q]) / k for q, r in zip(gold, ranks)]
        out[f"R@{k}"], out[f"P@{k}"] = round(sum(rec) / len(rec), 3), round(sum(prec) / len(prec), 3)
    return out


res = pd.DataFrame({name: metrics(r) for name, r in rankings.items()}).T
res

# %% [markdown]
# ## Misses of the best method at k=5

# %%
best = res["R@5"].idxmax()
print("best at R@5:", best)
for (qid, r), q in zip(zip(gold, rankings[best]), questions):
    miss = gold[qid] - set(r[:5])
    if miss:
        print(f"- {q}\n    missed: {[names[i] for i in miss]}\n    top5:   {[names[i] for i in r[:5]]}")
