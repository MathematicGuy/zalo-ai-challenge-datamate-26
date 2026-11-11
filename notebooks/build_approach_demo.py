"""Writes notebooks/06_approach_demo.ipynb (run: .venv/bin/python notebooks/build_approach_demo.py). Edit here, not in the notebook."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

OVERVIEW = '''flowchart LR
    A[Raw shop files<br/>xlsx / csv / sqlite + README] --> B[Ingest<br/>DuckDB tables, clean headers,<br/>dates, numbers, names]
    B --> C[Canonical views<br/>K LLM drafts of v_sales,<br/>tested on the data, majority]
    B --> D[Debt layer<br/>K drafts of debt macros,<br/>tested, majority]
    C --> E[Prompt per question<br/>schema + views + rulebook +<br/>date hints + matching values + examples]
    D --> E
    E --> F[8 plain SQL samples<br/>2 prompts x 4, LoRA]
    F --> G[Execute read-only<br/>DuckDB, failed ones repaired]
    G -->|vote share < 0.75<br/>or debt question| H[4 thinking samples]
    H --> G
    G --> I[Vote<br/>answer if >= 50% agree,<br/>else abstain]'''

VIEWS = '''stateDiagram-v2
    [*] --> Draft: README + SCHEMA + RULEBOOK
    Draft: K drafts of ONE mapping SELECT (thinking mode)
    Draft --> Test
    Test: build v_sales / v_orders, run data tests (catalog, units, time, splits)
    Test --> Repair: a draft fails
    Repair: error back to the model (up to 2 turns)
    Repair --> Test
    Test --> Vote: drafts that pass
    Vote: group by result fingerprint, most common wins
    Vote --> Used: at least K/4 agree
    Vote --> NoView: too little agreement
    Used: views shown in every prompt of the shop
    NoView: raw tables only (a wrong view is worse than none)'''

SAMPLE = '''stateDiagram-v2
    [*] --> Plain: 8 samples (4 per prompt), temperature 0.7, LoRA
    Plain --> Exec: Notes, TYPE line, one SQL block
    Exec: parse TYPE + SQL, run read-only on DuckDB
    Exec --> Repair: error or implausible result
    Repair: error + hint back, greedy (up to 2 turns)
    Repair --> Exec
    Exec --> Agreement
    Agreement --> Think: vote share < 0.75 or a debt question
    Think: 4 thinking-mode samples with examples
    Think --> Exec
    Agreement --> Vote: enough agreement'''

cells = [
md("""# 06 · The DataMate approach, step by step

**The task** (Zalo AI Challenge 2026, Track 2). A Vietnamese shop owner asks a question about their own messy books, e.g. *"T9/2025 bán được bao nhiêu kg Chuối già?"* (how many kg of bananas were sold in September 2025?). The data is a folder of spreadsheets (xlsx / csv / sqlite) with a README, and every shop records things differently: header rows in odd places, units mixed per line, cancelled orders flagged in notes, debts spread over several sheets. The answer must be **typed** (number, percent, date, string, list, table, boolean or unanswerable). Scoring is **Penalized Accuracy**: +1 right, -1 wrong, 0 for leaving it blank. A wrong answer costs as much as a right one earns, so knowing when *not* to answer matters.

**The approach in one sentence.** Load every shop into DuckDB, let the LLM write a small semantic layer per shop (canonical sales views + debt macros) that is **tested on the data**, then for each question sample **many SQL queries**, execute them, repair failures, and answer with the **majority**, or abstain when the samples disagree.

Why this shape:
- **SQL, not free text.** The model never computes numbers in its head. DuckDB does, exactly.
- **Self-consistency by execution.** Two samples agree when their *results* agree, not their text. Disagreement is a free confidence signal that drives abstention.
- **A tested semantic layer.** Most errors come from misreading a shop's quirks (units, cancellations). Writing that mapping once per shop, checking it against the data, and keeping the majority draft fixes those quirks for every question of the shop.

This notebook follows **one real question** through every stage, calling the same functions the pipeline uses (`datamate/*.py`). Each step has:
- 🎯 its goal;
- 🔀 a diagram of what happens;
- 💻 the code calling the real function;
- 📥📤 its input and output.

Two modes:
- **`replay`** (default) rebuilds the shop layers and re-executes the samples saved by a finished run, with no GPU and no API.
- **`live`** asks the Qwen3.5-9B server (`inference/qwen35-9b`) for fresh samples."""),
code("""# Setup: the repo on sys.path, autoreload so edits in datamate/ show up without a restart
%load_ext autoreload
%autoreload 2
import json, sys, tempfile
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Markdown, display

from datamate import predict, scoring, voting
from datamate.viz import show_io, show_statechart, show_trace

pd.set_option("display.max_colwidth", 120)"""),
code("""# ── Configuration ─────────────────────────────────────────────────────────────────────────────────────────────
SPLIT = "synth_private"           # data/<SPLIT>: synth_private has answers, public_test is scored against the silver key
RUN = "runs/qwen35_private"       # a finished run of that split (scripts/run_compare.sh): its layers and samples are replayed
SHOP = None                       # None = the shop of QID; a shop id like "priv_014" picks its first question
QID = None                        # None = the first question of SHOP (or of the run) that has a trace
MODE = "replay"                   # "live": fresh samples from the Qwen server at localhost:8000

run_dir = ROOT / RUN / "shard0"
trace = {r["id"]: r for r in map(json.loads, (run_dir / "trace.jsonl").open(encoding="utf-8"))} if (run_dir / "trace.jsonl").exists() else {}
questions = {q["id"]: q for q in map(json.loads, (ROOT / "data" / SPLIT / "questions.jsonl").open(encoding="utf-8-sig"))}
if QID is None:
    pool = [i for i in trace if SHOP is None or questions[i]["business_id"] == SHOP] or [i for i, q in questions.items() if q["business_id"] == SHOP]
    QID = pool[0]
q = dict(questions[QID]); SHOP = q["business_id"]
print(f"run {RUN} ({len(trace)} traced questions) | shop {SHOP} | question {QID}\\n  {q['question']}  (as of {q['as_of_date']})")"""),
md("""## The map

| # | Step | What it does | Function | File |
|---|---|---|---|---|
| 1 | Ingest | every sheet becomes a clean DuckDB table | `load_business` | `datamate/ingest.py` |
| 2 | Schema | a compact text description of the tables for the prompt | `schema_text` | `datamate/ingest.py` |
| 3 | Canonical views | the LLM maps raw tables to `v_sales` / `v_orders`; drafts are tested and voted | `Pipeline.draft_views`, `views.check`, `views.pick` | `datamate/pipeline.py`, `datamate/views.py` |
| 4 | Debt layer | the same recipe for debts: macros `debt_at`, `debt_new`, `debt_repaid` | `Pipeline.draft_debts`, `debts.check` | `datamate/debts.py` |
| 5 | Question hints | date windows, values of the shop named in the question, out-of-range periods | `question_hints`, `entity_hits`, `outside` | `datamate/hints.py`, `datamate/ingest.py` |
| 6 | Few-shot examples | similar train questions from *other* shops, with their SQL | `fewshot.Bank.retrieve` | `datamate/fewshot.py` |
| 7 | Prompt | system (rules + rulebook) + user (shop context + hints + examples + question) | `Pipeline.build_chats`, `prompts.chat` | `datamate/prompts.py` |
| 8 | Sampling | 8 plain SQL samples with the LoRA, 2 prompts x 4 | `Pipeline.sample` | `datamate/pipeline.py` |
| 9 | Execute + repair | parse TYPE + SQL, run read-only, feed errors back | `sqlexec.execute`, `Pipeline._repair` | `datamate/sqlexec.py` |
| 10 | Thinking samples | 4 extra reasoning-mode samples when agreement is low | `Pipeline.sample` | `datamate/pipeline.py` |
| 11 | Vote | group samples by answer key, majority, answer or abstain | `voting.vote`, `voting.final_answer` | `datamate/voting.py` |
| 12 | Score | Penalized Accuracy against the reference | `scoring.score` | `datamate/scoring.py` |"""),
code(f"show_statechart('''{OVERVIEW}''')"),
code("""# The pipeline object with the submitted flags (scripts/run_compare.sh). It reads the data and, in replay mode, reuses the run's
# view and debt layers (--views-from) so this notebook rebuilds exactly the prompts that run saw.
out = Path(tempfile.mkdtemp()); (out / "logs").mkdir()   # scratch output: setup_shop writes the layer logs here
flags = ["--data", str(ROOT / "data" / SPLIT), "--output", str(out), "--model", "qwen3.5-9b", "--k", "8", "--think-k", "4",
         "--notes", "--max-tokens", "2048", "--debt-views", "--debt-route", "--canon-names", "--answer-unanswerable",
         "--context", "32768", "--think-budget", "12288", "--fewshot", str(ROOT / "assets/fewshot_bank.jsonl"), "--lora", str(ROOT / "adapters/sql_rft_v3")]
if run_dir.exists():
    flags += ["--views-from", str(run_dir)]
a = predict.parse_args(flags)
from datamate.pipeline import Pipeline
pipe = Pipeline(a)
print("system prompts:", len(pipe.systems), "| few-shot bank:", len(pipe.bank.rows), "train examples")"""),
md("""## Step 1 · Ingest

🎯 **Goal.** Turn a folder of hand-made spreadsheets into clean SQL tables:
- find the real header row, which often sits under a title block;
- parse Vietnamese dates and amounts (`1.250.000đ`, `12/9/25`);
- union sheets that are one table split by month;
- write every name of the shop one way (`--canon-names`: `Chuối già` = `chuoi gia` = `Chuối Già `).

Nothing here uses an LLM."""),
code("""from datamate.ingest import load_business, read_rows
folder = ROOT / "data" / SPLIT / "data" / SHOP
display(pd.DataFrame([{"file": f.name, "KB": round(f.stat().st_size / 1024, 1)} for f in sorted(folder.iterdir())]))
first = next(f for f in sorted(folder.iterdir()) if f.suffix.lower() in (".xlsx", ".csv", ".sqlite", ".db"))
_, sheet, rows = next(iter(read_rows(first)))
raw = pd.DataFrame(rows[:10])
loaded = pipe.load_shops([SHOP])
db, meta = loaded[SHOP]
t0 = next(t for t in meta["tables"] if first.name in t["source"])
clean = db.execute(f'SELECT * FROM "{t0["name"]}" LIMIT 8').df()
show_io(raw, clean, f"{first.name}{' / ' + sheet if sheet else ''}: raw cells → DuckDB table {t0['name']} ({t0['rows']} rows)")"""),
md("""## Step 2 · Schema text

🎯 **Goal.** Describe the shop's tables compactly enough to fit a prompt but rich enough to write SQL without guessing: each column's type, number of distinct values and **example values** (the model must see that the status column says `Đã hủy`, not `cancelled`). The shop's README comes first, since it explains the business in the owner's words."""),
code("""from datamate.ingest import schema_text
print("README (start):\\n" + meta["readme"][:1200] + "\\n...\\n")
print(schema_text(meta)[:3500])"""),
md("""## Step 3 · Canonical views (the semantic layer)

🎯 **Goal.** Every shop names its sales table, columns and statuses differently. Instead of making every question rediscover that, the model writes **one mapping SELECT per shop** to a fixed set of columns (`order_id, order_date, product_name, qty` in base units, `amount`, `is_cancelled`, ...). Generic SQL then derives `v_sales` / `v_orders`: valid orders only, with whole-order cancellation as the rulebook says.

The drafts are not trusted blindly:
1. **K = 8 drafts** in thinking mode.
2. Each is **built and tested on the data**: are all catalog items matched? Are quantities in one unit? Does it span the data's dates?
3. Failures get **repair turns** with the error.
4. The passing drafts are **grouped by their result fingerprint**, and the largest group wins.
5. With too little agreement (< K/4), no view is used. A wrong view would poison every question of the shop."""),
code(f"show_statechart('''{VIEWS}''')"),
code("""import re
from datamate import views
pipe.reuse_layers(loaded)
saved = pipe.vfixed.get(SHOP)
if saved:
    drafts = saved["drafts"]
    print(f"{len(drafts)} drafts saved by the run, {saved['agree']} agree on the chosen mapping, used = {saved.get('used')}")
    checked = views.check(db, drafts)   # re-run the data tests on every draft (no LLM)
    fp_ids = {}
    rows = [{"draft": j, "passed": fp is not None, "group": fp_ids.setdefault(str(fp), len(fp_ids)) if fp else None,
             "error": (err or "")[:110], "SQL (start)": " ".join(views.parse_select(d.split("#REPAIR")[-1]) .split())[:110] if views.parse_select(d.split("#REPAIR")[-1]) else ""}
            for j, (d, (fp, err)) in enumerate(zip(drafts, checked))]
    display(pd.DataFrame(rows))
    print("chosen mapping SELECT:\\n", saved["sql"][:2500] if saved["sql"] else None)
else:
    print("no saved run for this split: run scripts/run_compare.sh first, or set MODE = 'live' and RUN to an existing run")"""),
md("""## Step 4 · Debt layer

🎯 **Goal.** Debt questions ("khách X còn nợ bao nhiêu?", how much does customer X still owe) are the hardest: debts are spread over sales on credit, repayments and opening balances, often in different sheets. The same recipe as the views (K drafts, data tests, repair, majority) produces debt tables plus three macros the questions call: `debt_at(day)` for what is owed at the end of a day, and `debt_new(a, b)` / `debt_repaid(a, b)` for a period. `setup_shop` builds both layers into the shop's database and composes the shop part of the prompt."""),
code("""shop = pipe.setup_shop(SHOP, db, meta)
print("views used:", bool(shop.view), "| debt layer used:", bool(shop.debt))
if shop.view:
    display(shop.db.execute("SELECT * FROM v_sales LIMIT 6").df())
print(shop.debt[:1500] if shop.debt else "(no debt layer for this shop)")"""),
md("""## Step 5 · Question hints

🎯 **Goal.** Ground the question in this shop's data before the model reads it:
- **Date anchors**: what "last month", "hôm qua" (yesterday) or "T9/2025" mean relative to the question's `as_of_date`.
- **Period hints**: the exact date window to filter.
- **Values matching the question**: shop values close to the words in the question (`Chuối già hen` → the catalog's `Chuối già`), so the SQL filters on a value that exists.
- **Out of range**: when every period the question names lies outside the shop's data, the answer is `unanswerable` without asking the model."""),
code("""todo = [q]
chats = pipe.build_chats(shop, todo)   # fills q["_hits"], q["_periods"], q["_oor"] and builds the two chats
from datamate.hints import date_anchors
show_io({"question": q["question"], "as_of_date": q["as_of_date"]},
        {"date anchors": date_anchors(q["as_of_date"]).splitlines()[:8], "period hints": q["_periods"], "values matching": q["_hits"][:8], "outside the data": q["_oor"]},
        "hints for this question")"""),
md("""## Step 6 · Few-shot examples

🎯 **Goal.** Show the model how a similar question maps to TYPE + SQL **on the canonical views**. Examples come from the train split, **from other shops only** (never this shop's own answers), ranked by TF-IDF similarity and de-duplicated by SQL shape. Only the second prompt gets them, so the two prompts disagree in useful ways."""),
code("""ex = pipe.bank.retrieve(q["question"], SHOP, a.fewshot_k, debt=bool(shop.debt)) if shop.view else []
pd.DataFrame([{"shop": r["business_id"], "question": r["question"], "type": r.get("type"), "SQL": " ".join(r["sql"].split())[:140]} for r in ex])"""),
md("""## Step 7 · The prompt

🎯 **Goal.** One self-contained prompt per sample.
- **System**: output format (short notes, a `TYPE:` line, one SQL block), DuckDB rules, then the competition **rulebook** (what counts as revenue, how cancellations work, rounding).
- **User**: the shop context (README, schema, canonical views, debt macros), then the per-question part (date anchors, matching values, examples, question).

The shop context comes first and is identical for every question of the shop, so vLLM's prefix cache (and Claude's prompt cache) reuse it."""),
code("""system, user = chats[0][0][0]["content"], chats[0][0][1]["content"]
print(f"system: {len(system):,} chars | user: {len(user):,} chars | prompt tokens ≈ {(len(system) + len(user)) // 3:,}")
print("\\n── system (start) ──\\n" + system[:1400])
print("\\n── user (end: the per-question part) ──\\n" + user[user.find("DATE ANCHORS"):][:2500])"""),
md("""## Step 8 · Sampling

🎯 **Goal.** Many independent attempts instead of one. Eight plain samples at temperature 0.7 (4 per prompt), with the **SQL LoRA** (`adapters/sql_rft_v3`, trained with rejection sampling on train questions whose SQL produced the right answer). With `--notes`, each sample starts with short notes (the model plans before writing SQL): on hard holdout questions, right samples rose from 25% to 44%.

In `replay` the samples come from the run's trace. In `live` the Qwen server generates new ones through the same `Pipeline.sample`."""),
code(f"show_statechart('''{SAMPLE}''')"),
code("""from datamate.sqlexec import execute, parse
if MODE == "live":
    from datamate.llm_remote import OpenAIEngine
    pipe.engine = OpenAIEngine("qwen3.5-9b")
    texts, preds, errs, pools, n_rep, _, _ = pipe.sample(shop, todo, chats, lambda: False)
    texts, preds, errs, pool = texts[0], preds[0], errs[0], pools[0]
else:
    rec = trace[QID]
    texts = [s["text"] for s in rec["samples"]]
    preds, errs = [], []
    for t in texts:  # re-execute the final query of every saved sample against the rebuilt database
        try:
            preds.append(execute(shop.db, t.split("\\n#REPAIR\\n")[-1], q["question"])); errs.append(None)
        except Exception as e:
            preds.append(None); errs.append(str(e)[:200])
    same = sum(json.dumps(p, sort_keys=True, default=str) == json.dumps(s["pred"], sort_keys=True, default=str) for p, s in zip(preds, rec["samples"]))
    print(f"replayed {len(texts)} samples: {same}/{len(texts)} give the same result as in the run")

def kind(j, t):
    return ("thinking" if "<think>" in t else "plain") + (" + repair" if "#REPAIR" in t else "")
table = pd.DataFrame([{"#": j, "kind": kind(j, t), "TYPE": parse(t.split("\\n#REPAIR\\n")[-1])[0], "SQL": " ".join((parse(t.split("\\n#REPAIR\\n")[-1])[1] or "").split())[:120],
                       "result": p, "error": e} for j, (t, p, e) in enumerate(zip(texts, preds, errs))])
table"""),
code("""print("── one plain sample, as generated ──\\n" + texts[0][:1800])"""),
md("""## Step 9 · Execute and repair

🎯 **Goal.** Turn text into answers safely and catch the model's slips:
- `parse` reads the `TYPE:` line and the SQL block. Only one read-only SELECT is allowed, and the database has external access off.
- `execute` runs the query and converts the result to the answer type: one cell for a number, rows for a list or table.
- **Plausibility lints** become errors on purpose:
  - a % change with no subtraction;
  - a negative gross profit;
  - a debt question answered from raw tables when the debt macros exist.
- A failed sample gets the error plus a hint and answers again: greedy, up to 2 turns. That is the `#REPAIR` marker in the text."""),
code("""rep = [j for j, t in enumerate(texts) if "#REPAIR" in t]
if rep:
    t = texts[rep[0]]
    first_try, fixed = t.split("\\n#REPAIR\\n")[0], t.split("\\n#REPAIR\\n")[-1]
    show_io(first_try[-1500:], fixed[-1500:], f"sample {rep[0]}: first attempt → after repair (result: {preds[rep[0]]})")
else:
    print("no sample of this question needed a repair; pick another QID to see one")"""),
md("""## Step 10 · Thinking samples (adaptive compute)

🎯 **Goal.** Spend more compute only where it is needed. After the 8 plain samples, a question whose vote share is below 0.75, or that asks about debts on a shop with the debt layer, gets **4 thinking-mode samples**: Qwen reasons inside `<think>…</think>` before writing SQL, with the few-shot examples. Easy questions stay cheap. On 873 hard train questions, thinking samples with examples were right 86% of the time vs 71.5% without."""),
code("""th = [j for j, t in enumerate(texts) if "<think>" in t]
print(f"{len(th)} thinking samples for this question")
if th:
    t = texts[th[0]]
    thought = t.split("<think>", 1)[1].split("</think>")[0]
    print("── thought (start) ──\\n" + thought.strip()[:1500] + "\\n...\\n── answer ──\\n" + t.split("</think>")[-1].strip()[:800])"""),
md("""## Step 11 · Vote, answer or abstain

🎯 **Goal.** One answer from many samples, or none.
- Samples are grouped by an **answer key** that follows the scorer: numbers to 2 decimals, text without case or punctuation, a table as its set of rows.
- On a debt question, only samples that call a debt macro vote.
- The largest group wins if it holds **at least 50% of the samples**. Otherwise the pipeline abstains (0 points instead of a likely -1).
- Votes for 0 / empty, and periods outside the data, are handled by rules learned on train."""),
code("""from datamate.hints import DEBT_WORDS
from datamate.ingest import norm
debt_q = bool(shop.debt) and a.debt_route and bool(DEBT_WORDS.search(norm(q["question"])))
pool = voting.debt_pool(texts, 2 * pipe.half, debt_q)
groups = {}
for j in pool:
    if preds[j]:
        groups.setdefault(voting.answer_key(preds[j]), []).append(j)
display(pd.DataFrame([{"answer": preds[js[0]], "samples": js, "votes": len(js)} for js in sorted(groups.values(), key=len, reverse=True)]))
ans, conf = voting.vote([preds[j] for j in pool], len(pool))
final = voting.final_answer(ans, conf, a.threshold, q["_oor"], a.answer_unanswerable)
print(f"voted {ans}  share {conf:.2f}  threshold {a.threshold}  →  final: {final}")"""),
md("""## Step 12 · Score

🎯 **Goal.** Penalized Accuracy for this question: +1 if the type and value match the reference (string and list items also match aliases), -1 if they don't, 0 for an abstain."""),
code("""ans_file = ROOT / "data" / SPLIT / "answers.jsonl"
if ans_file.exists():
    gold, ref = {r["id"]: r for r in map(json.loads, ans_file.open(encoding="utf-8"))}, "answer"
else:
    gold, ref = {r["id"]: r for r in map(json.loads, (ROOT / "data/silver_public/submission.jsonl").open(encoding="utf-8"))}, "silver answer"
s = scoring.score(final, gold[QID])
print(f"reference ({ref}): {gold[QID]}\\nfinal: {final}\\nscore: {s:+d}")"""),
md("""## The whole path of this question"""),
code("""steps = [{"kind": "pre", "label": f"Ingest {SHOP}", "detail": f"{len(meta['tables'])} tables"},
         {"kind": "llm", "label": "Views + debt layer", "detail": f"views {'used' if shop.view else 'not used'}, debts {'used' if shop.debt else 'none'}"},
         {"kind": "pre", "label": "Hints + examples", "detail": f"{len(q['_hits'])} matching values, {len(ex)} examples"}]
n_plain = sum("<think>" not in t for t in texts)
steps += [{"kind": "llm", "label": f"{n_plain} plain samples", "detail": f"{sum(p is not None for p, t in zip(preds, texts) if '<think>' not in t)} executed"}]
if rep:
    steps += [{"kind": "error", "label": f"{len(rep)} repaired", "detail": "error fed back"}]
if th:
    steps += [{"kind": "llm", "label": f"{len(th)} thinking samples", "detail": "low agreement or debt"}]
steps += [{"kind": "post", "label": f"Vote share {conf:.2f}", "detail": str(final)[:50]},
          {"kind": "exec" if s == 1 else "error" if s == -1 else "post", "label": f"Score {s:+d}", "detail": ref}]
show_trace(steps, q["question"])"""),
md("""## Where to go next

- `observation.ipynb` compares this pipeline run with **Qwen3.5-9B + LoRA**, **Claude Haiku 5.5** and **Gemini 3.5 Flash Lite** on the public test (silver key) and the synthetic private test (real answers).
- Change `QID` / `SHOP` above to follow another question, e.g. a debt question, or one that abstained.
- `MODE = "live"` regenerates the samples with the Qwen server. Start it with `bash inference/qwen35-9b/server.sh start` and the settings in `inference/qwen35-9b/README.md`."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("06_approach_demo.ipynb"))
print("wrote notebooks/06_approach_demo.ipynb")
