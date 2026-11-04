"""Runs the reviewer (sqlbase/roles.py) over the samples of an existing run, so no new samples are needed.
python runs/analysis/review_trace.py RUN OUT.jsonl --data data/train [--views-from RUN] [--dry-run]. --dry-run builds every prompt on CPU with a scripted reply."""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import debtview
import llmviews
import predict as P
import roles
from ingest import entity_hits, load_business, profile, schema_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--data", default="data/train")
    p.add_argument("--views-from")
    p.add_argument("--model", default="sqlbase/models/Qwen3.5-9B")
    p.add_argument("--below", type=float, default=0.75)
    p.add_argument("--start", type=int, default=8, help="first thinking sample, for debt routing")
    p.add_argument("--think-budget", type=int, default=24576)
    p.add_argument("--context", type=int, default=49152)
    p.add_argument("--retry-from", help="keep the choices of an earlier output, review only the replies it lost to the budget")
    p.add_argument("--only-abstained", action="store_true", help="review only questions the vote leaves unanswered")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run, data = R / a.run, R / a.data
    layers = R / (a.views_from or a.run) / "logs"
    tr = defaultdict(list)
    for line in open(run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        tr[t["business_id"]].append(t)
    asof = {q["id"]: q["as_of_date"] for q in (json.loads(x) for x in open(data / "questions.jsonl", encoding="utf-8-sig") if x.strip())}
    rulebook = (data / "rulebook.md").read_text(encoding="utf-8")
    rsys = roles.REVIEW_SYSTEM + "\n\nRULEBOOK:\n" + rulebook
    rrows, _ = roles.number_lines(rulebook)
    if a.dry_run:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(R / a.model / "tokenizer.json"))
        sizes = []
        def generate(chats):
            sizes.extend(len(tok.encode(c[0]["content"] + c[1]["content"]).ids) for c in chats)
            return ["</think>\nCHOICE: NONE"] * len(chats)
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(R / a.model), gpu_memory_utilization=0.92, max_model_len=a.context, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8",
                  kv_cache_dtype="fp8", max_num_seqs=128, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
        def generate(chats):
            sp = SamplingParams(n=1, temperature=0, max_tokens=a.think_budget + 256, seed=0)
            return [o.outputs[0].text for o in llm.chat(chats, sp, chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)]
    t0, c = time.time(), Counter()
    prev = {r["id"]: r for r in map(json.loads, open(R / a.retry_from, encoding="utf-8")) if r["choice"]} if a.retry_from else {}
    with open(R / a.out, "w", encoding="utf-8") as out:
        for b, ts in sorted(tr.items()):
            db, meta = load_business(data, b)
            v = json.loads((layers / f"_views_{b}.json").read_text(encoding="utf-8"))
            d = json.loads((layers / f"_debt_{b}.json").read_text(encoding="utf-8")) if (layers / f"_debt_{b}.json").exists() else {}
            view = debt = ""
            if v.get("sql"):
                llmviews.build(db, v["sql"])
                if v["agree"] * 4 >= v.get("k", 8):
                    view = llmviews.describe(db, v["sql"], v["agree"], v.get("k", 8))
                    meta["tables"] += [profile(db, n, "canonical view") for n in ("v_sales", "v_orders")]
            if d.get("sqls") and d.get("used"):
                debtview.build(db, d["sqls"])
                debt = debtview.describe(d["agree"], d.get("k", 8))
            ctx = "README:\n" + meta["readme"] + "\n\nSCHEMA:\n" + schema_text(meta) + ("\n\nCANONICAL VIEWS:\n" + view if view else "") + ("\n\n" + debt if debt else "")
            items = []
            for t in ts:
                if t["final"].get("reason") or not t["samples"]:
                    continue
                q = {"id": t["id"], "question": t["question"], "as_of_date": asof[t["id"]]}
                pool = P.debt_pool([s["text"] for s in t["samples"]], a.start)
                if a.only_abstained:
                    ans, conf = P.vote([t["samples"][j]["pred"] for j in pool], len(pool))
                    if P.decide(ans, conf, 0.5):
                        continue
                cands = roles.candidates(q["question"], [(t["samples"][j]["text"], t["samples"][j]["pred"]) for j in pool], P.answer_key, P.parse, lambda ps: P.vote(ps, len(ps))[0], a.below)
                if not cands:
                    continue
                q["_hits"] = entity_hits(db, meta, q["question"])
                q["_periods"] = P.period_hints(q["question"], q["as_of_date"], db)
                q["_rules"] = roles.rule_block(rrows, t.get("picks") or [])
                items.append((t["id"], cands, roles.review_chat(rsys, P.chat("", ctx, q)[1]["content"], cands)))
            for i, _, _ in [x for x in items if x[0] in prev]:
                c[str(prev[i]["choice"])] += 1
                out.write(json.dumps(prev[i], ensure_ascii=False, default=str) + "\n")
            items = [x for x in items if x[0] not in prev]
            for (i, cands, _), text in zip(items, generate([x[2] for x in items])):
                choice = roles.parse_choice(text, cands)
                c[str(choice)] += 1
                out.write(json.dumps({"id": i, "choice": choice, "candidates": cands, "reply": text[-1500:]}, ensure_ascii=False, default=str) + "\n")
            out.flush()
            db.close()
            print(f"[{b}] {len(items)} reviewed, {time.time() - t0:.0f}s, so far {c.most_common()}", flush=True)
    print(f"reviewed {sum(c.values())} in {time.time() - t0:.0f}s: {c.most_common()}", flush=True)
    if a.dry_run:
        print(f"prompt tokens: median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}", flush=True)


if __name__ == "__main__":  # vLLM spawns its engine process, which re-imports this file
    main()
