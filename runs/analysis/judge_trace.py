"""Scores every answer group of an existing run with a judge model that reads one token, P(Yes) (the flash-sqlite judge recipe, in DuckDB with this shop's rules).
python runs/analysis/judge_trace.py RUN OUT.jsonl --data data/train --views-from RUN --model PATH [--dry-run]. --dry-run builds every prompt on CPU."""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import debtview
import llmviews
import predict as P
from ingest import load_business, profile, schema_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--data", default="data/train")
    p.add_argument("--views-from")
    p.add_argument("--model", default="sqlbase/models/flash-sqlite")
    p.add_argument("--start", type=int, default=8, help="first thinking sample, for debt routing")
    p.add_argument("--top", type=int, default=5, help="largest answer groups judged per question")
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


    def preview(pred):
        x = pred.get("answer")
        if pred["type"] == "table" and isinstance(x, dict):
            rows = x.get("rows", [])
            return len(rows), json.dumps(x.get("columns"), ensure_ascii=False, default=str) + "\n" + "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows[:5])
        if pred["type"] == "list" and isinstance(x, list):
            return len(x), "\n".join(json.dumps(v, ensure_ascii=False, default=str) for v in x[:5])
        return 1, json.dumps(x, ensure_ascii=False, default=str)


    def prompt(ctx, q, as_of, sql, pred):
        n, shown = preview(pred)
        return ("Task Overview:\nYou are a data science expert. Below, you are provided with a database schema, a natural language question, a candidate SQL query and the result of executing it. "
                "Your task is to decide whether the candidate SQL query answers the question correctly.\n\nDatabase Engine:\nDuckDB\n\n"
                f"Database Schema:\n{ctx}\n\nBusiness rules:\n{rulebook}\n\nDate anchors:\n{P.date_anchors(as_of)}\n\nQuestion:\n{q}\n\n"
                f"Candidate SQL:\n```sql\n{sql or '(no query)'}\n```\n\nExecution result ({n} rows, first 5 shown):\n{shown}\n\n"
                "Does the candidate SQL query answer the question correctly, returning exactly the requested information? Answer with one word: Yes or No.")


    if a.dry_run:
        from tokenizers import Tokenizer
        ttok = Tokenizer.from_file(str(R / "sqlbase/models/Qwen3.5-9B/tokenizer.json"))
        sizes = []
        def judge(prompts):
            sizes.extend(len(ttok.encode(x).ids) for x in prompts)
            return [0.5] * len(prompts)
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(str(R / a.model))
        yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
        llm = LLM(model=str(R / a.model), gpu_memory_utilization=0.92, max_model_len=32768, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8",
                  kv_cache_dtype="fp8", max_num_seqs=64, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
        sp = SamplingParams(temperature=0, max_tokens=1, logprobs=2, logprob_token_ids=[yes, no])  # top-k logprobs alone can miss Yes
        def judge(prompts):
            texts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in prompts]
            out = []
            for o in llm.generate(texts, sp, use_tqdm=False):
                lp = o.outputs[0].logprobs[0]
                y, n = math.exp(lp[yes].logprob) if yes in lp else 0.0, math.exp(lp[no].logprob) if no in lp else 0.0
                out.append(y / (y + n) if y + n else 0.5)
            return out
    t0, total = time.time(), 0
    with open(R / a.out, "w", encoding="utf-8") as f:
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
                debt = debtview.describe(d["agree"], d.get("k", 8))
            ctx = "README:\n" + meta["readme"] + "\n\nSCHEMA:\n" + schema_text(meta) + ("\n\nCANONICAL VIEWS:\n" + view if view else "") + ("\n\n" + debt if debt else "")
            db.close()
            items, prompts = [], []
            for t in ts:
                if t["final"].get("reason") or not t["samples"]:
                    continue
                idx = P.debt_pool([s["text"] for s in t["samples"]], a.start)
                groups = {}
                for j in idx:
                    s = t["samples"][j]
                    if s["pred"]:
                        groups.setdefault(P.answer_key(s["pred"]), []).append(s)
                ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:a.top]
                for key, g in ranked:
                    sql = next((x for x in (P.parse(s["text"].split("\n#REPAIR\n")[-1])[1] for s in g) if x), None)
                    pred = P.vote([s["pred"] for s in g], len(g))[0]
                    items.append((t["id"], key, len(g), len(idx), pred))
                    prompts.append(prompt(ctx, t["question"], asof[t["id"]], sql, pred))
            by_q = defaultdict(list)
            for (i, key, n, total_n, pred), py in zip(items, judge(prompts)):
                by_q[i].append({"votes": n, "share": n / max(1, total_n), "pyes": py, "pred": pred})
            for i, gs in by_q.items():
                f.write(json.dumps({"id": i, "groups": gs}, ensure_ascii=False, default=str) + "\n")
            f.flush()
            total += len(prompts)
            print(f"[{b}] {len(by_q)} q, {len(prompts)} judged, {time.time() - t0:.0f}s", flush=True)
    print(f"judged {total} groups in {time.time() - t0:.0f}s", flush=True)
    if a.dry_run:
        print(f"prompt tokens: median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}", flush=True)


if __name__ == "__main__":  # vLLM spawns its engine process, which re-imports this file
    main()
