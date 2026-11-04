"""The exploration agent (sqlbase/explore.py) on the questions of a run where plain samples disagreed: 4 independent runs each, on base weights unless --lora.
python runs/analysis/explore_trace.py RUN OUT.jsonl --data data/train --views-from RUN [--lora PATH] [--dry-run]. --dry-run replays a scripted model on CPU."""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import debtview
import explore
import fewshot
import harness
import llmviews
import predict as P
import prompts
import roles
from ingest import entity_hits, load_business, profile, schema_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--data", default="data/train")
    p.add_argument("--views-from")
    p.add_argument("--model", default="sqlbase/models/Qwen3.5-9B")
    p.add_argument("--lora")
    p.add_argument("--bank", default="sqlbase/fewshot_bank24.jsonl")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--limit-shops", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run, data = R / a.run, R / a.data
    layers = R / (a.views_from or a.run) / "logs"
    tr = defaultdict(list)
    for line in open(run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if len(t["samples"]) > 8 and not t["final"].get("reason"):
            tr[t["business_id"]].append(t)
    asof = {q["id"]: q["as_of_date"] for q in (json.loads(x) for x in open(data / "questions.jsonl", encoding="utf-8-sig") if x.strip())}
    rulebook = (data / "rulebook.md").read_text(encoding="utf-8")
    rrows, _ = roles.number_lines(rulebook)
    system = prompts.SYSTEM_V3 + explore.ADDON + "\n\nRULEBOOK:\n" + rulebook
    bank = fewshot.Bank(R / a.bank)
    if a.dry_run:
        def generate(convs, keys):
            out = []
            for c in convs:
                n = sum(m["role"] == "assistant" for m in c)
                out.append(["ACTION: VALUES\nTERM: a", "ACTION: RULE\nTERM: don hang", "ACTION: QUERY\n```sql\nSELECT 1 AS x\n```", "ACTION: FINAL\nTYPE: number\n```sql\nSELECT 1\n```"][min(n, 3)])
            return out
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(model=str(R / a.model), gpu_memory_utilization=0.92, max_model_len=32768, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8", kv_cache_dtype="fp8",
                  max_num_seqs=128, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN", **({"enable_lora": True, "max_lora_rank": 64, "max_loras": 1} if a.lora else {}))
        lk = {"lora_request": LoRARequest("sql", 1, str(R / a.lora))} if a.lora else {}
        def generate(convs, keys):
            sps = [SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=(1000 + k[1]) * (len(c) + 1)) for c, k in zip(convs, keys)]
            return [o.outputs[0].text for o in llm.chat(convs, sps, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False, **lk)]
    t0, c = time.time(), Counter()
    with open(R / a.out, "w", encoding="utf-8") as out:
        for b, ts in sorted(tr.items())[:a.limit_shops or None]:
            db, meta = load_business(data, b)
            db.execute("SET enable_external_access = false")
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
            known = harness.known_values(db)
            tools = explore.Tools(db, P.run_sql, known, rrows)
            items = {}
            for t in ts:
                q = {"id": t["id"], "question": t["question"], "as_of_date": asof[t["id"]]}
                q["_hits"] = entity_hits(db, meta, q["question"])
                q["_periods"] = P.period_hints(q["question"], q["as_of_date"], db)
                shots = fewshot.block(bank.retrieve(q["question"], b, 4, debt=bool(debt))) if view else ""
                for k in range(a.k):
                    items[t["id"], k] = (P.chat(system, ctx, q, shots), tools)
            done = explore.run_batch(generate, items, a.steps)
            preds, errs = {}, {}
            for key, s in done.items():
                if s["final"] is None:
                    errs[key] = "no final"
                    continue
                try:
                    preds[key] = P.execute(db, s["final"], next(t["question"] for t in ts if t["id"] == key[0]))
                except Exception as e:
                    errs[key] = str(e)[:300]
            retry = [k for k in errs if done[k]["final"] is not None]
            if retry:
                convs = [done[k]["conv"] + [{"role": "user", "content": f"ERROR: {errs[k]}\n{P.repair_hint(errs[k])}Fix the query. Reply with ACTION: FINAL, the TYPE line and one ```sql``` block."}] for k in retry]
                for k, reply in zip(retry, generate(convs, retry)):
                    done[k]["steps"].append("REPAIR")
                    done[k]["final"] += "\n#REPAIR\n" + reply
                    try:
                        preds[k] = P.execute(db, reply, next(t["question"] for t in ts if t["id"] == k[0]))
                        errs.pop(k, None)
                    except Exception as e:
                        errs[k] = str(e)[:300]
            for t in ts:
                samples = [{"text": done[t["id"], k]["final"] or "", "pred": preds.get((t["id"], k)), "error": errs.get((t["id"], k)), "steps": done[t["id"], k]["steps"]} for k in range(a.k)]
                out.write(json.dumps({"id": t["id"], "samples": samples}, ensure_ascii=False, default=str) + "\n")
                for s in samples:
                    c["answered" if s["pred"] else "no answer"] += 1
                    c.update(s["steps"])
            out.flush()
            db.close()
            print(f"[{b}] {len(ts)} q, {time.time() - t0:.0f}s, {c.most_common()}", flush=True)
    print(f"done in {time.time() - t0:.0f}s: {c.most_common()}", flush=True)


if __name__ == "__main__":  # vLLM spawns its engine process, which re-imports this file
    main()
