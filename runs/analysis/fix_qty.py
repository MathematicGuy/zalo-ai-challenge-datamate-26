"""Rewrites the qty column of saved views into base units with one model turn per shop (sqlbase/llmviews.py), keeping the other columns.
python runs/analysis/fix_qty.py --data data/public_test --views-from DIR --out DIR [--dry-run]. --dry-run checks the old views against themselves on CPU."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import llmviews
from ingest import load_business, schema_text


def select_of(reply):
    try:
        return llmviews.parse_select(reply)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--views-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="sqlbase/models/Qwen3.5-9B")
    p.add_argument("--only", help="comma-separated shops; the others are left as they are")
    p.add_argument("--budget", type=int, default=12288)
    p.add_argument("--turns", type=int, default=2, help="repair turns")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    data, src, dst = R / a.data, R / a.views_from / "logs", R / a.out / "logs"
    dst.mkdir(parents=True, exist_ok=True)
    rulebook = (data / "rulebook.md").read_text(encoding="utf-8")
    shops = sorted(f.name[7:-5] for f in src.glob("_views_*.json") if not a.only or f.name[7:-5] in a.only.split(","))
    loaded, chats = {}, {}
    for b in shops:
        v = json.loads((src / f"_views_{b}.json").read_text(encoding="utf-8"))
        db, meta = load_business(data, b)
        loaded[b] = db, v
        if v.get("sql"):
            chats[b] = llmviews.qty_chat(rulebook, meta["readme"], schema_text(meta), v["sql"])
    if a.dry_run:
        generate = lambda cs: ["```sql\n" + loaded[b][1]["sql"] + "\n```" for b in chats]
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(R / a.model), gpu_memory_utilization=0.92, max_model_len=49152, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8",
                  kv_cache_dtype="fp8", max_num_seqs=64, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
        sp = SamplingParams(n=1, temperature=0, max_tokens=a.budget, seed=0)
        generate = lambda cs: [o.outputs[0].text for o in llm.chat(cs, sp, chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)]
    t0 = time.time()
    replies = dict(zip(chats, generate(list(chats.values())))) if chats else {}
    fails = {}
    for b, reply in replies.items():
        db, v = loaded[b]
        sql = select_of(reply)
        err = llmviews.qty_check(db, v["sql"], sql) if sql else "no SELECT in the reply"
        if err:
            fails[b] = (reply, err)
        else:
            v["sql_before_qty"], v["sql"] = v["sql"], sql
    for turn in range(a.turns):  # repair turns with the failed check
        names = list(fails)
        if not names:
            break
        rch = [chats[b] + [{"role": "assistant", "content": fails[b][0].split("</think>")[-1].strip()}, {"role": "user", "content": f"That query fails a check: {fails[b][1]}\nFix it and answer with the whole SELECT in one ```sql``` block."}] for b in names]
        for b, reply in zip(names, generate(rch)):
            db, v = loaded[b]
            sql = select_of(reply)
            err = llmviews.qty_check(db, v["sql"], sql) if sql else "no SELECT in the reply"
            if err:
                fails[b] = (reply, err)
                if turn == a.turns - 1:
                    print(f"[qty] {b}: kept the old view: {err[:200]}", flush=True)
            else:
                v["sql_before_qty"], v["sql"] = v["sql"], sql
                fails.pop(b)
    for b in shops:
        db, v = loaded[b]
        if v.get("sql_before_qty"):
            print(f"[qty] {b}: {llmviews.qty_report(db, v['sql_before_qty'], v['sql'])}", flush=True)
        elif v.get("sql") and b not in fails:
            print(f"[qty] {b}: unchanged", flush=True)
        (dst / f"_views_{b}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        if (src / f"_debt_{b}.json").exists() and src != dst:
            shutil.copy(src / f"_debt_{b}.json", dst / f"_debt_{b}.json")
        db.close()
    print(f"qty turn on {len(chats)} shops in {time.time() - t0:.0f}s, {len(fails)} kept the old view", flush=True)


if __name__ == "__main__":
    main()
