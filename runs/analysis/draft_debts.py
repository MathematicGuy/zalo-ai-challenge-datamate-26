"""Debt layers with thinking drafts and two repair turns (debtview.draft), on every train and public shop, one spelling per name. A new layer is adopted only where the current one is not in use, so what already works stays.
Writes runs/debts_think/{train,public}/all (every shop) and .../logs (the layers to adopt). python runs/analysis/draft_debts.py [--dry-run]; --dry-run replays saved layers through the checks on CPU."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import debtview
from ingest import load_business, schema_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", help="comma-separated shops")
    a = p.parse_args()
    if a.dry_run:
        class O:
            def __init__(s, text): s.text = text
        class Out:
            def __init__(s, texts): s.outputs = [O(t) for t in texts]
        saved = {}
        def chat(cs, sp, think):
            if think:  # 2 good drafts and one that needs a repair
                return [Out([saved["text"], saved["text"], "```sql\nSELECT nope\n```\n```sql\nSELECT 1\n```"]) for _ in cs]
            return [Out([saved["text"]]) for _ in cs]
        sampling = lambda **kw: kw
    else:
        from vllm import LLM, SamplingParams as sampling
        llm = LLM(model=str(R / "sqlbase/models/Qwen3.5-9B"), gpu_memory_utilization=0.92, max_model_len=32768, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8", kv_cache_dtype="fp8",
                  max_num_seqs=64, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
        chat = lambda cs, sp, think: llm.chat(cs, sp, chat_template_kwargs={"enable_thinking": think}, use_tqdm=False)
    for label, data, cur in [("train", "data/train", "runs/views_v3_train/logs"), ("public", "data/public_test", "runs/views_canon_public/logs")]:
        root = R / data
        rulebook = (root / "rulebook.md").read_text(encoding="utf-8-sig")
        names = sorted(x.name for x in (root / "data").iterdir() if x.is_dir() and (not a.only or x.name in a.only.split(",")))
        if not names:
            continue
        shops = {}
        for b in names:
            db, meta = load_business(root, b, canon=True)
            db.execute("SET enable_external_access = false")
            db.execute("SET memory_limit = '6GB'")
            shops[b] = db, meta
        if a.dry_run:
            old = json.loads((R / cur / f"_debt_{names[0]}.json").read_text(encoding="utf-8"))
            saved["text"] = "```sql\n" + old["sqls"][0] + "\n```\n```sql\n" + old["sqls"][1] + "\n```"
        res = debtview.draft(chat, sampling, shops, rulebook, schema_text, k=3 if a.dry_run else 8)
        (R / "runs/debts_think" / label / "all").mkdir(parents=True, exist_ok=True)
        (R / "runs/debts_think" / label / "logs").mkdir(parents=True, exist_ok=True)
        for b, (texts, checked) in res.items():
            sqls, n = debtview.pick(checked)
            fps = [str(fp) for fp, _ in checked]
            k = len(checked)
            ok = bool(sqls) and debtview.accept(fps, n, k)
            f = R / cur / f"_debt_{b}.json"
            old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            same = bool(old.get("used")) and any(x in fps for x in old.get("checked", []) if x not in ("None", "none"))
            new = {"agree": n, "k": k, "used": ok, "sqls": sqls, "checked": fps, "errors": [e for fp, e in checked if fp is None]}
            text = json.dumps(new, ensure_ascii=False, indent=1)
            (R / "runs/debts_think" / label / "all" / f"_debt_{b}.json").write_text(text, encoding="utf-8")
            adopt = ok and not old.get("used")
            if adopt and not a.dry_run:
                (R / "runs/debts_think" / label / "logs" / f"_debt_{b}.json").write_text(text, encoding="utf-8")
            votes = Counter("none" if x == "none" else "failed" if x == "None" else "layer" for x in fps)
            print(f"[debts] {label} {b}: new {n}/{k}, usable {ok}, {dict(votes)} | current agree {old.get('agree')}, used {old.get('used')}, same numbers {same}{' -> adopted' if adopt else ''}", flush=True)
            shops[b][0].close()
    print("debt drafts finished", flush=True)


if __name__ == "__main__":
    main()
