"""Do plain samples gain from a few lines of reasoning before the SQL? On the hard questions of a holdout run (plain agreement under .75), 8 first-turn samples per question, no repairs:
lora_plain as at inference; lora_notes the same LoRA with the reply forced to start with notes; base_notes base weights with the same start. Train labels only.
Pre-set rule, written to the flag file: use notes in LoRA v3 if lora_notes beats lora_plain by at least 3 points of right samples and its vote score is not lower.
Second test, on debt questions of shops with a debt layer: the few-shot bank with examples that call the debt macros against the old bank (examples from other shops only). Use it if right samples do not fall and macro use rises.
python runs/analysis/notes_test.py [--run RUN] [--views DIR] [--n 170] [--dry-run]; --dry-run scores saved samples with the notes text added, on CPU."""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import debtview
import fewshot
import llmviews
import predict as P
from datamate.scoring import score_one
from ingest import load_business, norm

def main():
    PREFIX = "Notes:\n-"
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sql_v5_holdout632")
    ap.add_argument("--views", default="runs/views_qty_holdout")
    ap.add_argument("--n", type=int, default=170)
    ap.add_argument("--model", default="sqlbase/models/Qwen3.5-9B")
    ap.add_argument("--lora", default="runs/lora/sql_rft_v2/adapter")
    ap.add_argument("--flag", default="runs/analysis/v3_use_notes")
    ap.add_argument("--bank-flag", default="runs/analysis/v3_use_bank")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
    rulebook = (R / "data/train/rulebook.md").read_text(encoding="utf-8-sig")
    system = P.SYSTEM + "\n\nRULEBOOK:\n" + rulebook
    bank = fewshot.Bank(R / "sqlbase/fewshot_bank24.jsonl")
    hard, debt_q = [], []
    for line in open(R / a.run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if "DEBTS (" in (t.get("prompt_user") or "") and re.search(r"(^|_)no(_|$)|cong_no|mua_chiu|ghi_no", norm(t["question"])):
            debt_q.append(t)
        if t.get("prompt_user") and not t["final"].get("reason") and P.vote([s["pred"] for s in t["samples"][:8]], 8)[1] < 0.75:
            hard.append(t)
    random.Random(0).shuffle(hard)
    hard = hard[:a.n]
    by = defaultdict(list)
    for t in hard:
        by[t["business_id"]].append(t)
    print(f"{len(hard)} hard questions in {len(by)} shops", flush=True)
    if a.dry_run:
        generate = None
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(model=str(R / a.model), gpu_memory_utilization=0.92, max_model_len=32768, enable_prefix_caching=True, seed=0, language_model_only=True, quantization="fp8", kv_cache_dtype="fp8",
                  enable_lora=True, max_lora_rank=64, max_loras=1, max_num_seqs=128, max_num_batched_tokens=8192, attention_backend="TRITON_ATTN")
        sp = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=1024, seed=0)
        lora = LoRARequest("sql", 1, str(R / a.lora))
        def generate(chats, use_lora, notes):
            kw = {"chat_template_kwargs": {"enable_thinking": False}, "use_tqdm": False}
            if use_lora:
                kw["lora_request"] = lora
            if notes:
                chats = [c + [{"role": "assistant", "content": PREFIX}] for c in chats]
                kw.update(add_generation_prompt=False, continue_final_message=True)
            return llm.chat(chats, sp, **kw)
    CONDS = {"lora_plain": (True, False), "lora_notes": (True, True), "base_notes": (False, True)}
    stats = {c: Counter() for c in CONDS}
    hold = Counter()
    for b, ts in sorted(by.items()):
        db, meta = load_business(R / "data/train", b, canon=True)
        db.execute("SET memory_limit = '6GB'")
        v = json.loads((R / a.views / "logs" / f"_views_{b}.json").read_text(encoding="utf-8"))
        if v.get("sql"):
            llmviews.build(db, v["sql"])
        df = R / a.views / "logs" / f"_debt_{b}.json"
        d = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
        if d.get("sqls") and d.get("used"):
            debtview.build(db, d["sqls"])
        chats = []
        for t in ts:
            user = t["prompt_user"]
            shots = fewshot.block(bank.retrieve(t["question"], b, 4, debt="DEBTS (" in user)) if "CANONICAL VIEWS" in user else ""
            chats.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
            chats.append([{"role": "system", "content": system}, {"role": "user", "content": user.replace("\n\nQUESTION: ", "\n\n" + shots + "\n\nQUESTION: ", 1) if shots else user}])
        for t in ts:
            th = t["samples"][8:]
            hold["thinking samples"] += len(th)
            hold["thinking right"] += sum(bool(s.get("pred")) and score_one({"id": t["id"], **s["pred"]}, gold[t["id"]]) == 1 for s in th)
        for c, (use_lora, notes) in CONDS.items():
            if a.dry_run:
                texts = [[(PREFIX + " x\n" if notes else "") + s["text"].split("\n#REPAIR\n")[0] for s in t["samples"][:8]] for t in ts]
                cut = [[False] * 8 for _ in ts]
            else:
                outs = generate(chats, use_lora, notes)
                texts, cut = [], []
                for i in range(len(ts)):
                    got = list(outs[2 * i].outputs) + list(outs[2 * i + 1].outputs)
                    texts.append([(PREFIX if notes else "") + o.text for o in got])
                    cut.append([o.finish_reason == "length" for o in got])
            s = stats[c]
            for t, tx, cu in zip(ts, texts, cut):
                preds = []
                for x, k in zip(tx, cu):
                    try:
                        p = P.execute(db, x, t["question"])
                    except Exception:
                        p = None
                    preds.append(p)
                    r = score_one({"id": t["id"], **p}, gold[t["id"]]) if p else None
                    s["samples"] += 1
                    s["right"] += r == 1
                    s["wrong"] += r == -1
                    s["no answer"] += p is None
                    s["cut off"] += k
                    s["chars"] += len(x)
                ans, conf = P.vote(preds, len(preds))
                final = ans if P.decide(ans, conf, 0.5) else {"type": "abstain"}
                sc = 0 if final.get("type") == "abstain" else score_one({"id": t["id"], **final}, gold[t["id"]])
                s["questions"] += 1
                s["vote right"] += sc == 1
                s["vote wrong"] += sc == -1
                s["agreement"] += conf
        db.close()
        print(f"[{b}] {len(ts)} q", flush=True)
    res = {}
    for c, s in stats.items():
        q, n = max(s["questions"], 1), max(s["samples"], 1)
        res[c] = {"right": s["right"] / n, "wrong": s["wrong"] / n, "none": s["no answer"] / n, "cut": s["cut off"] / n, "vote": (s["vote right"] - s["vote wrong"]) / q, "agree": s["agreement"] / q, "chars": s["chars"] / n}
        print(f"[notes] {c}: right {100*res[c]['right']:.1f}%, wrong {100*res[c]['wrong']:.1f}%, no answer {100*res[c]['none']:.1f}%, cut off {100*res[c]['cut']:.1f}% | vote {s['vote right']}/{s['vote wrong']}/{q - s['vote right'] - s['vote wrong']} = {res[c]['vote']:.3f}, agreement {res[c]['agree']:.3f}, {res[c]['chars']:.0f} chars", flush=True)
    print(f"[notes] reference, thinking samples in {a.run}: right {100*hold['thinking right']/max(hold['thinking samples'],1):.1f}% of {hold['thinking samples']}", flush=True)
    use = res["lora_notes"]["right"] - res["lora_plain"]["right"] >= 0.03 and res["lora_notes"]["vote"] >= res["lora_plain"]["vote"]
    print(f"[notes] decision by the pre-set rule: {'use notes in LoRA v3' if use else 'keep plain rows as they are'}", flush=True)
    if not a.dry_run:
        flag = R / a.flag
        if use:
            flag.write_text("notes\n")
        elif flag.exists():
            flag.unlink()
        DEBT = re.compile(r"debt_(at|new|repaid)\s*\(")
        banks = {"old bank": bank, "macro bank": fewshot.Bank(R / "sqlbase/fewshot_bank24m.jsonl")}
        sb = {c: Counter() for c in banks}
        by = defaultdict(list)
        for t in debt_q:
            by[t["business_id"]].append(t)
        for b, ts in sorted(by.items()):
            db, meta = load_business(R / "data/train", b, canon=True)
            db.execute("SET memory_limit = '6GB'")
            v = json.loads((R / a.views / "logs" / f"_views_{b}.json").read_text(encoding="utf-8"))
            if v.get("sql"):
                llmviews.build(db, v["sql"])
            debtview.build(db, json.loads((R / a.views / "logs" / f"_debt_{b}.json").read_text(encoding="utf-8"))["sqls"])
            for c, bk in banks.items():
                chats = []
                for t in ts:
                    user = t["prompt_user"]
                    shots = fewshot.block(bk.retrieve(t["question"], b, 4, debt=True))
                    chats.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
                    chats.append([{"role": "system", "content": system}, {"role": "user", "content": user.replace("\n\nQUESTION: ", "\n\n" + shots + "\n\nQUESTION: ", 1)}])
                outs = generate(chats, True, False)
                for i, t in enumerate(ts):
                    preds = []
                    for o in list(outs[2 * i].outputs) + list(outs[2 * i + 1].outputs):
                        try:
                            p = P.execute(db, o.text, t["question"])
                        except Exception:
                            p = None
                        preds.append(p)
                        sb[c]["samples"] += 1
                        sb[c]["right"] += bool(p) and score_one({"id": t["id"], **p}, gold[t["id"]]) == 1
                        sb[c]["macro"] += bool(DEBT.search(o.text))
                    ans, conf = P.vote(preds, len(preds))
                    fin = ans if P.decide(ans, conf, 0.5) else {"type": "abstain"}
                    sc = 0 if fin.get("type") == "abstain" else score_one({"id": t["id"], **fin}, gold[t["id"]])
                    sb[c]["vote"] += sc
                    sb[c]["q"] += 1
            db.close()
        for c, x in sb.items():
            print(f"[notes] {c}, {x['q']} debt questions: right {100*x['right']/max(x['samples'],1):.1f}%, calls a macro {100*x['macro']/max(x['samples'],1):.1f}%, vote score {x['vote']/max(x['q'],1):.3f}", flush=True)
        o, n = sb["old bank"], sb["macro bank"]
        use_bank = n["right"] >= o["right"] and n["macro"] > o["macro"]
        print(f"[notes] decision by the pre-set rule: {'use the macro bank' if use_bank else 'keep the old bank'}", flush=True)
        bf = R / a.bank_flag
        if use_bank:
            bf.write_text("macro bank\n")
        elif bf.exists():
            bf.unlink()


if __name__ == "__main__":
    main()
