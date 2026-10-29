"""Run the agent over a split: python -m datamate.predict --data /data --out /result

Writes <out>/submission.jsonl (appended per business, so a timeout keeps finished answers) and one
<out>/logs/<id>.json per question with the generated code, execution output and timing (Rule 8).
A rerun into the same --out skips questions already in submission.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

from .agent import Agent, AgentConfig
from .io import read_jsonl, to_json_line
from .llm import ChatModel, VLLMChatModel
from .prompts import PROMPTS
from .tables import cache_path, load_business_cached


def run(data: Path, out: Path, cfg: AgentConfig, ids: set[str] | None = None, llm: ChatModel | None = None) -> None:
    questions = [q for q in read_jsonl(data / "questions.jsonl") if ids is None or q["id"] in ids]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    sub_path = out / "submission.jsonl"
    done = {json.loads(line)["id"] for line in sub_path.open(encoding="utf-8")} if sub_path.exists() else set()
    by_biz: dict[str, list[dict]] = {}
    for q in questions:
        if q["id"] not in done:
            by_biz.setdefault(q["business_id"], []).append(q)
    if not by_biz:
        return

    agent = Agent(llm or VLLMChatModel(), cfg, (data / "rulebook.md").read_text(encoding="utf-8"),
                  data / "calendar.csv")
    t_all = time.time()
    for bi, (biz_id, qs) in enumerate(sorted(by_biz.items())):
        biz = load_business_cached(data, biz_id)
        br = agent.answer_business(biz, cache_path(data, biz_id), qs)
        with sub_path.open("a", encoding="utf-8") as f:
            for r in br.results:
                f.write(to_json_line(r.row) + "\n")
                (out / "logs" / f"{r.row['id']}.json").write_text(json.dumps(r.log, ensure_ascii=False, indent=1),
                                                                  encoding="utf-8")
        answered = sum(r.row["type"] != "abstain" for r in br.results)
        print(f"[{bi + 1}/{len(by_biz)}] {biz_id}: {len(qs)} q, {answered} answered, llm {br.llm_seconds:.0f}s, "
              f"total {br.total_seconds:.0f}s | elapsed {time.time() - t_all:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/data"))
    ap.add_argument("--out", type=Path, default=Path("/result"))
    ap.add_argument("--ids", type=Path, help="file with one question id per line (validation subsets)")
    d = AgentConfig()
    ap.add_argument("--prompt", default=d.prompt, choices=sorted(PROMPTS))
    ap.add_argument("--k", type=int, default=d.k)
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--tau", type=float, default=d.tau)
    ap.add_argument("--max-repairs", type=int, default=d.max_repairs)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--rules-k", type=int, default=d.rules_k)
    ap.add_argument("--rule-retriever", default=d.rule_retriever, help="lexical | dense[:model] | hybrid")
    a = ap.parse_args()
    cfg = AgentConfig(**{f.name: getattr(a, f.name) for f in fields(AgentConfig)})
    run(a.data, a.out, cfg, set(a.ids.read_text().split()) if a.ids else None)


if __name__ == "__main__":
    main()
