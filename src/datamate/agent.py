"""The agent: for one business, prompt -> sample k programs -> execute -> repair -> vote -> answer or abstain.

It depends on a ChatModel (llm.py), so tests drive it with a fake model and no GPU. Writing the
submission and logs is predict.py's job.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .answers import extract_code, normalize_answer, postprocess
from .context import describe_business
from .llm import ChatModel
from .prompts import PROMPTS, REPAIR
from .retrieval import ValueIndex, format_rules, format_values, make_rule_retriever, top_rules, value_tokens
from .sandbox import ExecResult, Executor
from .select import decide, vote
from .tables import Business


@dataclass
class AgentConfig:
    prompt: str = "v2"         # key of prompts.PROMPTS
    k: int = 5                 # samples per question; k=1 decodes greedily
    temperature: float = 0.7
    tau: float = 0.6           # minimum share of the k samples agreeing on the answer, else abstain
    max_repairs: int = 1       # repair turns, only for questions where every sample failed
    thinking: bool = False
    rules_k: int = 0           # rulebook rows repeated next to the question; 0 = off (see retrieval/rules.py)
    rule_retriever: str = "lexical"


@dataclass
class QuestionResult:
    row: dict                  # submission line
    log: dict                  # per-question log (Rule 8): prompts, every sample's code/stdout/answer, vote


@dataclass
class BusinessRun:
    results: list[QuestionResult] = field(default_factory=list)
    llm_seconds: float = 0.0
    total_seconds: float = 0.0


def _attempt(reply: str) -> tuple[str | None, dict]:
    code = extract_code(reply)
    log = {"reply": reply, "code": code}
    if code is None:
        log["error"] = "No python code block in the reply."
    return code, log


def _settle(r: ExecResult, log: dict) -> dict | None:
    """One execution result -> normalized answer or None; fills the sample log."""
    log.update(stdout=r.stdout, error=r.error, exec_seconds=round(r.elapsed, 3))
    if not r.ok:
        return None
    try:
        log["answer"] = normalize_answer(r.answer)  # kept per sample so votes can be recomputed offline
        return log["answer"]
    except ValueError as e:
        log["error"] = f"Invalid ANSWER {r.answer!r}: {e}"
        return None


def execute_replies(ex: Executor, items: list[tuple[str, str]]) -> list[tuple[dict | None, dict]]:
    """items: (reply text, as_of). Runs every extracted program in parallel; results in input order."""
    attempts = [(_attempt(reply), as_of) for reply, as_of in items]
    results = iter(ex.run_many([(code, as_of) for (code, _), as_of in attempts if code is not None]))
    return [(_settle(next(results), log) if code is not None else None, log) for (code, log), _ in attempts]


class Agent:
    def __init__(self, llm: ChatModel, cfg: AgentConfig, rulebook: str, calendar_csv: Path):
        self.llm, self.cfg, self.rulebook, self.calendar_csv = llm, cfg, rulebook, calendar_csv
        self.spec = PROMPTS[cfg.prompt]
        self.rule_retriever = make_rule_retriever(cfg.rule_retriever, rulebook) if cfg.rules_k else None

    # ------------------------------------------------------------------ prompts

    def _user_turn(self, q: dict, values: ValueIndex | None) -> str:
        if values is None:
            return self.spec.build_user(q)
        hits = values.match(q["question"])
        rules = (top_rules(self.rule_retriever, q["question"], self.cfg.rules_k, value_tokens(hits))
                 if self.rule_retriever else [])
        return self.spec.build_user(q, rules=format_rules(rules), values=format_values(hits))

    def _chat(self, convs: list[list[dict]], n: int, temperature: float, run: BusinessRun) -> list[list[str]]:
        t = time.time()
        out = self.llm.chat(convs, n=n, temperature=temperature, thinking=self.cfg.thinking)
        run.llm_seconds += time.time() - t
        return out

    # ------------------------------------------------------------------ pipeline

    def answer_business(self, biz: Business, biz_pickle: Path, questions: list[dict]) -> BusinessRun:
        cfg, run, t0 = self.cfg, BusinessRun(), time.time()
        system = self.spec.build_system(self.rulebook, describe_business(biz))
        values = ValueIndex(biz) if self.spec.uses_context else None
        users = [self._user_turn(q, values) for q in questions]
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": u}] for u in users]
        logs = [{"id": q["id"], "business_id": biz.biz_id, "question": q["question"], "as_of_date": q["as_of_date"],
                 "user_prompt": u, "samples": []} for q, u in zip(questions, users)]
        answers: list[list[dict | None]] = [[] for _ in questions]

        replies = self._chat(convs, cfg.k, cfg.temperature if cfg.k > 1 else 0.0, run)
        with Executor(biz_pickle, self.calendar_csv) as ex:
            owners = [i for i, texts in enumerate(replies) for _ in texts]
            items = [(text, questions[i]["as_of_date"]) for i, texts in enumerate(replies) for text in texts]
            for i, (ans, slog) in zip(owners, execute_replies(ex, items)):
                answers[i].append(postprocess(questions[i]["question"], ans))
                logs[i]["samples"].append(slog)
            self._repair(ex, convs, questions, answers, logs, run)

        run.total_seconds = time.time() - t0
        per_q = run.total_seconds / max(len(questions), 1)
        for q, ans, lg in zip(questions, answers, logs):
            voted = vote(ans, cfg.k)
            row = {"id": q["id"], **decide(voted, cfg.tau)}
            lg.update(final=row, support=voted["support"], clusters=voted["clusters"], elapsed_seconds=round(per_q, 3))
            run.results.append(QuestionResult(row, lg))
        return run

    def _repair(self, ex: Executor, convs, questions, answers, logs, run: BusinessRun) -> None:
        """Repair only questions where no sample ran: diverse first samples beat repair turns at equal
        cost (Olausson et al. 2024). A repaired answer carries the support of a single sample."""
        for _ in range(self.cfg.max_repairs):
            retry = [i for i, a in enumerate(answers) if all(x is None for x in a)]
            if not retry:
                return
            rconvs = []
            for i in retry:
                first = logs[i]["samples"][0]
                rconvs.append(convs[i] + [
                    {"role": "assistant", "content": first["reply"]},
                    {"role": "user", "content": REPAIR.format(problem=first.get("error"),
                                                              stdout=(first.get("stdout") or "")[-1500:])}])
            fixed = self._chat(rconvs, 1, 0.0, run)
            items = [(texts[0], questions[i]["as_of_date"]) for i, texts in zip(retry, fixed)]
            for i, (ans, slog) in zip(retry, execute_replies(ex, items)):
                slog["repair"] = True
                logs[i]["samples"].append(slog)
                if ans is not None:
                    answers[i] = [postprocess(questions[i]["question"], ans)] + [None] * (self.cfg.k - 1)
