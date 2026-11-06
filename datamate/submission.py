"""The submission file: written atomically during a run, checked before upload.

    python -m datamate.submission check RUN/submission.jsonl [--questions data/public_test/questions.jsonl]
    python -m datamate.submission merge RUN --questions DATA/questions.jsonl   (RUN/shard*/ -> RUN/submission.jsonl + RUN/logs/)

The check wants one line per known id, a valid type and an answer in the format that type requires (rulebook §9), and prints
a tally of what will be scored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def write_submission(path, qs, results):
    """Atomic rewrite: every question id, abstain when not answered yet."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps({"id": q["id"], **results.get(q["id"], {"type": "abstain"})}, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def problem(row: dict) -> str | None:
    """What is wrong with one submission line, or None."""
    t, a = row.get("type"), row.get("answer")
    if t not in TYPES:
        return f"unknown type {t!r}"
    if t in ("number", "percent") and not is_num(a):
        return f"{t} answer must be a JSON number, got {a!r}"
    if t == "string" and not (isinstance(a, str) and a.strip()):
        return "string answer must be a non-empty string"
    if t == "date" and not (isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a)):
        return f"date must be YYYY-MM-DD or YYYY-MM, got {a!r}"
    if t == "boolean" and not isinstance(a, bool):
        return "boolean answer must be true/false"
    if t == "list" and not (isinstance(a, list) and a and all(is_num(x) or isinstance(x, str) for x in a)):
        return "list answer must be a non-empty array of names or numbers"
    if t == "table":
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            return "table answer must be {columns, rows}"
        if any(not isinstance(r, list) or len(r) != len(a["columns"]) for r in a["rows"]):
            return "every table row needs one cell per column"
    if t == "unanswerable" and "reason" in row and not isinstance(row["reason"], str):
        return "reason must be a string (rulebook §7)"
    return None


def check(submission: Path, questions: Path) -> list[str]:
    """Print the tally of a submission file; return its errors (an empty list means it can be uploaded)."""
    ids = [json.loads(line)["id"] for line in questions.open(encoding="utf-8-sig") if line.strip()]
    known, seen, errors, types = set(ids), Counter(), [], Counter()
    for n, line in enumerate(submission.open(encoding="utf-8"), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not JSON ({e})")
            continue
        qid = row.get("id")
        seen[qid] += 1
        if qid not in known:
            errors.append(f"line {n}: unknown id {qid!r}")
        if (p := problem(row)):
            errors.append(f"line {n} ({qid}): {p}")
        types[row.get("type")] += 1
    dups = [q for q, c in seen.items() if c > 1]
    errors += [f"duplicated id {q}" for q in dups]
    missing = [q for q in ids if q not in seen]
    print(f"{sum(seen.values())} lines, {len(seen)} ids | missing {len(missing)} (scored as abstain) | duplicated {len(dups)}")
    print("types:", dict(types.most_common()))
    print(f"answered {sum(c for t, c in types.items() if t != 'abstain')} / {len(ids)} questions")
    return errors


def merge_shards(out: Path, questions: Path) -> int:
    """One submission from the shard dirs of a multi-GPU run (out/shard<i>/), every question id in order; logs copied to out/logs."""
    import shutil
    got = {}
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for s in sorted(out.glob("shard*/")):
        if (s / "submission.jsonl").exists():
            got.update((r["id"], r) for r in map(json.loads, (s / "submission.jsonl").read_text(encoding="utf-8").splitlines()))
        for f in (s / "logs").glob("*.json") if (s / "logs").exists() else []:
            shutil.copy(f, out / "logs" / f.name)
    ids = [json.loads(line)["id"] for line in questions.open(encoding="utf-8-sig") if line.strip()]
    with (out / "submission.jsonl").open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps(got.get(i, {"id": i, "type": "abstain"}), ensure_ascii=False) + "\n")
    print(f"merged {sum(i in got for i in ids)}/{len(ids)} questions -> {out / 'submission.jsonl'}")
    return sum(i in got for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "merge"])
    ap.add_argument("path", type=Path, help="check: the submission file; merge: the run dir holding shard*/")
    ap.add_argument("--questions", type=Path, default=Path("data/public_test/questions.jsonl"))
    args = ap.parse_args()
    if args.command == "merge":
        merge_shards(args.path, args.questions)
        return
    errors = check(args.path, args.questions)
    for e in errors[:20]:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("format OK")


if __name__ == "__main__":
    main()
