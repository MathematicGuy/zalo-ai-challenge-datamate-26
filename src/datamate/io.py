"""Read a DataMate split: questions, answers, business READMEs and raw tables."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"


def split_dir(split: str) -> Path:
    return DATA / split


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_json_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def load_questions(split: str) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(split_dir(split) / "questions.jsonl"))


def load_answers(split: str) -> pd.DataFrame | None:
    path = split_dir(split) / "answers.jsonl"
    return pd.DataFrame(read_jsonl(path)) if path.exists() else None


def business_ids(split: str) -> list[str]:
    return sorted(p.name for p in (split_dir(split) / "data").iterdir() if p.is_dir())


@dataclass
class Readme:
    text: str
    files: list[str]
    # "file" or "file, sheet X" -> {column name: role phrase}
    columns: dict[str, dict[str, str]] = field(default_factory=dict)
    conventions: list[str] = field(default_factory=list)
    last_date: str | None = None


_SECTION = re.compile(r"^## (.+)$", re.M)
_COL = re.compile(r"^- \*\*(.+?)\*\*: (.+)$")


def parse_readme(path: Path) -> Readme:
    text = path.read_text(encoding="utf-8")
    heads = list(_SECTION.finditer(text))
    files, columns, conventions = [], {}, []
    for i, h in enumerate(heads):
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        title = h.group(1).strip()
        if title == "File":
            files = re.findall(r"^- `([^`]+)`", body, re.M)
        elif title.startswith("Cột trong"):
            key = title.removeprefix("Cột trong").strip().replace("`", "")
            columns[key] = {m.group(1): m.group(2) for m in map(_COL.match, body.splitlines()) if m}
        elif title == "Quy ước":
            conventions = [ln[2:] for ln in body.splitlines() if ln.startswith("- ")]
    m = re.search(r"Dữ liệu có (?:từ [\d/]+ )?đến hết (\d{2}/\d{2}/\d{4})", text)
    return Readme(text, files, columns, conventions, m.group(1) if m else None)


@dataclass
class RawTable:
    file: str
    sheet: str | None  # xlsx sheet or sqlite table; None for csv
    df: pd.DataFrame   # header=None, every cell as read


def read_raw_tables(biz_dir: Path) -> list[RawTable]:
    """Every sheet/table of a business, without header inference, so EDA can see header rows."""
    out = []
    for p in sorted(biz_dir.iterdir()):
        suf = p.suffix.lower()
        if suf == ".csv":
            out.append(RawTable(p.name, None, pd.read_csv(p, header=None, dtype=str, encoding="utf-8-sig",
                                                          keep_default_na=False, na_values=[""])))
        elif suf == ".xlsx":
            for sheet, df in pd.read_excel(p, sheet_name=None, header=None).items():
                out.append(RawTable(p.name, sheet, df))
        elif suf == ".sqlite":
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
            for t in names:
                df = pd.read_sql_query(f'select * from "{t}"', con)
                out.append(RawTable(p.name, t, pd.concat([df.columns.to_frame().T, df], ignore_index=True)
                                    .set_axis(range(df.shape[1]), axis=1)))
            con.close()
    return out
