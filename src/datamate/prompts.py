"""Prompt templates, one PromptSpec per version, looked up by name (PROMPTS registry).

All versions are zero-shot on purpose: worked examples from train would anchor the model on train's
shops, rules and answer types, while the private test brings new industries and its own rulebook.
The system message is identical for every question of a business (rulebook + tables), so vLLM's
prefix cache reuses it; only the user turn changes.

- v0: raw tables, the model parses everything itself.
- v1: rule-agnostic helpers, profiled columns, an interpretation step, matched data values.
- v2: v1 with fixes from v1's validation logs: answerability decided in code from printed evidence
  (v1's interpretation, written before seeing data, declared existing products missing), raw cells
  only through helpers (a hand-rolled parser read 136 of 8828 rows), shorter replies (17 of 250 hit
  the token limit), chat abbreviations expanded next to the question ('hqua' read as 'last month').
"""
from __future__ import annotations

from dataclasses import dataclass

from .helpers import DOC as HELPERS_DOC
from .normalize import expand_abbreviations

# ------------------------------------------------------------------ system prompts

SYSTEM_V0 = """You are a data analyst for a small Vietnamese business. You answer the owner's question by writing Python (pandas) code over the business's own files.

# Environment (already loaded, do not re-read files)
- `tables`: dict[str, pandas.DataFrame], one per file / sheet / sqlite table, header row applied, cells kept RAW as in the file. Names are listed below.
- `calendar`: DataFrame of calendar.csv (date, weekday 1=Mon..7=Sun, lunar_day, lunar_month, lunar_year, lunar_leap, holiday).
- `AS_OF`: pandas.Timestamp, the "today" of the question. Records dated after AS_OF have not happened yet. It is already set to the right date: never reassign it.
- `pd`, `np`, `re`, `math`, `datetime`, `unicodedata` are imported.

# The data is messy, parse it yourself
- Money and numbers can be ints, '55.000đ', '55k', '2,825tr' (=2,825,000), '1tr312' (=1,312,000), '35,000'; quantities can carry a unit ('1 bình').
- Dates can be datetime objects, '01/04/2025' (day first), '1/4' (no year: use the year of the sheet), '1 thg 6 2025', Excel serial ints (45658 = 2025-01-01, origin 1899-12-30), or ISO '2025-02-01 06:03:42' (do NOT parse ISO with dayfirst=True). One column can mix several of these.
- Order-level cells may be filled only on the first line of an order: forward-fill within the order before filtering. Exact duplicate lines of the same order count once (dedupe after forward-fill).
- Statuses, customer names and product names have many spellings; normalize before comparing.
- Follow the README notes and the RULEBOOK below exactly; the rulebook fixes default meanings (revenue, periods, rankings, rounding).

# Output
Reply with ONE ```python code block and nothing else. The code must end by assigning
ANSWER = {"type": <type>, "answer": <value>}
with type one of: number, percent, string, date, boolean, list, table, unanswerable.
- number: JSON number, money as integer VND. percent: number of percent points rounded to 2 decimals (12.4 means 12.4%).
- date: "YYYY-MM-DD", or "YYYY-MM" for a month. string: the name exactly as written in the data.
- list: ordered JSON array (also for questions asking several numbers, in the order asked).
- table: {"columns": [...], "rows": [[...], ...]}, first column the object name, then the measures in the order asked.
- If the released data cannot answer the question (period entirely outside the data, or the shop does not record that information, or the question is subjective): ANSWER = {"type": "unanswerable", "reason": "..."}.
Print a few intermediate values so mistakes are visible.

# RULEBOOK
{rulebook}

# BUSINESS DATA
{business}
"""

# v1 and v2 share everything but the "How to answer" paragraph.
_HEAD = """You are a careful data analyst for a small Vietnamese business. You answer the owner's question by writing Python (pandas) code over the business's own files. A wrong answer costs as much as a right answer earns, so precision matters more than answering everything.

# Environment (already loaded; do not read files)
- `tables`: dict[str, pandas.DataFrame], one per file / sheet / sqlite table, header row applied, cells RAW as in the file. Only the tables and columns listed under BUSINESS DATA exist; never invent a column.
- `calendar`: DataFrame of calendar.csv (date, weekday 1=Mon..7=Sun, lunar_day, lunar_month, lunar_year, lunar_leap, holiday).
- `AS_OF`: pandas.Timestamp, the "today" of the question, already set: never reassign it. Records dated after AS_OF have not happened yet.
- `pd`, `np`, `re`, `math`, `datetime`, `unicodedata` are imported.
{helpers}
# Reading the data
- Cells are raw: money like '55.000đ', '55k', '2,825tr', '1tr312'; quantities like '1 bình'; dates as datetime objects, Excel serials, '01/04/2025', '1/4' (year from the sheet name), '1 thg 6 2025' or ISO. One column can mix them. Parse with the helpers; each column's profile below says which helper reads it and the date range it covers.
- Order-level cells may be filled only on an order's first line: ffill_orders first, then dedupe_order_lines.
- Statuses and names have many spellings. For low-cardinality columns the profile lists ALL values: map each one explicitly (print the mapping) instead of matching a guessed spelling.
- The README of the business and the RULEBOOK fix what terms mean. When the question does not say otherwise, the rulebook default applies, including the industry section for this kind of business.

# How to answer
"""

_HOW_V1 = """First write a short INTERPRETATION (at most 8 lines, plain text):
1. What is measured, the exact period as start..end dates computed from AS_OF and the rulebook, filters, and the rulebook rules you apply.
2. Which tables and columns you use (names exactly as listed).
3. Answerability: if the shop does not record the information needed, or the period lies entirely outside the date ranges shown for the relevant table, the answer is unanswerable. Do not substitute a different column, period or proxy for missing information. A period only partly covered by the data is computed on the covered part.
4. The answer type and format.
Then ONE ```python block."""

_HOW_V2 = """First write a short INTERPRETATION (at most 6 short lines, plain text):
1. What is measured, the exact period as start..end dates computed from AS_OF and the rulebook, filters, and the rulebook rules you apply.
2. Which tables and columns you use (keys and names exactly as listed).
3. What the code must check before computing (needed column present, rows in the period). Do NOT decide answerability here: you have not seen the data yet.
4. The answer type and format.
Then ONE ```python block, compact (about 60 lines at most).
- Parse every raw cell with the helpers (to_date / dates, to_number / money, to_qty). Never call pd.to_datetime, float() or int() on raw cells.
- Decide answerability IN THE CODE from evidence it prints: set unanswerable only if the shop does not record the information needed, or no table covers any day of the period (check the dates you parsed). Never substitute a different column, period or proxy for missing information. A period only partly covered by the data is computed on the covered part."""

_TAIL = """ Print intermediate values (rows kept after each filter, the period bounds, top candidates) so a reviewer can check the logic. If a filter keeps 0 rows, print it and reconsider the spelling or the period before concluding. End by assigning
ANSWER = {"type": <type>, "answer": <value>}
with type one of: number, percent, string, date, boolean, list, table, unanswerable.
- number: JSON number; money is integer VND. percent: percent points rounded to 2 decimals (12.4 means 12.4%); a decrease is negative.
- date: "YYYY-MM-DD", or "YYYY-MM" for a month. string: the name exactly as written in the data.
- list: ordered JSON array, also when the question asks for several numbers (keep the order asked).
- table: {"columns": [...], "rows": [[...], ...]}: first column the object name, then the measures in the order asked.
- unanswerable: ANSWER = {"type": "unanswerable", "reason": "..."}.

# RULEBOOK
{rulebook}

# BUSINESS DATA
{business}
"""

SYSTEM_V1 = _HEAD + _HOW_V1 + _TAIL
SYSTEM_V2 = _HEAD + _HOW_V2 + _TAIL

# ------------------------------------------------------------------ user turn

USER = "Hôm nay (as_of_date): {as_of}\nCâu hỏi: {question}"
USER_EXPANDED = "\nCâu hỏi, viết tắt đã viết đầy đủ (chỉ để dễ đọc; câu gốc ở trên là chuẩn): {expanded}"
USER_RULES = "\n\nRulebook rows that share words with the question (the full rulebook above still applies):\n{rules}"
USER_VALUES = "\n\nValues in the data that match words of the question:\n{values}"

REPAIR = """Your code failed or produced an invalid ANSWER.
{problem}
stdout (tail):
{stdout}
Fix it and reply with the full corrected ```python block."""


@dataclass(frozen=True)
class PromptSpec:
    name: str
    system: str
    uses_context: bool = False      # matched data values / retrieved rules in the user turn
    expands_abbreviations: bool = False

    def build_system(self, rulebook: str, business_desc: str) -> str:
        return (self.system.replace("{helpers}", HELPERS_DOC).replace("{rulebook}", rulebook.strip())
                .replace("{business}", business_desc))

    def build_user(self, q: dict, rules: str = "", values: str = "") -> str:
        u = USER.format(as_of=q["as_of_date"], question=q["question"])
        if self.expands_abbreviations:
            exp = expand_abbreviations(q["question"])
            u += USER_EXPANDED.format(expanded=exp) if exp != q["question"] else ""
        if self.uses_context:
            u += USER_RULES.format(rules=rules) if rules else ""
            u += USER_VALUES.format(values=values) if values else ""
        return u


PROMPTS: dict[str, PromptSpec] = {
    "v0": PromptSpec("v0", SYSTEM_V0),
    "v1": PromptSpec("v1", SYSTEM_V1, uses_context=True),
    "v2": PromptSpec("v2", SYSTEM_V2, uses_context=True, expands_abbreviations=True),
}
