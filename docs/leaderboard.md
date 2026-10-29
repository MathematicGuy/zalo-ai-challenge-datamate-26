# Validation vs public leaderboard

Validation = `splits/val300.txt` (10 train questions per shop, 30 shops, seed 0), scored in
`notebooks/03_evaluate.ipynb` (§1 scores, §2 τ sweep). Public = the score the portal shows for
`runs/<run>_public/submission.jsonl`. Train has none of the four public-test industries
(210 / 1000 public questions), so expect public < val.

| run | date | change | questions | val PA | precision | τ=0 precision | public PA |
|---|---|---|---|---|---|---|---|
| v0 | 2026-10-28 | raw tables, greedy, 1 repair | first 80 | −0.075 | 0.44 | 0.44 | — |
| v1 | 2026-10-28 | helpers + profiled columns + interpretation + values, k=5 T=0.7 τ=0.6 | first 80 | +0.0625 | 0.65 | 0.24 | — |
| v2 | 2026-10-28 | v1 + answerability in code, shorter replies, abbreviations, % guard | first 30 | −0.233 | 0.11 | 0.13 | — |
| v3 | — | v2 + normalized tables (`__date` / `__num`, fill-down, `ALL` views), `parse_*` names | not run yet | | | | |

Same 30 questions (biz_101–103) for all three: v0 −0.067 (11 correct / 13 wrong), v1 −0.100, v2 −0.233.

Reading the rows:
- The v0→v1 gain is the abstain gate only; τ=0 precision fell from 0.44 to 0.24.
- v1 logs: interpretation-before-code declared existing data missing (11 wrong "unanswerable"),
  hand-rolled parsers, 17/250 replies cut at 2048 tokens, percent typed as number.
- v2 logs: the model still parsed raw cells itself (`pd.to_datetime` on '1 thg 6 2025' → NaT → answer 0),
  and mixed our `to_datetime` up with pandas'. Prompting alone did not fix it; v3 moves parsing into
  the data layer, verified against 71 train answers without a GPU.
- v2 ran 3 of 30 shops before the dev machine rebooted (NVIDIA driver lost on a kernel upgrade).
