# Validation vs public leaderboard

Validation = `splits/val300.txt` (10 train questions per shop, 30 shops, seed 0), scored by
`python -m datamate.evaluate runs/<run> --ids splits/val300.txt`.
Public = the score the portal shows for `runs/<run>_public/submission.jsonl`.

Train has none of the four public-test industries (210 / 1000 public questions), so expect public < val.

| run | date | change | val PA | val precision | val abstain | public PA | gap |
|---|---|---|---|---|---|---|---|
| v0 | 2026-10-28 | raw tables, greedy, 1 repair | −0.075 (80 q) | 0.44 | 28/80 | — | — |
| v1 | 2026-10-28 | helpers + profiled columns + interpretation + values, k=5 T=0.7 τ=0.6 | +0.0625 (80 q) | 0.65 | 63/80 | — | — |

The 80 questions are the first 8 shops of val300 (biz_101–108), the only ones v0 and v1 both ran.

v1 at τ=0 (always answer the top cluster): PA −0.475, precision 0.24, worse than v0's 0.44. The whole
v0→v1 gain is the abstain gate; the v1 prompt made answers worse. Causes found in the logs:
13 top answers "unanswerable" (11 wrong) because the interpretation, written before the code saw any
data, declared existing products missing; hand-rolled parsers instead of the helpers; 17/250 replies
cut by the 2048-token limit; percent questions typed as number (5/13). v2 targets these.
