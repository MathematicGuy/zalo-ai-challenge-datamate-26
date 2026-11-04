#!/usr/bin/env bash
# Base-unit qty: one model turn per shop on the saved views (holdout to check it matches the hand patch, then public), saved SQL re-run on them, submission files.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "qty: repair turn on the holdout views"
python3 runs/analysis/fix_qty.py --data data/train --views-from runs/sql_v2_holdout632 --out runs/views_qty_holdout > runs/views_qty_holdout.log 2>&1
say "qty holdout exit $?"
grep -aE "^\[qty\]|^qty turn" runs/views_qty_holdout.log | cut -c1-150 | tee -a $LOG
python3 runs/analysis/reexec_views.py runs/sql_v2_holdout632 runs/views_qty_holdout runs/sql_v2_qtyauto632 --data data/train 2>&1 | grep -v Warning | tail -n 1 | cut -c1-110 | tee -a $LOG
python3 runs/analysis/trace_compare.py runs/sql_v2_holdout632 runs/sql_v2_qtyfix632 runs/sql_v2_qtyauto632 2>&1 | grep -v Warning | cut -c1-175 | tee -a $LOG
say "qty: repair turn on the public views"
python3 runs/analysis/fix_qty.py --data data/public_test --views-from sqlbase/config_v2_public --out runs/views_qty_public > runs/views_qty_public.log 2>&1
say "qty public exit $?"
grep -aE "^\[qty\]|^qty turn" runs/views_qty_public.log | cut -c1-150 | tee -a $LOG
python3 runs/analysis/reexec_views.py runs/sql_v2_public runs/views_qty_public runs/sql_v2_public_qty --data data/public_test 2>&1 | grep -v Warning | tail -n 1 | cut -c1-110 | tee -a $LOG
python3 runs/analysis/apply_review.py runs/sql_v2_public_qty none runs/sub_qty --data data/public_test 2>&1 | grep -v Warning | tee -a $LOG
python3 runs/analysis/apply_review.py runs/sql_v2_public_qty none runs/sub_qty_unans --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sub_qty/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 train/check_submission.py runs/sub_qty_unans/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
say "chain qty finished"
