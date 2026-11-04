# DataMate — Zalo AI Challenge 2026

Trả lời câu hỏi tiếng Việt trên dữ liệu bán hàng lộn xộn của từng shop, hoặc bỏ trống nếu không chắc. Chấm theo Penalized Accuracy: đúng +1, bỏ trống 0, sai −1.

Pipeline hiện tại nằm trong `sqlbase/`: Qwen3.5-9B kèm LoRA tự train viết nhiều câu SQL cho mỗi câu hỏi, chạy trên DuckDB rồi bỏ phiếu. Chi tiết ở [sqlbase/README.md](sqlbase/README.md).

Repo này chỉ dành cho thành viên trong đội. Quy chế §4.3 và §8.2 cấm chia sẻ code và dữ liệu ra ngoài đội.

## Chạy nhanh

Cần: GPU từ 24 GB (đã chạy trên 1 RTX PRO 4000 Blackwell 24 GB, RAM 30 GB), Python 3.12, `git-lfs`, khoảng 30 GB đĩa.

```bash
# 1. Lấy code và LoRA (LoRA nằm trong Git LFS)
git clone git@github.com:hugebenevolence/zalo-ai-challenge-datamate-26.git
cd zalo-ai-challenge-datamate-26
git lfs install && git lfs pull

# 2. Cài thư viện
python3 -m venv .venv && source .venv/bin/activate
pip install -r sqlbase/requirements.txt

# 3. Tải model gốc, khoảng 19 GB
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir sqlbase/models/Qwen3.5-9B

# 4. Đặt dữ liệu của BTC vào data/ (không có trong repo)
#    data/public_test/{questions.jsonl,rulebook.md,calendar.csv,data/<shop>/...}
#    data/train/{questions.jsonl,answers.jsonl,rulebook.md,calendar.csv,data/<shop>/...}

# 5. Chạy public, khoảng 2 giờ trên một GPU 24 GB
bash sqlbase/run_public.sh runs/my_run
# File để nộp: runs/my_run/submission.jsonl
```

GPU khác cấu hình thì chỉnh bằng biến môi trường, ví dụ `GPU_MEM=0.85 bash sqlbase/run_public.sh runs/my_run`. Hết bộ nhớ thì giảm `max_num_seqs` trong biến `ENGINE`.

### Docker

```bash
docker build -f sqlbase/Dockerfile -t datamate .
docker run --rm --gpus all --network none -v /duong/dan/split:/data:ro -v "$PWD/result:/result" datamate
```

Image chứa sẵn model và LoRA, chạy offline, tự soạn view cho từng shop và ghi `/result/submission.jsonl`. Trước khi build phải làm xong bước 1 và 3.

Ảnh dựng được, nặng khoảng 68,5 GB, nên cần còn trống khoảng 100 GB đĩa khi build. Code nạp được và container thấy GPU, nhưng ảnh chưa chạy trọn một lượt trên GPU. Mỗi lệnh `docker` có thể cần `sudo`.

## Trọng số

| Thành phần | Nguồn | Dung lượng |
|---|---|---|
| Qwen3.5-9B | Hugging Face, lệnh ở bước 3; không đưa vào repo vì quá lớn cho Git LFS | Khoảng 19 GB |
| LoRA v2 | `runs/lora/sql_rft_v2/adapter`, qua Git LFS | 185 MB |
| Bộ ví dụ few-shot | `sqlbase/fewshot_bank24.jsonl` | Trong repo |
| View và lớp công nợ đã lưu của public | `sqlbase/config_v4_public` | Trong repo |
| LoRA v3 | Đang train; khi có sẽ nằm ở `runs/lora/sql_rft_v3/adapter`, dùng bằng `LORA_DIR=runs/lora/sql_rft_v3/adapter` | |

## Điểm public

| Bản | Điểm |
|---|---|
| LoRA v2, few-shot, macro công nợ (`runs/analysis/run_v2_public.sh`) | 0.491 |
| Thêm `qty` theo đơn vị cơ bản | 0.506 |
| Thêm trả lời `unanswerable` theo đa số | 0.516 |
| Thêm chuẩn hóa tên và hệ số từ giá (`sqlbase/run_public.sh`) | Chưa có điểm |

Mỗi lần chạy sinh mẫu mới nên điểm có thể lệch nhẹ.

## Pipeline notebook cũ (Gemma/pandas, không còn dùng)

Public cao nhất 0.25. Giữ lại để tham khảo; phần dưới mô tả pipeline đó, không phải `sqlbase/`.

### Layout

```
notebooks/
  01_data.ipynb       data layer (read, header detection, normalization, table description) + EDA + data cards
  02_agent.ipynb      the agent: prompts v2/v3, retrieval, LLM, sandboxed execution, voting, submission
  03_evaluate.ipynb   scores, τ sweep, error reading, rule-retrieval study, normalization check
src/datamate/         the parts that must be importable files
  helpers.py          parsers injected into the generated code (money, dates, fill-down, lunar dates)
  sandbox.py          spawned worker that runs generated code with a timeout
  scoring.py          the PA metric from the Evaluation tab
  nblib.py            run_lib_cells: load a notebook's `lib` cells into another notebook
run.sh                run 02_agent headless on a split, one shard per GPU
predict.sh            Docker entrypoint: /data -> /result
docs/                 challenge spec, data notes, research → design decisions, results
splits/               validation ids; hand-made rule-relevance labels (train only)
tests/                pytest, no GPU (scripted model for the end-to-end test)
```

Cells tagged `lib` in 01 and 02 are the code; the other cells explore or run. 02 and 03 load the
`lib` cells of the notebooks before them, so editing a function in 01 changes it everywhere.

### Setup (server)

Needs CUDA ≥ 12.8 (Blackwell), Python 3.12.

```bash
git clone https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26 && cd zalo-ai-challenge-datamate-26
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[llm,dev]"          # add ,retrieval only for dense rule retrieval
hf download Qwen/Qwen3.5-9B             # ~19 GB, once
```

Put the released zips in `data/` and unzip them there: `data/train/`, `data/public_test/`
(`data/` is gitignored). The first run caches normalized tables in `data/cache/` (a few minutes).

### Run

```bash
pytest -q                                                    # CPU only, ~15 s: everything but the LLM

## 1. smoke test on the GPUs: 6 questions, 1 sample each
./run.sh train runs/smoke -p IDS ../splits/smoke6.txt -p K 1

## 2. validation: v3, and v2 on the same questions for comparison
./run.sh train runs/v3_val300 -p IDS ../splits/val300.txt
./run.sh train runs/v2_val300 -p IDS ../splits/val300.txt -p PROMPT v2

## 3. public test -> upload runs/v3_public/submission.jsonl
./run.sh public_test runs/v3_public
```

`run.sh` starts one process per visible GPU (`DATAMATE_NGPU` to override), each on a share of the
shops, and merges `submission.jsonl` + `logs/` when all finish. A rerun into the same folder resumes.
Progress is in `runs/<run>/shard<i>.txt`. Parameters go through papermill: `-p K 5 -p TAU 0.6
-p TEMPERATURE 0.7 -p THINKING True -p PROMPT v2` (see the first cell of `02_agent.ipynb`).
Engine settings are environment variables; on GPUs nobody else uses:

```bash
export DATAMATE_GPU_UTIL=0.9 DATAMATE_MAX_SEQS=48
```

Then open `03_evaluate.ipynb`, set `RUNS = ["v3_val300", "v2_val300"]` and run it: PA per version on
the same questions, τ sweep, wrong answers with their code and output.

**Send back** `runs/<run>/` (submission, logs, shard*.txt). They contain competition data: share them
inside the team only.

### Status (2026-10-30)

| version | validation | notes |
|---|---|---|
| v0 | PA −0.075 on 80 q | raw tables, greedy |
| v1 | PA +0.0625 on 80 q | gain is the abstain gate only (τ=0 precision 0.24) |
| v2 | PA −0.233 on 30 q | the model still parsed raw dates itself → NaT → answered 0 |
| v3 | **not run yet** | normalized tables; checked against 71 train answers without a GPU |

Details: [docs/leaderboard.md](docs/leaderboard.md). The dev machine's GPU driver is down, so v3's
first real numbers come from the server runs above.

### Before committing

Notebook outputs contain competition data (regulations §4.3); a test fails if a committed notebook
has outputs. Clear them first:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb && pytest -q && ruff check src tests notebooks
```

### Competition constraints this code respects

- Offline inference, ≤ 15B parameters in total (Qwen3.5-9B, Apache-2.0; optional embedders 0.6B).
- Whole private test in ≤ 6 h on 2 × RTX 5090; answers are written per shop, so a timeout keeps them.
- Per-question logs in `<out>/logs/` (Rule 8); re-run PA must stay within 0.03 of the upload.
- No training on test data; extra labels only on train (regulations §4.2).

Docs: [challenge spec](docs/challenge.md) · [data notes](docs/data-notes.md) ·
[research and design decisions](docs/research.md) · [results](docs/leaderboard.md)
