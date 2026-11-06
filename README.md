# DataMate — Zalo AI Challenge 2026

Trả lời câu hỏi tiếng Việt trên dữ liệu bán hàng lộn xộn của từng shop (csv / xlsx / sqlite), hoặc bỏ trống khi không chắc.
Chấm theo Penalized Accuracy: đúng +1, bỏ trống 0, sai −1.

Cách làm: Qwen3.5-9B kèm một LoRA tự train viết nhiều câu SQL DuckDB cho mỗi câu hỏi, chạy chúng trên dữ liệu đã nạp, bỏ phiếu
theo kết quả, và chỉ trả lời khi đủ đồng thuận.

Repo này chỉ dành cho thành viên trong đội: quy chế §4.3 và §8.2 cấm chia sẻ code và dữ liệu ra ngoài đội.

## Điểm public

| Bản | Thay đổi chính | Public |
|---|---|---|
| v2 | LoRA v2, few-shot, macro công nợ | 0.491 |
| | `qty` theo đơn vị cơ bản, trả lời `unanswerable` theo đa số | 0.516 |
| | Sinh mẫu mới, chuẩn hóa tên, hệ số đơn vị từ giá | 0.60 |
| v6 | Ghi chú trước SQL, ví dụ macro công nợ | 0.63 |
| v8 | LoRA v3 (thêm dữ liệu từ model 27B làm teacher), kiểm tra view theo dữ liệu | 0.639 |
| **v9** | v8 + nhánh `sql-hints` (gợi ý theo dạng câu hỏi, khóa bỏ phiếu theo cách chấm, sửa lớp công nợ) | **0.683** |

Mỗi lần chạy sinh mẫu mới nên điểm lệch được khoảng ±1 điểm. Lịch sử chi tiết và cách đọc điểm: [docs/leaderboard.md](docs/leaderboard.md).

## Pipeline

```
questions.jsonl ─┐
data/<shop>/  ───┼─► ingest      nạp mọi bảng vào DuckDB, đoán header, chuẩn hóa ngày/tiền/tên, profile cột
rulebook.md   ───┘      │
                        ▼
                     views       mỗi shop một lần: model viết K bản nháp SELECT ánh xạ dữ liệu thô sang
                     debts       v_sales / v_orders và lớp công nợ (macro debt_at, debt_new, debt_repaid);
                        │        kiểm tra trên dữ liệu (catalog, giờ, đơn vị), sửa lỗi, chọn bản đa số
                        ▼
                     hints       khung ngày, gợi ý theo dạng câu hỏi, giá trị khớp câu hỏi, giải nghĩa từ chat
                     prompts     2 system prompt × K/2 mẫu, few-shot từ ngân hàng ví dụ train
                        │
                        ▼
                     llm         vLLM + LoRA: 8 mẫu SQL thường, 4 mẫu thinking cho câu khó và câu công nợ
                     sqlexec     chạy read-only có timeout, ép kiểu đáp án, lỗi → vòng sửa (repair)
                        │
                        ▼
                     voting      khóa đáp án theo cách chấm, bỏ phiếu, ngưỡng 0.5, bỏ trống khi không chắc
                        │
                        ▼
                     submission  ghi submission.jsonl sau mỗi shop (hết giờ vẫn giữ phần đã trả lời)
```

`datamate/pipeline.py` nối các bước trên; mỗi bước là một method nên chạy được riêng, không cần model (xem `analysis/replay_run.py`).

## Cấu trúc repo

```
datamate/                 package suy luận (python -m datamate.predict)
  predict.py              CLI: đọc tham số, khởi động model, chạy Pipeline
  pipeline.py             các bước: nạp shop → view/công nợ → prompt → lấy mẫu → bỏ phiếu → ghi kết quả
  llm.py                  adapter quanh vLLM (chỉ file này import vllm)
  ingest.py               nạp và chuẩn hóa dữ liệu shop, profile cột, giá trị khớp câu hỏi
  views.py  debts.py      view chuẩn v_sales/v_orders và lớp công nợ: soạn, kiểm tra, chọn, mô tả
  hints.py  prompts.py    gợi ý từ câu hỏi; system prompt, user message, gợi ý sửa lỗi
  sqlexec.py voting.py    chạy một mẫu; bỏ phiếu và quyết định trả lời
  fewshot.py              ngân hàng ví dụ train và truy xuất
  scoring.py              Penalized Accuracy theo luật công bố (không dùng khi suy luận)
  submission.py           ghi, gộp shard, kiểm tra định dạng file nộp
  values.py               giá trị lọc không có trong dữ liệu (cho analysis/flags.py)
scripts/                  run_public.sh, run_holdout.sh, flags.sh (cờ của bản đã nộp, dùng chung cho mọi lệnh chạy)
docker/                   Dockerfile, predict.sh (entry point), requirements, model_manifest.json
training/                 dữ liệu SFT, ghi chú, ngân hàng few-shot, train LoRA, kiểm tra kernel fla, gộp LoRA
analysis/                 công cụ đánh giá offline: error_matrix, vote_analysis, flags, view_*, replay_run, gate_cv
notebooks/                01_eda.ipynb (dữ liệu), 02_evaluation.ipynb (điểm, lỗi, so sánh bản nộp)
configs/public_v9/        view và lớp công nợ của lượt 0.683 (chỉ để chạy lại public, không dùng cho private)
assets/fewshot_bank.jsonl ngân hàng ví dụ (câu hỏi train và SQL đúng)
adapters/sql_rft_v3/      LoRA v3 (Git LFS)
splits/                   danh sách id: holdout600, holdout_shops, val300, smoke6, rft_train24
tests/                    pytest, chỉ CPU
docs/                     đề bài, ghi chú dữ liệu, nghiên cứu, lịch sử điểm
```

Không có trong git: `data/` (dữ liệu BTC), `models/` (trọng số gốc), `runs/` (kết quả chạy, log, checkpoint).

## Cài đặt

Cần GPU từ 24 GB (đã chạy trên 1 RTX PRO 4000 Blackwell 24 GB, RAM 30 GB), CUDA ≥ 12.8, Python 3.12, `git-lfs`.

```bash
git clone git@github.com:hugebenevolence/zalo-ai-challenge-datamate-26.git && cd zalo-ai-challenge-datamate-26
git lfs install && git lfs pull                              # LoRA v3
python3 -m venv .venv && .venv/bin/pip install -e ".[llm,dev]"
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
# Dữ liệu BTC: data/train/{questions,answers}.jsonl, data/public_test/questions.jsonl, mỗi split kèm rulebook.md, calendar.csv, data/<shop>/
.venv/bin/python -m pytest -q                                # khoảng 1 giây, không cần GPU
```

## Chạy

```bash
bash scripts/run_public.sh runs/my_run            # public với view/công nợ đã lưu của bản 0.683; khoảng 3,5 giờ trên 1 GPU 24 GB
FRESH=1 bash scripts/run_public.sh runs/fresh     # soạn lại mọi view/công nợ, đúng như bản chạy private trong Docker
bash scripts/run_holdout.sh runs/holdout          # 632 câu holdout của train, kèm điểm (chỉ là smoke test, xem docs/leaderboard.md)
python -m datamate.submission check runs/my_run/submission.jsonl     # kiểm tra định dạng trước khi nộp
```

Đổi cấu hình bằng biến môi trường: `GPU_MEM=0.85`, `NUM_GPUS=2`, `MODEL_DIR=...`, `LORA_DIR=...`, `ENGINE='{...}'` (giảm
`max_num_seqs` khi hết bộ nhớ). Cờ của bản đã nộp nằm ở `scripts/flags.sh`; `python -m datamate.predict --help` liệt kê mọi cờ.

### Docker (bản chạy private)

```bash
docker build -f docker/Dockerfile -t datamate .
docker run --rm --gpus all --network none -v /duong/dan/split:/data:ro -v "$PWD/result:/result" datamate
```

Image chứa model gốc, LoRA v3 và ngân hàng ví dụ; chạy offline, mỗi GPU một tiến trình, tự soạn view cho từng shop, ghi
`/result/submission.jsonl` và log từng câu ở `/result/logs/`. Mỗi tiến trình dừng ở `TIMEOUT_SECONDS` (mặc định 21000 giây), phần
đã trả lời vẫn được tính. Cần làm xong bước tải model và `git lfs pull` trước khi build; image khoảng 70 GB.

## Train LoRA

Môi trường train riêng (`.venv-train`, xem `training/requirements.txt`; cài thêm `fla-core==0.6.0` để có kernel Gated DeltaNet).

```bash
python training/build_pool.py ...                  # gộp các lượt chạy train có nhãn thành pool mẫu
python training/build_sft.py RUN ... --teacher TEACHER_RUN ... --fewshot assets/fewshot_bank.jsonl --out data/sft/sql_rft_v3.jsonl --easy 800
python training/add_notes.py data/sft/sql_rft_v3.jsonl data/sft/sql_rft_v3_notes.jsonl
.venv-train/bin/python training/check_fla.py       # kernel fla khớp bản tham chiếu PyTorch
.venv-train/bin/python training/train_lora.py data/sft/sql_rft_v3_notes.jsonl --out runs/lora/sql_rft_v3 --model models/Qwen3.5-9B \
    --init-adapter adapters/sql_rft_v3 --max-len 32768 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' \
    --no-thinking --completion-logits
```

LoRA v3 được train tiếp từ v2 (v2 còn trong lịch sử git); bản sau train tiếp từ `adapters/sql_rft_v3` như lệnh trên. Chỉ dùng dữ
liệu train; câu hỏi public và private không bao giờ vào dữ liệu train. LoRA v3: 2910 dòng, 364 bước, khoảng 8,2 giờ
trên 1 GPU 24 GB (fla và `--completion-logits` cho khoảng 1300 tok/s).

## Đánh giá

- `notebooks/01_eda.ipynb`: các shop, bảng, định dạng file, loại đáp án, dạng câu hỏi; so train với public (public chỉ có số đếm).
- `notebooks/02_evaluation.ipynb`: điểm một lượt chạy train theo loại / shop / dạng câu, nhóm lỗi, oracle@K, ngưỡng bỏ phiếu;
  so các bản nộp public với nhau không cần nhãn.
- `analysis/`: `error_matrix.py` (ma trận lỗi), `vote_analysis.py` (điểm mất ở bước quyết định), `flags.py` (cờ không cần nhãn
  theo từng bước), `view_audit.py` / `view_tests.py` / `view_flags.py` (kiểm tra view đã lưu), `gate_cv.py` (ngưỡng theo
  leave-one-shop-out), `replay_run.py` (dựng lại prompt và bỏ phiếu của một lượt đã lưu: một thay đổi không được phép đổi hành
  vi phải cho 0 khác biệt).

Notebook được commit **không có output** (output chứa dữ liệu thi, quy chế §4.3): chạy lại trên máy để xem.

## Luật thi và cách repo tuân thủ

- Suy luận offline, tổng ≤ 15B tham số (Qwen3.5-9B Apache-2.0 + LoRA 43M), không có người can thiệp.
- Không hard-code theo id câu hỏi hay id shop. `configs/public_v9` (view lưu theo shop) chỉ dùng để chạy lại public; bản Docker
  cho private luôn soạn view mới.
- Không train trên tập test; chú thích thêm chỉ trên train (§4.2). Các gợi ý trong prompt là luật trong rulebook hoặc kiến thức
  tiếng Việt chung, đã đối chiếu với đáp án train; những gì chỉ khớp câu hỏi public (danh sách từ đồng nghĩa lấy từ tên món
  public, gợi ý dòng sản phẩm) đã bị loại.
- Toàn bộ private test chạy ≤ 6 giờ trên 2 × RTX 5090; log từng câu cho Rule 8 (chạy lại lệch ≤ 0.03).

Tài liệu: [đề bài](docs/challenge.md) · [ghi chú dữ liệu](docs/data-notes.md) · [nghiên cứu và quyết định thiết kế](docs/research.md) ·
[khảo sát kiến trúc](docs/architecture-research-2026-10-29.md) · [lịch sử điểm](docs/leaderboard.md)
