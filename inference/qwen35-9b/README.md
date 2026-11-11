# Qwen3.5-9B trên vLLM: máy chủ LLM local cho DataMate

Thư mục này đóng gói đúng cấu hình máy chủ đã tinh chỉnh cho `Qwen/Qwen3.5-9B`, để cả nhóm dựng lại được trên máy mình. Máy chủ là vLLM với API kiểu OpenAI trên cổng `:8000`, model tên `qwen3.5-9b`. Cấu hình đã đo trên RTX 5060 Ti 16 GB (Blackwell `sm_120`, Ubuntu 26.04) ngày 2026-11-03/04.

| File | Vai trò |
|---|---|
| `setup.sh` | Cài một lần, chạy lại không sao (idempotent), không cần sudo. Các bước: tạo venv uv Python 3.12, tải mã vLLM đúng commit, vá patch, cài gói đã ghim, tải trọng số |
| `serve.sh` | Chạy vLLM ở foreground với các cờ đã tinh chỉnh. Mọi giá trị mặc định đổi được bằng biến môi trường. `DRY_RUN=1` chỉ in lệnh |
| `server.sh` | `start` / `stop` / `status` / `logs`: chạy nền, quản lý bằng pidfile |
| `smoke_test.py` | Kiểm tra nhanh: `/v1/models`, một câu chat, một lời gọi tool, tính lặp lại. Chỉ dùng thư viện chuẩn của Python |
| `requirements-vllm.txt` | Phiên bản ghim của mọi gói trong venv đã đo (198 gói) |
| `patches/gdn-batch-invariance.patch` | Patch vLLM bắt buộc để bật batch invariance cho Qwen3.5 (xem [Tái lập kết quả](#tái-lập-kết-quả)) |

Venv (`.venv-vllm/`), mã vLLM (`vendor/`) và log (`logs/`) nằm ngay trong thư mục này nhưng đã được `.gitignore`, không commit.

## Yêu cầu

| Thành phần | Yêu cầu | Đã đo trên |
|---|---|---|
| GPU | NVIDIA, **≥ 16 GB VRAM**. Ở mặc định máy chủ dùng ~15,0 GB: trọng số FP8 10,5 GiB + KV 3,4 GiB + CUDA graph/activation ~0,6 GiB | RTX 5060 Ti 16 GB, `sm_120` |
| Driver | Bản R580 trở lên (wheel CUDA 13.0) | `595.99.02` (nvidia-smi báo CUDA 13.2) |
| CUDA | Không cần cài CUDA toolkit: CUDA runtime đi kèm wheel (`torch 2.14.0+cu130`), còn vLLM dùng binary dựng sẵn nên không biên dịch. **Blackwell bắt buộc wheel CUDA ≥ 12.8** (wheel cu12.1/12.4 cũ không chạy trên `sm_120`) | |
| Ổ đĩa | Trọng số BF16 ~19 GB (HF cache), venv ~8,2 GB, mã vLLM ~1 GB, cộng thêm cache của uv. **Nên còn trống ≥ 40 GB** | |
| Phần mềm | `uv` (≥ 0.11, tự tải Python 3.12 nếu máy chưa có), `git`, `curl`. Chỉ cần mạng lúc cài | uv 0.11.19 |
| Hệ điều hành | Linux x86_64 (file ghim được freeze trên x86_64) | Ubuntu 26.04, kernel 7.0 |

**Cài driver** (làm tay, cần sudo; `setup.sh` không làm bước này). Trên Ubuntu 26.04 với card Blackwell:

```bash
sudo apt install nvidia-driver-595-open linux-modules-nvidia-595-open-$(uname -r) \
                 linux-modules-nvidia-595-open-generic-hwe-26.04   # gói cuối: tự theo khi nâng kernel
sudo reboot
nvidia-smi        # phải thấy GPU và "CUDA Version: 13.x"
```

Nếu `nvidia-smi` báo lỗi sau khi nâng kernel, máy đang thiếu module cho kernel mới: cài lại `linux-modules-nvidia-595-open-$(uname -r)` rồi khởi động lại.

## Cài đặt

```bash
cd inference/qwen35-9b
./setup.sh                       # ~30 GB tải về lần đầu; chạy lại thì bỏ qua các bước đã xong
```

Script làm 5 bước:

1. `uv venv --python 3.12 .venv-vllm`
2. Tải mã vLLM (fetch nông theo SHA) về `vendor/vllm`: nhánh PR LFM3 của vLLM (`atmikshetty/vllm@5f67fbc`, PR #64959 + #64647, nền upstream `fac2ce2`). Đây đúng là cây mã dùng để đo profile. Nếu commit fork không còn, script tự chuyển sang upstream `vllm-project/vllm@fac2ce2`. Hai commit LFM3 chỉ thêm `lfm3.py` và nhánh conv width-7, Qwen3.5 không dùng tới.
3. `git apply patches/gdn-batch-invariance.patch`. Đã vá rồi thì bỏ qua.
4. `uv pip install --torch-backend cu130 -r requirements-vllm.txt`, sau đó cài vLLM dạng editable trên binary dựng sẵn (`VLLM_USE_PRECOMPILED=1`). Commit binary được ghim là `fac2ce2`, biến thể `cu130`, tải từ `wheels.vllm.ai`.
5. `hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Từ huggingface-hub 2.x, lệnh `huggingface-cli` đổi tên thành `hf`. Model công khai nên không cần `HF_TOKEN`.

Các biến hay dùng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VENV`, `VLLM_SRC` | `./.venv-vllm`, `./vendor/vllm` | nơi cài |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | nơi lưu trọng số |
| `SKIP_WEIGHTS=1` | | bỏ qua bước tải trọng số, vd. khi trọng số đã có sẵn trong cache |
| `FORCE_REINSTALL=1` | | cài lại gói dù stamp ghi là đã xong |
| `TORCH_BACKEND`, `WHEEL_VARIANT` | `cu130` | đổi biến thể CUDA. Đổi là không còn đúng bộ đã đo |

## Chạy và dừng máy chủ

```bash
./server.sh start      # chạy nền (setsid + nohup), ghi pid vào logs/vllm.pid, log vào logs/vllm.log, chờ /v1/models
./server.sh status     # pid + danh sách model
./server.sh logs       # tail -f log
./server.sh stop       # SIGTERM theo pidfile, chờ tối đa 120 s cho tắt sạch
./serve.sh             # hoặc chạy foreground (Ctrl-C để dừng)
DRY_RUN=1 ./serve.sh   # chỉ in lệnh vllm serve, không đụng GPU
```

- Khởi động mất ~50-60 giây (không có torch.compile). `start` chờ tối đa `START_TIMEOUT=600` giây.
- **Luôn dừng bằng pidfile (`./server.sh stop` hoặc `kill $(cat logs/vllm.pid)`), không bao giờ dùng `pkill -f` / `pgrep -f`**: hai lệnh này khớp cả shell của chính nó và có thể giết nhầm tiến trình khác. `stop` không tự gửi SIGKILL (lý do ở [Xử lý sự cố](#xử-lý-sự-cố)).
- Mỗi lúc chỉ chạy một model trên một GPU. Nếu cổng đã có máy chủ khác thì `start` từ chối chạy.
- Tham số thêm sau `start` được chuyển thẳng cho `vllm serve`, vd. `./server.sh start --max-num-seqs 8`.
- Chạy offline sau khi đã tải trọng số: `HF_HUB_OFFLINE=1 ./server.sh start`.

Kiểm tra nhanh, từ Python 3.10+ bất kỳ:

```bash
python3 smoke_test.py                                  # mặc định http://localhost:8000/v1, model qwen3.5-9b
python3 smoke_test.py --base-url http://localhost:8001/v1
```

Kết quả mong đợi: 4 dòng `[PASS]`: `models`, `chat`, `tool call` (gọi `revenue` với ngày `2026-02-09`), `determinism`. Cuối cùng in `OK`, mã thoát 0.

## Using it from this repo (datamate-26)

> This section is specific to `zalo-ai-challenge-datamate-26`; the rest of this README comes from Zalo-Track-2.

The pipeline talks to the server through `--backend openai` (`datamate/llm_remote.py`), with the SQL LoRA served as model `sql`.
Settings that fit a 16 GB RTX 5060 Ti with the LoRA loaded (MTP draft + LoRA runs out of memory, fp8 KV doubles the KV room,
and fp8 KV needs the Triton attention backend when nvcc is missing):

```bash
REASONING_PARSER= SPEC_CONFIG= BATCH_INVARIANT=0 MAX_NUM_SEQS=32 ATTENTION_BACKEND=TRITON_ATTN \
  EXTRA_ARGS="--kv-cache-dtype fp8" START_TIMEOUT=900 bash inference/qwen35-9b/server.sh start
```

| Variable | Value here | Why |
|---|---|---|
| `LORA`, `LORA_NAME` | `adapters/sql_rft_v3`, `sql` (defaults) | the pipeline sends LoRA samples to model `sql` |
| `MAX_MODEL_LEN` | `32768` (default) | the submission uses 49152; that does not fit next to the LoRA on 16 GB |
| `REASONING_PARSER` | empty | the pipeline parses `<think>` itself and needs the raw text |
| `SPEC_CONFIG` | empty | MTP's draft head (~2 GB) + LoRA does not fit |
| `EXTRA_ARGS` | `--kv-cache-dtype fp8` | ~167k KV tokens instead of ~80k |

Then run a split (Qwen, Claude Haiku 5.5 or Gemini 3.5 Flash Lite behind the same pipeline):

```bash
bash scripts/run_compare.sh openai synth_private runs/qwen35_private
bash scripts/run_compare.sh claude synth_private runs/haiku55_private   # CLAUDE_API / ANTHROPIC_API_KEY in ./env
bash scripts/run_compare.sh gemini synth_private runs/gemini35fl_private # every GEMINI_API_KEY* in ./env, rotated
```

`notebooks/observation.ipynb` compares the runs.

## Số liệu kỳ vọng

Đo trên RTX 5060 Ti 16 GB với cấu hình mặc định: chế độ tool, thinking bật, `--dialect matched`, 8 worker. PA là điểm theo luật cuộc thi.

| Lần chạy | Câu | Đúng / Sai / Bỏ qua | PA | giây/câu (wall) |
|---|---|---|---|---|
| eval 60 (`20261103-2240-qwen35-9b-tools60-r`) | 60 | 60 / 0 / 0 | +1.000 | 0,77 |
| hold-out 60 (`20261103-2237-qwen35-9b-holdout60`; lần chạy lại `…-holdout60-r` cùng kết quả) | 60 | 59 / 0 / 1 | +0.983 | 0,94 |
| train 1000 (`20261103-2338-qwen35-9b-train1000`) | 1000 | 973 / 21 / 6 | +0.952 | 0,75 |
| train 3000 (`20261103-2351-qwen35-9b-train3000`) | 3000 | 2930 / 60 / 10 | +0.957 | 0,73 |
| bộ tinh chỉnh A+B (72 câu train) | 72 | 67 / 3 / 2 | +0.889 | 0,91-0,98 |
| như trên, thinking tắt (`LOCAL_LLM_TOOL_THINKING=0`) | 72 | 69 / 1 / 2 | +0.931 | 0,44-0,48 |

Thông lượng: ~41,7 token/giây mỗi luồng ở 8 worker (MTP k=3, tỉ lệ nhận 79%), 62-67 token/giây khi chạy 1 worker. Chạy tuần tự (1 worker) mất ~4,6 giây/câu. 16 worker không nhanh hơn (~1,0 giây/câu) vì KV pool chỉ chứa ~5-6 prompt 10,4k token cùng lúc. Prefix cache trúng 89-94%. VRAM ~15,0 GB.

Chi tiết phép đo nằm trong profile `llm-profiles/qwen3.5-9b.md` ở workspace của người tinh chỉnh (không có trong repo). Phần chính đã chép vào các chú thích của `serve.sh`.

## Cấu hình khuyến nghị

Dùng đúng cấu hình của máy đã đo: **1 × RTX 5060 Ti 16 GB** (Blackwell `sm_120`). Mọi số liệu ở trên đều đo trên cấu hình này. Mặc định của `serve.sh` đã khớp, nên không cần đặt thêm biến nào:

| Thiết lập | Giá trị |
|---|---|
| GPU | 1 card, chạy một máy chủ duy nhất trên `:8000` |
| Trọng số | FP8 (`QUANT=fp8_per_tensor`), ~10,5 GiB |
| KV cache | `KV_CACHE_MEMORY_BYTES=3481M` (để chừa chỗ cho màn hình desktop chạy cùng) |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `MAX_MODEL_LEN` | `16384` (prompt tool ~10,45k + 4k đầu ra) |
| `MAX_NUM_SEQS` | `16` |
| Speculative decoding | MTP, `num_speculative_tokens=3` |
| Batch invariance | bật (`BATCH_INVARIANT=1`), seed `42` |
| Harness | `LOCAL_LLM_WORKERS=8`, `LOCAL_LLM_TOOL_THINKING=1` |

Khi dùng cấu hình này:

- **Không đổi các cờ trên.** Đổi bất kỳ cờ nào là ra khỏi bộ đã đo: vài câu có thể đổi đáp án, và số liệu kỳ vọng không còn đúng.
- **Tắt các tiến trình khác đang dùng GPU** trước khi `start`. vLLM không chịu khởi động nếu bộ nhớ trống < 0,90 × tổng VRAM.
- **Giữ 8 worker.** 16 worker không nhanh hơn (~1,0 giây/câu), vì KV pool chỉ chứa ~5-6 prompt cùng lúc.

## Tái lập kết quả

Luật cuộc thi: chạy lại phải ra PA trong khoảng ±0,03. Cấu hình này cho cùng record 36/36 giữa các lần chạy lại và giữa 1 / 8 / 16 worker. Có được điều đó là nhờ:

- **Batch invariance** (`VLLM_BATCH_INVARIANT=1`, mặc định `BATCH_INVARIANT=1`): kết quả greedy không phụ thuộc số request chung batch. Qwen3.5 có 24 lớp Gated DeltaNet (GDN), mà vLLM gốc từ chối khởi động với lỗi `VLLM batch_invariant mode is not supported for GDN_ATTN`. **Patch `gdn-batch-invariance.patch`** sửa hai chỗ:
  1. Backend GDN khai báo `supports_batch_invariance`: các kernel GDN tính theo từng sequence, không cộng gộp qua batch.
  2. Khi bật chế độ này, bước decode MTP của GDN luôn chạy đường Triton. Lý do: kernel CUDA hợp nhất chỉ được dùng ở bước không có prefill, nên cùng một token có thể đi qua hai kernel làm tròn khác nhau tuỳ request khác có đang prefill hay không.

  Đo được: không có patch (BI tắt) chỉ 33-35/36 record khớp. Có patch thì 36/36, tốc độ như cũ.
- **`temperature=0`, `seed=42`** ở cả máy chủ (`--seed 42`) và client (`llm/config.py`). Thêm `--generation-config vllm` để cấu hình sinh của checkpoint không ghi đè tham số của harness.
- **FLASH_ATTN** (mặc định trên `sm_120`). Hai lựa chọn TRITON_ATTN và `--kv-cache-dtype fp8` nhanh hơn ~1,2 lần nhưng record đổi theo số worker, nên không dùng.
- **Giữ MTP bật** (`SPEC_CONFIG` mặc định k=3). Tắt spec (`SPEC_CONFIG=`) thì đường decode GDN không batch-invariant hoàn toàn: 34/36 lời gọi tool khớp.
- **Client**: làm nóng prefix chung + `cache_salt` theo run (mặc định bật trong `llm/config.py`).
- **Ghim** phiên bản gói (`requirements-vllm.txt`), commit vLLM, patch và revision trọng số (`MODEL_REVISION`).
- Nâng vLLM thì phải vá lại patch và đo lại 1/8/16 worker: bản mới có thể thêm đường kernel GDN khác chọn theo thành phần batch.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `nvidia-smi` lỗi, hoặc `No CUDA GPUs are available` sau khi nâng kernel | Thiếu module driver cho kernel mới: `sudo apt install linux-modules-nvidia-595-open-$(uname -r)`, khởi động lại |
| `No CUDA GPUs are available` ngay sau khi dừng máy chủ; `journalctl -k` có `Xid 62` / `Xid 154` | GPU bị lỗi phần cứng lúc tắt (gặp một lần trên máy dev, ở lần restart thứ 7 liên tiếp, khi vLLM force-kill EngineCore). Phải reset GPU: khởi động lại máy, hoặc `sudo nvidia-smi -r` khi không có ứng dụng đồ hoạ. Tránh restart dồn dập; mỗi lần đánh giá chỉ khởi động một lần. Vì thế `server.sh stop` không tự SIGKILL |
| `VLLM batch_invariant mode is not supported for GDN_ATTN` | Patch chưa được vá, hoặc đang dùng nhầm venv/vLLM khác. Chạy lại `./setup.sh`, kiểm tra `git -C vendor/vllm diff --stat` |
| OOM lúc khởi động, khi đo bộ nhớ CUDA graph | `MAX_NUM_SEQS` quá lớn (mặc định 256 của vLLM). Giữ 16, tối đa 32 |
| vLLM không khởi động vì bộ nhớ GPU trống nhỏ hơn mức `GPU_MEMORY_UTILIZATION` yêu cầu | GPU đang có tiến trình khác (desktop, server khác). Dừng chúng, hoặc hạ `GPU_MEMORY_UTILIZATION` / `KV_CACHE_MEMORY_BYTES` |
| `--kv-cache-memory-bytes` báo lỗi khi parse | Hậu tố nhị phân chỉ nhận số nguyên: `3481M`, không phải `3.4G` |
| Đặt `VLLM_ATTENTION_BACKEND` mà không có tác dụng | Bản vLLM này không đọc biến đó. Dùng `ATTENTION_BACKEND=...` (thành cờ `--attention-backend`). Lưu ý TRITON_ATTN làm mất tính lặp lại |
| Đặt `--kv-cache-dtype fp8` thì attention tự đổi sang TRITON_ATTN | FA2 trên `sm_120` không hỗ trợ KV FP8. Không dùng (mất tính lặp lại theo số worker) |
| Log báo torch.compile không hỗ trợ model; chạy CUDA graph `FULL_DECODE_ONLY` | Bình thường với Qwen3.5 trong bản này. Prefill chạy eager, chỉ chiếm ≤ 4% thời gian |
| 24 dòng cảnh báo `Attempted to load weight layers.N.linear_attn.norm.weight with dtype torch.float32 ...` | Vô hại, lần nào khởi động cũng có |
| Bản dựng sẵn Blackwell mặc định hỏng attention | Thử `ATTENTION_BACKEND=TRITON_ATTN ./server.sh start` để khoanh lỗi (nhớ là mất batch invariance theo số worker) |
| `setup.sh` không tải được binary vLLM | Kiểm tra mạng tới `wheels.vllm.ai`. Có thể đặt `WHEEL_COMMIT` (SHA đủ 40 ký tự) sang commit tổ tiên gần nhất có wheel `cu130` |
| `setup.sh` báo `vendor/vllm is at ..., expected ...` | Thư mục mã vLLM đang ở commit khác. Xoá `vendor/vllm` (hoặc đặt `VLLM_SRC=...`) rồi chạy lại |
