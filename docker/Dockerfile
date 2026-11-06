# The private-test image: inference and training code, README, base model, LoRA v3 and the example bank. Never the competition data or run outputs.
# Build from the repo root after downloading the base model into models/Qwen3.5-9B and running git lfs pull:
#   docker build -f docker/Dockerfile -t datamate .
#   docker run --gpus all -v /path/to/test:/data:ro -v /path/to/result:/result datamate
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
WORKDIR /app
COPY docker/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# Training pins its own torch/transformers, so it gets a separate venv and the vLLM environment stays untouched.
COPY training/requirements.txt /app/training/requirements.txt
RUN python3 -m venv /opt/venv-train \
    && /opt/venv-train/bin/pip install --no-cache-dir -r /app/training/requirements.txt \
    && /opt/venv-train/bin/pip install --no-cache-dir fla-core==0.6.0 --no-deps
COPY models/Qwen3.5-9B /models/Qwen3.5-9B
COPY adapters/sql_rft_v3 /models/lora
COPY datamate /app/datamate
COPY training /app/training
COPY README.md /app/README.md
COPY assets/fewshot_bank.jsonl /app/assets/fewshot_bank.jsonl
COPY scripts/flags.sh docker/predict.sh docker/model_manifest.json /app/
RUN chmod +x /app/predict.sh
ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
