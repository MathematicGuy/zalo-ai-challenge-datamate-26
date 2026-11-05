# Build from the repo root, after downloading the base model into sqlbase/models/Qwen3.5-9B and running git lfs pull:
#   docker build -f sqlbase/Dockerfile -t datamate .
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
WORKDIR /app
COPY sqlbase/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY sqlbase/models/Qwen3.5-9B /models/Qwen3.5-9B
COPY runs/lora/sql_rft_v2/adapter /models/lora
COPY sqlbase/*.py sqlbase/*.json sqlbase/predict.sh sqlbase/fewshot_bank24m.jsonl /app/
RUN chmod +x /app/predict.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
