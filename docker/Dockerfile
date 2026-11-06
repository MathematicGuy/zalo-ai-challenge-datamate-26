# The private-test image: code, base model, LoRA v3 and the example bank; never the competition data or run outputs.
# Build from the repo root after downloading the base model into models/Qwen3.5-9B and running git lfs pull:
#   docker build -f docker/Dockerfile -t datamate .
#   docker run --gpus all -v /path/to/test:/data:ro -v /path/to/result:/result datamate
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
WORKDIR /app
COPY docker/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY models/Qwen3.5-9B /models/Qwen3.5-9B
COPY adapters/sql_rft_v3 /models/lora
COPY datamate /app/datamate
COPY assets/fewshot_bank.jsonl /app/assets/fewshot_bank.jsonl
COPY scripts/flags.sh docker/predict.sh docker/model_manifest.json /app/
RUN chmod +x /app/predict.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
