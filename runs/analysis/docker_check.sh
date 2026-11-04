#!/usr/bin/env bash
# Builds the image and checks what can be checked while the GPU is busy: the code loads inside it and the container sees the GPU.
cd "$(dirname "$0")/../.."
set -u
sudo docker build -f sqlbase/Dockerfile -t datamate . 2>&1 | tail -n 25
echo "build exit ${PIPESTATUS[0]}"
sudo docker images datamate --format '{{.Repository}} {{.Size}}'
sudo docker run --rm --network none --entrypoint python3 datamate predict.py --help 2>&1 | grep -E "usage|canon-names|answer-unanswerable|Error|Traceback" | cut -c1-120 | head -8
sudo docker run --rm --network none --entrypoint sh datamate -c 'ls /models/Qwen3.5-9B | wc -l; ls -la /models/lora | grep -c safetensors; ls /app | tr "\n" " "; du -sh /models/lora' 2>&1 | cut -c1-260
sudo docker run --rm --gpus all --network none --entrypoint nvidia-smi datamate --query-gpu=name,memory.total --format=csv,noheader 2>&1 | cut -c1-100
echo "docker check finished"
