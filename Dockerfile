# Qwen-Image 2.1 still worker for RunPod Serverless.
#
# Built by GitHub Actions (.github/workflows/build.yml) and published as a
# public image, so nobody needs Docker locally. The model weights are NOT in
# the image: the endpoint attaches Qwen/Qwen-Image-2.1 as a RunPod cached
# model (see README.md), which keeps this image around 7 GB.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/hf-home

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1

COPY requirements.txt /worker/requirements.txt
RUN pip install -r /worker/requirements.txt \
    && python -c "from diffusers import QwenImage21Pipeline; from transformers import Qwen3VLForConditionalGeneration; import runpod; from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_"

COPY handler.py /worker/handler.py
WORKDIR /worker
CMD ["python", "-u", "/worker/handler.py"]
