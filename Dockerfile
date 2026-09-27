# Qwen-Image 2.1 still worker for RunPod Serverless.
#
# Built by GitHub Actions (.github/workflows/build.yml) and published as a
# public image, so nobody needs Docker locally. The model weights are NOT in
# the image: the endpoint attaches Qwen/Qwen-Image-2.1 as a RunPod cached
# model (see README.md); only the small step-distilled LoRAs are baked in,
# which keeps this image around 9 GB.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/hf-home

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1

COPY requirements.txt /worker/requirements.txt
RUN pip install -r /worker/requirements.txt \
    && python -c "from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel; from transformers import Qwen3VLForConditionalGeneration; import runpod, peft; from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_"

# Step-distilled LoRAs for the `viggle-6` and `pruna-8` variants (~1.6 GB),
# pinned to the revisions handler.py names.
RUN python -c "from huggingface_hub import hf_hub_download as get; \
get('Viggle/Qwen-Image-2.1-viggle-turbo', 'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors', revision='bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13', local_dir='/models/lora'); \
get('PrunaAI/Pruna-Qwen-Image-2.1', 'p_qwen_image_2.1_8step_v0.1.safetensors', revision='113e63bb993001b3411eb3470b84fc444040cd7e', local_dir='/models/lora')" \
    && rm -rf /models/lora/.cache /tmp/hf-home

COPY handler.py /worker/handler.py
WORKDIR /worker
CMD ["python", "-u", "/worker/handler.py"]
