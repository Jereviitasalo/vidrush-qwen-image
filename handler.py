"""Qwen-Image 2.1 still worker for RunPod Serverless.

One job draws one image. The weights are not in the image: the endpoint
attaches `Qwen/Qwen-Image-2.1` as a RunPod cached model, so they are already on
the host at /runpod-volume/huggingface-cache/hub when the worker starts, and
RunPod does not bill the download. Without that cache (a local run, or an
endpoint set up by hand) the worker downloads the pinned revision itself.

Request (`input`):
    version     1
    prompt      str, 1-4000 chars
    width       int, multiple of 32, 256-2048   (default 1344)
    height      int, multiple of 32, 256-2048   (default 768)
    steps       int, 8-60                        (default 40, Qwen's value)
    seed        int | null
    references  [base64 image], 0-10: condition images, read in order
    reference_resolution  int, 256-1024 (default 512): side of the square
                area each condition image is resized to. Condition tokens
                join the sequence, so a 1 MP reference roughly quadruples a
                1 MP draw; 512 keeps the look and subject at a fraction.
    quality     int, JPEG quality 70-100         (default 92)

Reply: the JPEG as base64 plus timings and the GPU, so the app can price the
job from RunPod's execution time.
"""
import base64
import io
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

MODEL_REPO = 'Qwen/Qwen-Image-2.1'
MODEL_REVISION = os.environ.get('QWEN_MODEL_REVISION', '790c92633540aa0cb11d9abf19eb46d861714758')
RUNPOD_CACHE = Path('/runpod-volume/huggingface-cache/hub')
# Every weight resident in bf16 needs ~34-40 GB at 1 MP; smaller cards offload.
RESIDENT_MIN_VRAM_GB = 40
# FP8 transformer (weights and activations) on Ada/Hopper/Blackwell tensor
# cores: ~1.4x faster at the same look, with or without condition images
# (a posterized copy of the reference comes from reusing the reference's own
# seed, not from FP8). Ampere has no FP8 and stays bf16. QWEN_FP8=0 turns it off.
FP8 = os.environ.get('QWEN_FP8', '1') == '1'

MAX_REFERENCES = 10
MAX_PIXELS = 4_194_304  # 2048 x 2048
MAX_REFERENCE_BYTES = 12 * 1024 * 1024


def snapshot_path():
    """The cached snapshot RunPod mounted, else a download of the pinned revision."""
    snapshots = RUNPOD_CACHE / ('models--' + MODEL_REPO.replace('/', '--')) / 'snapshots'
    pinned = snapshots / MODEL_REVISION
    if (pinned / 'model_index.json').is_file():
        return str(pinned), 'runpod-cache'
    if snapshots.is_dir():
        for candidate in sorted(snapshots.iterdir()):
            if (candidate / 'model_index.json').is_file():
                print(f'[qwen-image] pinned revision not cached; using cached snapshot {candidate.name}', flush=True)
                return str(candidate), 'runpod-cache'
    from huggingface_hub import snapshot_download
    print('[qwen-image] no RunPod model cache; downloading the weights (billed, several minutes)', flush=True)
    return snapshot_download(MODEL_REPO, revision=MODEL_REVISION), 'download'


def to_fp8(pipe):
    """Quantize the transformer's linear layers to FP8, or leave bf16 when the card or torchao cannot."""
    if not FP8 or torch.cuda.get_device_capability(0) < (8, 9):
        return 'bf16'
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
        torch.cuda.empty_cache()
        return 'fp8'
    except Exception as error:  # keep serving in bf16 rather than failing the worker
        print(f'[qwen-image] FP8 unavailable, staying bf16: {error}', flush=True)
        return 'bf16'


def load_pipeline():
    from diffusers import QwenImage21Pipeline

    started = time.time()
    path, source = snapshot_path()
    pipe = QwenImage21Pipeline.from_pretrained(path, dtype=torch.bfloat16)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= RESIDENT_MIN_VRAM_GB:
        pipe.to('cuda')
        precision = to_fp8(pipe)
        placement = 'resident'
    else:
        pipe.enable_model_cpu_offload()
        precision, placement = 'bf16', 'offload'
    pipe.set_progress_bar_config(disable=True)
    return pipe, {
        'loadSeconds': round(time.time() - started, 2),
        'source': source,
        'placement': placement,
        'precision': precision,
        'revision': Path(path).name,
        'freeVramGb': round(torch.cuda.mem_get_info()[0] / 1024**3, 1),
    }


PIPE, LOAD = load_pipeline()
GPU = torch.cuda.get_device_name(0)
COLD = {'pending': True}
print(f'[qwen-image] ready on {GPU}: {LOAD}', flush=True)


class BadRequest(ValueError):
    pass


def int_field(payload, key, default, low, high, multiple=1):
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f'{key} must be an integer')
    if not low <= value <= high or value % multiple:
        raise BadRequest(f'{key} must be {low}-{high}' + (f' and a multiple of {multiple}' if multiple > 1 else ''))
    return value


def decode_reference(index, value):
    if not isinstance(value, str) or not value:
        raise BadRequest(f'references[{index}] must be a base64 string')
    if value.startswith('data:'):
        value = value.split(',', 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError:
        raise BadRequest(f'references[{index}] is not valid base64') from None
    if len(raw) > MAX_REFERENCE_BYTES:
        raise BadRequest(f'references[{index}] is larger than {MAX_REFERENCE_BYTES // 1024 // 1024} MB')
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise BadRequest(f'references[{index}] is not a readable image') from None
    return image.convert('RGB')


def parse(payload):
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise BadRequest('input.version must be 1')
    prompt = payload.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise BadRequest('prompt must be 1-4000 characters')
    width = int_field(payload, 'width', 1344, 256, 2048, 32)
    height = int_field(payload, 'height', 768, 256, 2048, 32)
    if width * height > MAX_PIXELS:
        raise BadRequest('width x height must be at most 2048 x 2048 pixels')
    seed = payload.get('seed')
    if seed is None:
        seed = random.randrange(2**31)
    elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise BadRequest('seed must be a non-negative integer or null')
    references = payload.get('references') or []
    if not isinstance(references, list) or len(references) > MAX_REFERENCES:
        raise BadRequest(f'references must be a list of at most {MAX_REFERENCES} images')
    return {
        'prompt': prompt.strip(),
        'width': width,
        'height': height,
        'steps': int_field(payload, 'steps', 40, 8, 60),
        'seed': seed,
        'references': [decode_reference(i, value) for i, value in enumerate(references)],
        'quality': int_field(payload, 'quality', 92, 70, 100),
        'reference_resolution': int_field(payload, 'reference_resolution', 512, 256, 1024, 32),
    }


def handler(job):
    try:
        request = parse(job.get('input'))
    except BadRequest as error:
        return {'error': f'bad request: {error}'}
    started = time.time()
    generator = torch.Generator(device='cuda').manual_seed(request['seed'])
    # no_grad, not inference_mode: torchao's FP8 tensors cannot take inference tensors.
    try:
        with torch.no_grad():
            image = PIPE(
                prompt=request['prompt'],
                image=request['references'] or None,
                width=request['width'],
                height=request['height'],
                num_inference_steps=request['steps'],
                generator=generator,
                # Width and height are explicit, so this sizes the condition images only.
                output_resolution=request['reference_resolution'],
            ).images[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        refs = len(request['references'])
        return {'error': f"out of GPU memory at {request['width']}x{request['height']} with {refs} reference(s) on {GPU}"}
    generate_seconds = time.time() - started
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=request['quality'], optimize=True)
    cold = COLD.pop('pending', False)
    return {
        'version': 1,
        'format': 'jpeg',
        'image': base64.b64encode(buffer.getvalue()).decode('ascii'),
        'width': image.width,
        'height': image.height,
        'seed': request['seed'],
        'steps': request['steps'],
        'references': len(request['references']),
        'timings': {
            'loadSeconds': LOAD['loadSeconds'],
            'coldStart': cold,
            'generateSeconds': round(generate_seconds, 2),
        },
        'gpu': GPU,
        'placement': LOAD['placement'],
        'precision': LOAD['precision'],
        'freeVramGb': LOAD['freeVramGb'],
        'model': {'repo': MODEL_REPO, 'revision': LOAD['revision'], 'source': LOAD['source']},
    }


if __name__ == '__main__':
    import runpod

    runpod.serverless.start({'handler': handler})
