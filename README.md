# vidrush-qwen-image

A RunPod Serverless worker that draws stills with [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) for the vidrush app. One job draws one image, with up to 10 optional reference images.

- **Image:** `ghcr.io/jereviitasalo/vidrush-qwen-image:<version>`, built by GitHub Actions on every `v*` tag. It holds the code only (about 7 GB with CUDA PyTorch). Nobody needs Docker locally.
- **Weights:** not in the image. The endpoint attaches `Qwen/Qwen-Image-2.1` as a [RunPod cached model](https://docs.runpod.io/serverless/endpoints/model-caching), so the 33 GB is already on the host when a worker starts, and RunPod does not bill the download.
- **GPU:** 48 GB cards keep every weight resident in bf16. Smaller cards work with CPU offload, but they are slower.

The app creates the template and endpoint with the user's own RunPod API key (Settings → AI image generation → Qwen-Image). Each user pays for their own GPU time.

## Request

```json
{"input": {"version": 1, "prompt": "…", "width": 1344, "height": 768, "steps": 40, "seed": 7, "references": ["<base64 jpeg>"]}}
```

The reply holds `image` (base64 JPEG), `width`, `height`, `seed`, `timings` (`loadSeconds`, `coldStart`, `generateSeconds`) and `gpu`.

## Release

```sh
git tag v1.0.0 && git push origin v1.0.0   # Actions builds and pushes ghcr.io/<owner>/vidrush-qwen-image:1.0.0
```

## License

The worker code is MIT. Qwen-Image 2.1 itself is distributed by Qwen under the Qwen Research License Agreement and is not included in this image. Qwen states that generated outputs are not part of the licensed Materials and that users retain the rights to the images they generate. Commercial use of the model requires a separate license from Qwen (model-business@notice.qwencloud.com). Each user is responsible for their own license.
