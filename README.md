# vidrush-qwen-image

A RunPod Serverless worker that draws stills with [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) for the vidrush app. One job draws one image, with up to 10 optional reference images.

- **Image:** `ghcr.io/jereviitasalo/vidrush-qwen-image:<version>`, built by GitHub Actions on every `v*` tag. It holds the code and two small step-distilled LoRAs (about 9 GB with CUDA PyTorch). Nobody needs Docker locally.
- **Weights:** not in the image. The endpoint attaches `Qwen/Qwen-Image-2.1` as a [RunPod cached model](https://docs.runpod.io/serverless/endpoints/model-caching), so the 33 GB is already on the host when a worker starts, and RunPod does not bill the download.
- **GPU:** 48 GB cards keep every weight resident, with the transformer in FP8 on Ada and newer. Smaller cards work with CPU offload, but they are slower.

The app creates the template and endpoint with the user's own RunPod API key (Settings → AI image generation → Qwen-Image). Each user pays for their own GPU time.

## Request

```json
{"input": {"version": 1, "prompt": "…", "width": 1344, "height": 768, "steps": 40, "seed": 7, "references": ["<base64 jpeg>"], "reference_resolution": 512}}
```

The reply holds `image` (base64 JPEG), `width`, `height`, `seed`, `variant`, `timings` (`loadSeconds`, `coldStart`, `swapSeconds`, `generateSeconds`) and `gpu`.

### Variants

`variant` picks the transformer. The LoRA variants ignore `steps` and run the sigma schedule their authors publish, with no CFG.

| variant | steps | LoRA |
|---|---|---|
| `base` | `steps` (default 40) | none |
| `viggle-6` | 6 | [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) v0.2.1 r256 |
| `pruna-8` | 8 | [PrunaAI/Pruna-Qwen-Image-2.1](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1) v0.1 |

The first job for a variant builds it: the worker reloads the base transformer, fuses that LoRA and quantizes it (`swapSeconds`, ~5-12 s, billed). Built variants stay on the GPU while there is room (base and one LoRA variant fit a 48 GB card in FP8), so switching back is free. `QWEN_PRELOAD` (comma list, e.g. `viggle-6`) builds variants at start; `QWEN_VARIANT` sets the default for jobs that do not name one.

## Release

```sh
git tag v1.0.0 && git push origin v1.0.0   # Actions builds and pushes ghcr.io/<owner>/vidrush-qwen-image:1.0.0
```

## License

The worker code is MIT. The LoRAs are distributed by their authors under the same Qwen Research License as the model. Qwen-Image 2.1 itself is distributed by Qwen under the Qwen Research License Agreement and is not included in this image. Qwen states that generated outputs are not part of the licensed Materials and that users retain the rights to the images they generate. Commercial use of the model requires a separate license from Qwen (model-business@notice.qwencloud.com). Each user is responsible for their own license.
