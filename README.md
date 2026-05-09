# HiDream_O1-ComfyUI

**HiDream O1 Image nodes for ComfyUI** — local HiDream O1 generation with text prompts, optional reference images, BF16/FP16/FP32/FP8 model loading, FlashAttention, SageAttention, preview updates, and ComfyUI DynamicVRAM/Aimdo integration.

[![HiDream O1 Model](https://img.shields.io/badge/HuggingFace-HiDream--O1--Image-blue)](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
[![GitHub](https://img.shields.io/badge/GitHub-Saganaki22%2FHiDream__O1--ComfyUI-black)](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## Features

- HiDream O1 Image generation directly inside ComfyUI
- Text-only and reference-image workflows
- Dynamic `image_1` to `image_12` inputs on the sampler node
- `keep_image1_aspect` toggle for reference-driven output aspect ratio
- BF16, FP16, FP32, FP8 E4M3FN, and FP8 E5M2 loader options
- FP8 mixed-weight loading using ComfyUI manual-cast style compute
- FlashAttention, SageAttention, and PyTorch SDPA attention backends
- Progress previews through ComfyUI's sampler progress bar
- ComfyUI model management, unload, DynamicVRAM, and Aimdo/VBAR support

## Installation

### Method 1: ComfyUI Manager

Search for `HiDream O1` or `HiDream_O1-ComfyUI` in ComfyUI Manager and install it.

### Method 2: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/HiDream_O1-ComfyUI.git
cd HiDream_O1-ComfyUI
python -m pip install -r requirements.txt
```

Restart ComfyUI after installing or updating.

**Suggested `transformers` version: 4.57.1 – 5.3** (newer versions may break compatibility).

## Model Setup

Download the complete model folder from one of the links below and place it inside `ComfyUI/models/diffusion_models/`:

| Precision | VRAM | Download |
|-----------|------|----------|
| BF16 / FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) |
| FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) |
| FP8 | ~10–11 GB | [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) |

> **Dev model quantized versions coming soon** — currently working on it.

**Example — FP8 (lowest VRAM):**

1. Go to [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
2. Download the **entire model folder** (all files, not just the safetensors)
3. Place it at `ComfyUI/models/diffusion_models/HiDream-O1-Image-fp8/`

The folder must contain the full Hugging Face support files:

```text
config.json
chat_template.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
model.safetensors
```

The original sharded format also works if the folder contains `model.safetensors.index.json` and all shard files.

If `download_if_missing` is enabled with an explicit precision, the loader uses or downloads the matching folder under `ComfyUI/models/diffusion_models`. With `precision=auto`, it downloads BF16 only when no local HiDream O1 folder exists.

| Precision | Hugging Face repo | Target folder |
|-----------|-------------------|---------------|
| `auto`, `bf16`, `fp32` | [`drbaph/HiDream-O1-Image-BF16`](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) | `HiDream-O1-Image-bf16` |
| `fp16` | [`drbaph/HiDream-O1-Image-FP16`](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) | `HiDream-O1-Image-fp16` |
| `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2` | [`drbaph/HiDream-O1-Image-FP8`](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) | `HiDream-O1-Image-fp8` |

## Nodes

### HiDream O1 Model Loader

Loads a local HiDream O1 model folder and returns a Comfy-managed model handle.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | first discovered folder | Complete HiDream O1 model folder |
| `precision` | `auto` | Detects safetensors dtype, or forces `bf16`, `fp16`, `fp32`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2` |
| `attention` | `auto` | `auto`, `flash`, `sdpa`, or `sage` |
| `download_if_missing` | `false` | With explicit precision, uses or downloads the matching drbaph BF16, FP16, or FP8 repo; with `auto`, downloads BF16 only when no local model exists |

### HiDream O1 Conditioning

Creates prompt conditioning for the sampler.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | cinematic portrait prompt | Text instruction for generation |
| `negative_prompt` | empty | Negative prompt used as the unconditional CFG branch in full mode when `guidance_scale` is above `1.0`; dev mode ignores CFG |

### HiDream O1 Sampler

Runs the model and outputs a ComfyUI `IMAGE`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `auto` | Uses `dev` settings if the model folder name contains `dev`, otherwise full settings |
| `width` | `2048` | Requested output width; internally snapped to a supported patch-aligned resolution |
| `height` | `2048` | Requested output height; internally snapped to a supported patch-aligned resolution |
| `steps` | `0` | `0` means auto: 50 for full, 28 for dev |
| `seed` | `42` | Random seed |
| `guidance_scale` | `5.0` | CFG scale for full mode; dev mode ignores CFG |
| `shift` | `-1.0` | `-1` means auto: 3.0 for full, 1.0 for dev |
| `noise_scale_start` | `7.5` | Initial noise scale |
| `noise_scale_end` | `7.5` | Final noise scale |
| `noise_clip_std` | `2.5` | Noise clipping standard deviation |
| `preview_every` | `4` | Sends a decoded preview every N steps; `0` disables previews |
| `keep_image1_aspect` | `false` | Only applies when `image_1` is connected |
| `force_offload` | `false` | Unloads the model immediately after generation |
| `image` | `0` | Dynamic reference image count, from `0` to `12` |

Reference image inputs are optional. Set `image` to `0` for text-only generation, or increase it to show `image_1`, `image_2`, and so on up to `image_12`.

## Precision Notes

`auto` detects the model storage dtype from the safetensors file. For native mixed FP8 folders, the large matrix weights should be `float8_e4m3fn` while small tensors such as norms and biases stay BF16/FP16.

Do not set `config.json` to `float8_e4m3fn`. Transformers may try to use FP8 as PyTorch's global default dtype, which fails. Keep config dtype as `bfloat16`; this node detects FP8 from the safetensors tensors themselves.

Use `fp8_e4m3fn_fast` to request ComfyUI's fast FP8 operation path where the current GPU, PyTorch, and ComfyUI build support it. Otherwise the model still stores large weights as FP8 and uses ComfyUI manual-cast style compute for safety.

## BF16 to FP8 Conversion

The converter should:

- Convert 2D and larger floating weight tensors to `torch.float8_e4m3fn`
- Keep 1D floating tensors such as norms, biases, and frequencies in BF16
- Keep non-floating tensors unchanged
- Keep `config.json` dtype as `bfloat16`

Example:

```bash
python convert_bf16_to_fp8.py ^
  --input "C:/path/to/HiDream-O1-Image-bf16/model.safetensors" ^
  --output-dir "C:/path/to/HiDream-O1-Image-fp8"
```

## Attention Backends

| Option | Description |
|--------|-------------|
| `auto` | Uses FlashAttention when available, otherwise SDPA |
| `flash` | Requires FlashAttention |
| `sage` | Requires the `sageattention` package |
| `sdpa` | Uses PyTorch scaled dot-product attention |

## DynamicVRAM and Unload

The loader wraps HiDream O1 in ComfyUI's `CoreModelPatcher`. When DynamicVRAM/Aimdo is enabled, ComfyUI should replace that with the dynamic patcher and page castable modules through VBAR. The sampler's `force_offload` option calls ComfyUI unload/cleanup after generation.

If a model still appears fully static in memory visualization, confirm ComfyUI started with DynamicVRAM enabled and that the model is loaded with a precision where weight dtype and compute dtype differ, such as FP8 storage with FP16/BF16 compute.

## Troubleshooting

### `couldn't find storage object Float8_e4m3fnStorage`

Your model `config.json` probably says `float8_e4m3fn`. Change both root `dtype` and `text_config.dtype` back to `bfloat16`. FP8 is detected from the safetensors file, not config.

### `NoneType` errors under DynamicVRAM/Aimdo

DynamicVRAM can temporarily eject weights from modules and replace them with paged VBAR storage. Update this nodepack and restart ComfyUI so the model code uses archived Comfy dtype metadata instead of directly reading ejected weights.

### Only grey blocks in MemoryVisualization

Grey means unloaded pages. Orange/yellow page grids require Aimdo/VBAR to be active and the model to be using Comfy castable modules. FP8 mixed loading should show dynamic staging when ComfyUI DynamicVRAM is enabled.

## Links

- BF16 model: [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16)
- FP16 model: [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16)
- FP8 model: [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
- Upstream project: [HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image)
- Node repository: [Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## License

This custom node is released under the MIT License. The HiDream O1 model has its own license and usage terms; check the upstream Hugging Face model page before redistribution or commercial use.
