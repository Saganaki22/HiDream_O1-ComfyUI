# HiDream_O1-ComfyUI

**HiDream O1 Image nodes for ComfyUI** — local HiDream O1 generation with text prompts, optional reference images, BF16/FP16/FP32/FP8 model loading, FlashAttention, SageAttention, preview updates, and ComfyUI DynamicVRAM/Aimdo integration.

[![HiDream O1 Model](https://img.shields.io/badge/HuggingFace-HiDream--O1--Image-blue)](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
[![Demo](https://img.shields.io/badge/Demo-HiDream--O1--Image-green)](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
[![GitHub](https://img.shields.io/badge/GitHub-Saganaki22%2FHiDream__O1--ComfyUI-black)](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

[中文文档](README_ZH.md)

<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/adfcbb51-6e04-4daf-82cf-99b2052f32de" />


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


<img width="2010" height="899" alt="image" src="https://github.com/user-attachments/assets/116f408b-dcce-4e01-b8e9-93566f8a2cca" />


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
| Full BF16 | ~18–20 GB | [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) |
| Full FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) |
| Full FP8 | ~10–11 GB | [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) |
| Dev BF16 | ~18–20 GB | [drbaph/HiDream-O1-Image-Dev-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16) |
| Dev FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-Dev-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16) |
| Dev FP8 | ~10–11 GB | [drbaph/HiDream-O1-Image-Dev-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8) |

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

The model loader always shows the six built-in model choices: Full/Dev BF16, FP16, and FP8. If the selected model already exists locally, it is used. If it is missing, enable `download_if_missing` and the selected model will be downloaded into `ComfyUI/models/diffusion_models`.

Local folder matching is case-insensitive, so `HiDream-O1-Image-Dev-FP8`, `hidream-o1-image-dev-fp8`, and the default target folder casing all resolve to the same built-in choice. The loader dropdown only shows the built-in HiDream O1 model choices.

### Upstream Artifact Note

The original/full HiDream O1 model can show grid artifacts or other reference-image artifacts. In the upstream issue tracker, a HiDream developer recommends trying the Dev model because it should have fewer grid artifacts, and notes that reference-image generation is still being improved: [HiDream-ai/HiDream-O1-Image issue #1](https://github.com/HiDream-ai/HiDream-O1-Image/issues/1#issuecomment-4412738522).

| Variant | Precision | Hugging Face repo | Target folder |
|---------|-----------|-------------------|---------------|
| Full | `auto`, `bf16`, `fp32` | [`drbaph/HiDream-O1-Image-BF16`](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) | `HiDream-O1-Image-bf16` |
| Full | `fp16` | [`drbaph/HiDream-O1-Image-FP16`](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) | `HiDream-O1-Image-fp16` |
| Full | `fp8_e4m3fn`, `fp8_e5m2` | [`drbaph/HiDream-O1-Image-FP8`](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) | `HiDream-O1-Image-fp8` |
| Dev | `auto`, `bf16`, `fp32` | [`drbaph/HiDream-O1-Image-Dev-BF16`](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16) | `HiDream-O1-Image-Dev-bf16` |
| Dev | `fp16` | [`drbaph/HiDream-O1-Image-Dev-FP16`](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16) | `HiDream-O1-Image-Dev-fp16` |
| Dev | `fp8_e4m3fn`, `fp8_e5m2` | [`drbaph/HiDream-O1-Image-Dev-FP8`](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8) | `HiDream-O1-Image-Dev-fp8` |

## Nodes

### HiDream O1 Model Loader

Loads a local HiDream O1 model folder and returns a Comfy-managed model handle.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `HiDream-O1-Image-BF16` | Built-in HiDream O1 model choice |
| `precision` | `auto` | Detects safetensors dtype, or forces `bf16`, `fp16`, `fp32`, `fp8_e4m3fn`, `fp8_e5m2` |
| `attention` | `auto` | `auto`, `flash`, `sdpa`, or `sage` |
| `download_if_missing` | `false` | Downloads the selected built-in model if it is not installed locally |

### HiDream O1 Conditioning

Creates prompt conditioning for the sampler.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | cinematic portrait prompt | Text instruction for generation |
| `negative_prompt` | empty | Negative prompt used as the unconditional CFG branch in full mode when `guidance_scale` is above `1.0`; dev mode ignores CFG |

### HiDream O1 LoRA

Applies a LoRA between the model loader and sampler:

```text
HiDream O1 Model Loader -> HiDream O1 LoRA -> HiDream O1 Sampler
```

The LoRA dropdown reads from `ComfyUI/models/lora/`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_name` | `None` when no LoRAs are found | LoRA file |
| `strength` | `1.0` | Model strength from `-10.0` to `10.0`; `0` disables the LoRA |

### HiDream O1 Sampler

Runs the model and outputs a ComfyUI `IMAGE`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `auto` | Uses `dev` settings if the model folder name contains `dev`, otherwise full settings |
| `width` | `2048` | Requested output width; internally snapped to a supported patch-aligned resolution |
| `height` | `2048` | Requested output height; internally snapped to a supported patch-aligned resolution |
| `steps` | `0` | `0` means auto: 50 for full; dev always uses the upstream fixed 28-step schedule |
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

The loader exposes the normal FP8 options only.

## Scheduler

The sampler automatically picks the scheduler based on model type:

| Model type | Scheduler | Notes |
|------------|-----------|-------|
| Full (`auto`) | `FlowUniPCMultistepScheduler` | Higher-order solver, generates more detail |
| Dev | `FlashFlowMatchEulerDiscreteScheduler` | Custom Euler with built-in noise scaling, tuned for fewer steps |

When `model_type` is `auto`, the folder name is checked for `dev` — if not found, the full model path is used with UniPC.

Dev follows the upstream recipe: fixed 28-step timetable, guidance `0.0`, shift `1.0`, and noise defaults `7.5 / 7.5 / 2.5`. If dev images look noisy or oddly colored, reset `noise_scale_start`, `noise_scale_end`, and `noise_clip_std` to those defaults and use the `flash` or `auto` attention backend.

## Attention Backends

| Option | Description |
|--------|-------------|
| `auto` | Uses FlashAttention when available, otherwise SDPA |
| `flash` | Requires FlashAttention |
| `sage` | Requires the `sageattention` package |
| `sdpa` | Uses PyTorch scaled dot-product attention |

## Links

- Demo: [HiDream-O1-Image](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
- BF16 model: [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16)
- FP16 model: [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16)
- FP8 model: [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
- Dev BF16 model: [drbaph/HiDream-O1-Image-Dev-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16)
- Dev FP16 model: [drbaph/HiDream-O1-Image-Dev-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16)
- Dev FP8 model: [drbaph/HiDream-O1-Image-Dev-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8)
- Upstream project: [HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image)
- Node repository: [Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## License

This custom node is released under the MIT License. The HiDream O1 model has its own license and usage terms; check the upstream Hugging Face model page before redistribution or commercial use.
