# HiDream O1 LoRA Training Notes

These notes describe the ComfyUI LoRA trainer added for HiDream O1 and the defaults chosen to match AI Toolkit's working O1 implementation.

## What This Trainer Does

HiDream O1 is a Qwen3-VL style model used as a pixel-level diffusion transformer. Instead of using a VAE latent UNet, the trainer patchifies RGB images directly, adds flow-matching noise, feeds the noisy image patches through the model's vision path, and trains LoRA layers on the model prediction.

The model returns an x0 prediction. AI Toolkit converts that x0 prediction into a flow velocity before applying the loss. This trainer now follows the same idea:

```text
scaled_noise = noise * noise_scale
noisy_image = (1 - sigma) * image + sigma * scaled_noise
velocity_pred = (noisy_image - x0_pred) / sigma
velocity_target = scaled_noise - image
loss = mse(velocity_pred, velocity_target)
```

Default `noise_scale` is `8.0`, matching AI Toolkit's HiDream O1 model kwargs.

## Use Full For Training

Use a Full model for LoRA training:

- `HiDream-O1-Image-BF16` is the default Comfy training choice.
- `HiDream-O1-Image` is the official upstream full BF16 folder choice.
- `HiDream-O1-Image-FP8` can be used when VRAM is tight, with BF16 compute where supported.

Dev and Dev-2604 are intentionally not exposed in the training node. They are distilled inference variants and can train unpredictably. Upstream also recommends the Full model for editing tasks.

## Dataset Layout

Create a folder with image files and matching `.txt` captions:

```text
my_dataset/
  img_0001.png
  img_0001.txt
  img_0002.jpg
  img_0002.txt
```

Run:

```text
HiDream O1 Dataset Maker -> HiDream O1 Train Config -> HiDream O1 LoRA Trainer
```

The Dataset Maker writes a JSONL manifest that the trainer consumes.

## Recommended Subject Captions

For a person LoRA, keep captions consistent and direct:

```text
photo of m0n0y0 person, close-up portrait, curly black hair, city street background
photo of m0n0y0 person, upper body portrait, neutral expression, natural light
photo of m0n0y0 person, candid portrait, black jacket, outdoor background
```

Use a unique trigger token that does not look like a common word. Put it early in every caption. Avoid captions that over-describe saturation, high contrast, harsh lighting, heavy cinematic grading, or stylized color if you want identity rather than baked-in style.

## Defaults

The current defaults are intended to match AI Toolkit's O1 path:

| Setting | Default | Why |
| --- | --- | --- |
| `base_model_name` | `HiDream-O1-Image-BF16` | Full model training, local converted BF16 default |
| `learning_rate` | `0.0001` | AI Toolkit job default |
| `lora_rank` | `32` | AI Toolkit linear LoRA default |
| `lora_alpha` | `32` | AI Toolkit linear alpha default |
| `target_preset` | `aitoolkit` | Train linear-like layers except `lm_head`, `patch_embed`, `visual` |
| `loss_target` | `velocity` | Match AI Toolkit x0-to-flow-velocity loss |
| `noise_scale` | `8.0` | Match AI Toolkit HiDream O1 noise scale |
| `timestep_type` | `linear` | AI Toolkit O1 override |
| `min_sigma` / `max_sigma` | `0.001` / `0.999` | Broad flow-matching timestep coverage |
| `max_loss` | `1.0` | Cap loss spikes like AI Toolkit |
| `weight_decay` | `0.0001` | AI Toolkit AdamW default |
| `warmup_steps` | `0` | AI Toolkit default scheduler behavior |
| `resolution` | `1024` | Practical direct-pixel training size |
| `caption_dropout` | `0.05` | AI Toolkit dataset default |
| `save_every_steps` | `250` | Frequent checkpoint comparison |

## When To Stop

Do not judge from one final checkpoint only. Save at least every 250 steps and sample several strengths:

```text
step 250: strengths 0.5, 0.7, 1.0
step 500: strengths 0.5, 0.7, 1.0
step 750: strengths 0.5, 0.7, 1.0
step 1000: strengths 0.5, 0.7, 1.0
```

For small person datasets, useful identity often appears between 500 and 1500 steps. If the image gets oversaturated, overly contrasty, or starts copying dataset lighting too strongly, stop earlier or lower inference strength.

## Troubleshooting

If the LoRA does not resemble the person:

- Make sure every caption contains the same trigger token.
- Use clearer face crops and varied angles.
- Keep the training target at `aitoolkit`; `attention` only can be too weak for identity.
- Try more steps before changing many settings.
- Sample with simpler prompts first, such as `photo of m0n0y0 person, portrait`.

If the LoRA is overbaked:

- Test earlier checkpoints.
- Lower LoRA strength at inference to `0.5` or `0.7`.
- Remove strong color/lighting words from captions.
- Lower learning rate to `5e-5`.
- Avoid training on many near-duplicate images.

If colors blow out immediately:

- Confirm `loss_target=velocity`.
- Confirm `noise_scale=8.0`.
- Confirm you restarted ComfyUI after updating the node.
- Recreate the Train Config node so old workflow defaults are not stuck in the graph.

## Current Limitations

- Batch size is currently effectively one image per micro-step.
- No resume training node yet.
- No reference-image/edit/IP LoRA training path yet.
- No validation sampler is run during training; compare saved checkpoints manually.
- The trainer runs in-process and blocks the ComfyUI queue.

## Dev-2604 Notes

Dev-2604 is available in the model loader for inference:

- `HiDream-O1-Image-Dev-2604`
- `HiDream-O1-Image-Dev-2604-BF16`
- `HiDream-O1-Image-Dev-2604-FP16`
- `HiDream-O1-Image-Dev-2604-FP8`

For Dev edit mode with exactly one reference image, the sampler defaults to `flow_match`, matching the May 13, 2026 upstream scheduler update. Text/subject Dev paths still use the fixed 28-step flash schedule by default.
