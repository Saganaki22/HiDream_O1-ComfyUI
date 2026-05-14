from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from ..comfy_runtime import (
    canonical_model_choice,
    compute_dtype_from_weight_dtype,
    find_existing_canonical_model,
    is_float8_dtype,
    load_hidream_model,
    maybe_download_model,
    resolve_model_name,
)
from ..models.pipeline import PATCH_SIZE, T_EPS, build_t2i_text_sample
from ..models.utils import resize_pilimage
from .dataset import load_image_caption_manifest
from .lora import inject_lora_layers, lora_parameters, lora_state_dict

LOGGER = logging.getLogger("HiDream_O1")

SAVE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_training_model_path(base_model_name: str, download_if_missing: bool) -> tuple[Path, str]:
    canonical_choice = canonical_model_choice(base_model_name)
    if canonical_choice is None:
        return resolve_model_name(base_model_name), "auto"

    model_variant, model_precision = canonical_choice
    model_dir = find_existing_canonical_model(base_model_name)
    if model_dir is None:
        if not download_if_missing:
            raise FileNotFoundError(
                f"{base_model_name} is not installed. Enable download_if_missing or place it in models/diffusion_models."
            )
        model_dir = maybe_download_model(precision=model_precision, model_variant=model_variant)
    return model_dir, model_precision


def clean_output_name(output_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in output_name.strip())
    return cleaned or "hidream_o1_lora"


def _save_dtype(name: str) -> torch.dtype:
    return SAVE_DTYPES.get((name or "bf16").lower(), torch.bfloat16)


def _set_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    try:
        qwen_model = model.model.model
        qwen_model.language_model.gradient_checkpointing = bool(enabled)
        for layer in qwen_model.language_model.layers:
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = bool(enabled)
    except Exception as exc:
        LOGGER.warning("Could not set HiDream O1 gradient checkpointing: %s", exc)


def _freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _image_to_patches(path: str, resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int, int]:
    image = Image.open(path).convert("RGB")
    image = resize_pilimage(image, int(resolution), PATCH_SIZE)
    width, height = image.size
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    patches = einops.rearrange(
        tensor,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=PATCH_SIZE,
        p2=PATCH_SIZE,
    )
    return patches.to(device=device, dtype=dtype), height, width


def _to_device(sample: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}


def _sample_sigma(
    batch_size: int,
    device: torch.device,
    shift: float,
    min_sigma: float,
    max_sigma: float,
    timestep_type: str,
) -> torch.Tensor:
    min_sigma = max(0.0001, min(0.9999, float(min_sigma)))
    max_sigma = max(min_sigma + 0.0001, min(0.9999, float(max_sigma)))
    timestep_type = (timestep_type or "linear").lower()

    if timestep_type == "sigmoid":
        sigma = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
    else:
        sigma = torch.rand(batch_size, device=device, dtype=torch.float32)

    sigma = min_sigma + (max_sigma - min_sigma) * sigma
    if timestep_type == "shift" and shift and shift > 0 and not math.isclose(shift, 1.0):
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma.clamp(0.0001, 0.9999)


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "cpu"} and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device.type, dtype=dtype, cache_enabled=False)
    return contextlib.nullcontext()


def _write_training_info(
    output_dir: str,
    *,
    base_model_name: str,
    model_dir: Path,
    precision: str,
    attention_backend: str,
    train_config: dict,
    output_name: str,
) -> None:
    info = {
        "base_model": base_model_name,
        "model_dir": str(model_dir),
        "precision": precision,
        "attention_backend": attention_backend,
        "output_name": output_name,
        "train_config": train_config,
    }
    with open(os.path.join(output_dir, "hidream_o1_lora_config.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def _save_checkpoint(layers, output_dir: str, output_name: str, step: int, save_dtype_name: str) -> str:
    path = os.path.join(output_dir, f"{output_name}_step_{step}.safetensors")
    save_file(lora_state_dict(layers, dtype=_save_dtype(save_dtype_name)), path)
    return path


@torch.inference_mode(False)
def run_hidream_o1_lora_training(
    *,
    base_model_name: str,
    precision: str,
    attention: str,
    train_config: dict,
    dataset_path: str,
    output_dir: str,
    output_name: str,
    max_steps: int,
    save_every_steps: int,
    num_workers: int,
    download_if_missing: bool,
) -> str:
    previous_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    output_name = clean_output_name(output_name)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_dir, canonical_precision = resolve_training_model_path(base_model_name, download_if_missing)
    if precision == "auto" and canonical_precision != "auto":
        precision = canonical_precision

    train_config = dict(train_config or {})
    resolution = int(train_config.get("resolution", 1024))
    grad_accum_steps = max(1, int(train_config.get("grad_accum_steps", 1)))
    caption_dropout = float(train_config.get("caption_dropout", 0.0))
    timestep_shift = float(train_config.get("timestep_shift", 3.0))
    timestep_type = str(train_config.get("timestep_type", "linear"))
    min_sigma = float(train_config.get("min_sigma", 0.001))
    max_sigma = float(train_config.get("max_sigma", 0.999))
    noise_scale = float(train_config.get("noise_scale", 8.0))
    loss_target = str(train_config.get("loss_target", "velocity")).lower()
    max_loss = float(train_config.get("max_loss", 1.0))
    target_preset = str(train_config.get("target_preset", "aitoolkit"))
    save_dtype_name = str(train_config.get("save_dtype", "bf16"))

    samples = load_image_caption_manifest(dataset_path)
    loader = DataLoader(
        samples,
        batch_size=1,
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        drop_last=False,
    )
    iterator = iter(loader)

    model_management.unload_all_models()
    model_management.soft_empty_cache()

    handle = None
    try:
        handle = load_hidream_model(model_dir, precision=precision, attention=attention)
        model_management.load_models_gpu([handle.patcher])
        model = handle.patcher.model
        device = model.device
        attention_backend = handle.resolve_attention_backend()

        _freeze_all(model)
        _set_gradient_checkpointing(model, bool(train_config.get("gradient_checkpointing", True)))
        model.train()

        if is_float8_dtype(handle.weight_dtype):
            compute_dtype = compute_dtype_from_weight_dtype(handle.weight_dtype)
        else:
            compute_dtype = handle.dtype
        for module in model.modules():
            module.hidream_compute_dtype = compute_dtype

        injection = inject_lora_layers(
            model,
            rank=int(train_config.get("lora_rank", 32)),
            alpha=float(train_config.get("lora_alpha", 32)),
            dropout=float(train_config.get("lora_dropout", 0.0)),
            target_preset=target_preset,
        )
        if not injection.layers:
            raise RuntimeError(f"No HiDream O1 LoRA target layers found for target_preset={target_preset!r}.")
        if injection.skipped:
            LOGGER.warning("Skipped %s non-linear LoRA target candidates.", len(injection.skipped))
        LOGGER.info("Injected HiDream O1 LoRA into %s layers.", len(injection.layers))

        params = lora_parameters(injection.layers)
        optimizer = AdamW(
            params,
            lr=float(train_config.get("learning_rate", 1e-4)),
            weight_decay=float(train_config.get("weight_decay", 1e-4)),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_config.get("warmup_steps", 0)),
            num_training_steps=max(1, int(max_steps)),
        )

        _write_training_info(
            output_dir,
            base_model_name=base_model_name,
            model_dir=model_dir,
            precision=precision,
            attention_backend=attention_backend,
            train_config=train_config,
            output_name=output_name,
        )

        processor = handle.processor
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        pbar = ProgressBar(max_steps)
        pbar.update(0)

        for step in range(int(max_steps)):
            model_management.throw_exception_if_processing_interrupted()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for _micro_step in range(grad_accum_steps):
                batch, iterator = _next_batch(iterator, loader)
                image_path = batch["image"][0]
                caption = batch["caption"][0]
                if caption_dropout > 0 and random.random() < caption_dropout:
                    caption = " "

                clean_patches, height, width = _image_to_patches(image_path, resolution, device, compute_dtype)
                text_sample = build_t2i_text_sample(caption, height, width, tokenizer, processor, model.config)
                text_sample = _to_device(text_sample, device)

                sigma = _sample_sigma(1, device, timestep_shift, min_sigma, max_sigma, timestep_type).to(dtype=compute_dtype)
                sigma_view = sigma.view(1, 1, 1)
                noise = torch.randn(clean_patches.shape, device=device, dtype=compute_dtype)
                scaled_noise = noise * noise_scale
                noisy_patches = (1.0 - sigma_view) * clean_patches + sigma_view * scaled_noise
                timestep = (1.0 - sigma).to(device=device, dtype=torch.float32)

                with _autocast_context(device, compute_dtype):
                    outputs = model(
                        input_ids=text_sample["input_ids"],
                        position_ids=text_sample["position_ids"],
                        vinputs=noisy_patches,
                        timestep=timestep,
                        token_types=text_sample["token_types"],
                        use_flash_attn=attention_backend == "flash",
                        use_sage_attn=attention_backend == "sage",
                    )
                    x0_pred = outputs.x_pred[0, text_sample["vinput_mask"][0]].unsqueeze(0)
                    if loss_target == "x0":
                        raw_loss = F.mse_loss(x0_pred.float(), clean_patches.float())
                    else:
                        sigma_loss = sigma_view.float().clamp_min(T_EPS)
                        velocity_pred = (noisy_patches.float() - x0_pred.float()) / sigma_loss
                        velocity_target = scaled_noise.float() - clean_patches.float()
                        raw_loss = F.mse_loss(velocity_pred, velocity_target)
                    if max_loss > 0:
                        raw_loss = torch.clamp(raw_loss, max=max_loss)
                    loss = raw_loss / grad_accum_steps

                if loss.grad_fn is None:
                    raise RuntimeError("HiDream O1 loss has no grad_fn. LoRA parameters are not receiving gradients.")
                loss.backward()
                total_loss += float(raw_loss.detach().cpu())

            max_grad_norm = float(train_config.get("max_grad_norm", 1.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            pbar.update(1)

            if step % 10 == 0:
                LOGGER.info(
                    "HiDream O1 LoRA step %s/%s loss=%.5f lr=%.8f",
                    step,
                    max_steps,
                    total_loss,
                    optimizer.param_groups[0]["lr"],
                )

            step_num = step + 1
            if step_num % int(save_every_steps) == 0 or step_num == int(max_steps):
                save_path = _save_checkpoint(injection.layers, output_dir, output_name, step_num, save_dtype_name)
                LOGGER.info("Saved HiDream O1 LoRA checkpoint: %s", save_path)

    finally:
        if handle is not None:
            try:
                handle.dispose()
            except Exception as exc:
                LOGGER.warning("HiDream O1 training cleanup warning: %s", exc)
        model_management.soft_empty_cache()
        torch.set_grad_enabled(previous_grad_state)

    return output_dir
