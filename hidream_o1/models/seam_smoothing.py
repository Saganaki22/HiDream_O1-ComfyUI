from __future__ import annotations

import math

import einops
import torch

SHIFT_MODES = ["static", "rotate", "all"]
SCHEDULES = ["constant", "linear", "cosine", "front_loaded", "late"]


def get_shift_offsets(h_patches: int, w_patches: int, smoothing_step_idx: int, mode: str = "rotate"):
    base_patterns = [
        (max(1, h_patches // 2), max(1, w_patches // 2)),
        (max(1, h_patches // 3), max(1, w_patches // 3)),
        (max(1, h_patches // 4), 0),
        (0, max(1, w_patches // 4)),
    ]
    if mode == "static":
        return [base_patterns[0]]
    if mode == "all":
        return base_patterns
    return [base_patterns[smoothing_step_idx % len(base_patterns)]]


def get_smoothing_strength(
    base_strength: float,
    smoothing_step_idx: int,
    total_smoothing_steps: int,
    schedule: str = "constant",
) -> float:
    if total_smoothing_steps <= 1:
        return base_strength

    t = smoothing_step_idx / (total_smoothing_steps - 1)
    if schedule == "linear":
        return base_strength * (1.0 - t)
    if schedule == "cosine":
        return base_strength * (0.5 * (1.0 + math.cos(math.pi * t)))
    if schedule == "front_loaded":
        return base_strength * (1.0 - t) ** 2
    if schedule == "late":
        return base_strength * t
    return base_strength


def estimate_seam_intensity(z: torch.Tensor, h_patches: int, w_patches: int) -> float:
    z_img = einops.rearrange(z.float(), "B (H W) C -> B C H W", H=h_patches, W=w_patches)
    dx = (z_img[..., 1:] - z_img[..., :-1]).abs().mean()
    dy = (z_img[..., 1:, :] - z_img[..., :-1, :]).abs().mean()
    return ((dx + dy) / 2.0).item()


def coherence_delta(z: torch.Tensor, x_pred_unshifted: torch.Tensor) -> float:
    return (x_pred_unshifted.float() - z.float()).abs().mean().item()


def apply_seam_smoothing(
    z: torch.Tensor,
    samples: list[dict],
    ref_patches: torch.Tensor | None,
    t_pixeldit: torch.Tensor,
    sigma: torch.Tensor,
    dtype: torch.dtype,
    h_patches: int,
    w_patches: int,
    smoothing_step_idx: int,
    total_smoothing_steps: int,
    base_strength: float,
    schedule: str,
    shift_mode: str,
    forward_once,
    guidance_scale: float,
    multiscale: bool = False,
    cfg_aware: bool = False,
    adaptive_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    info = {
        "strength_used": 0.0,
        "coherence_delta": 0.0,
        "seam_intensity": 0.0,
        "n_forwards": 0,
        "skipped": False,
        "offsets": [],
    }

    if adaptive_threshold and adaptive_threshold > 0.0:
        intensity = estimate_seam_intensity(z, h_patches, w_patches)
        info["seam_intensity"] = intensity
        if intensity < adaptive_threshold:
            info["skipped"] = True
            return z, info

    strength = get_smoothing_strength(base_strength, smoothing_step_idx, total_smoothing_steps, schedule)
    info["strength_used"] = strength
    if strength <= 0.0:
        info["skipped"] = True
        return z, info

    offsets = list(get_shift_offsets(h_patches, w_patches, smoothing_step_idx, shift_mode))
    if multiscale:
        fine = (max(1, h_patches // 8), max(1, w_patches // 8))
        if fine not in offsets:
            offsets.append(fine)
    info["offsets"] = offsets

    z_img = einops.rearrange(z, "B (H W) C -> B C H W", H=h_patches, W=w_patches)
    accumulated = None
    deltas = []
    n_forwards = 0
    use_cfg = cfg_aware and len(samples) > 1 and guidance_scale > 1.0

    for shift_h, shift_w in offsets:
        z_shifted = torch.roll(z_img, shifts=(shift_h, shift_w), dims=(2, 3))
        z_s = einops.rearrange(z_shifted, "B C H W -> B (H W) C")

        if ref_patches is None:
            x_pred_cond_s = forward_once(samples[0], z_s.clone(), t_pixeldit)
        else:
            x_pred_cond_s = forward_once(samples[0], torch.cat([z_s, ref_patches], dim=1), t_pixeldit)
        n_forwards += 1

        if use_cfg:
            if ref_patches is None:
                x_pred_uncond_s = forward_once(samples[1], z_s.clone(), t_pixeldit)
            else:
                x_pred_uncond_s = forward_once(samples[1], torch.cat([z_s, ref_patches], dim=1), t_pixeldit)
            n_forwards += 1

            v_cond_s = (x_pred_cond_s.float() - z_s.float()) / sigma
            v_uncond_s = (x_pred_uncond_s.float() - z_s.float()) / sigma
            v_guided_s = v_uncond_s + guidance_scale * (v_cond_s - v_uncond_s)
            x_pred_s = (z_s.float() + v_guided_s * sigma).to(dtype)
        else:
            x_pred_s = x_pred_cond_s.to(dtype)

        x_pred_s_img = einops.rearrange(x_pred_s, "B (H W) C -> B C H W", H=h_patches, W=w_patches)
        x_unshifted = torch.roll(x_pred_s_img, shifts=(-shift_h, -shift_w), dims=(2, 3))
        x_unshifted = einops.rearrange(x_unshifted, "B C H W -> B (H W) C")

        deltas.append(coherence_delta(z, x_unshifted))
        accumulated = x_unshifted if accumulated is None else accumulated + x_unshifted

    blended_pred = accumulated / float(len(offsets))
    z_new = ((1.0 - strength) * z + strength * blended_pred).to(dtype)

    info["coherence_delta"] = sum(deltas) / len(deltas) if deltas else 0.0
    info["n_forwards"] = n_forwards
    return z_new, info
