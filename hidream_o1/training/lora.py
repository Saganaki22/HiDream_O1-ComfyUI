from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}
PIXEL_TARGET_SUFFIXES = ("x_embedder.proj1", "x_embedder.proj2", "final_layer2.linear")
AITOOLKIT_IGNORE_SUBSTRINGS = ("lm_head", "patch_embed", "visual")


class HiDreamO1LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        lora_key: str,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        self.lora_key = lora_key
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        device = getattr(base.weight, "device", None)
        self.lora_down = nn.Parameter(torch.empty(self.rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_dtype = base_out.dtype if torch.is_floating_point(base_out) else x.dtype
        x_lora = self.dropout(x).to(lora_dtype)
        down = self.lora_down.to(device=x_lora.device, dtype=lora_dtype)
        up = self.lora_up.to(device=x_lora.device, dtype=lora_dtype)
        lora_out = F.linear(F.linear(x_lora, down), up)
        return base_out + lora_out.to(base_out.dtype) * self.scaling

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_down
        yield self.lora_up


@dataclass
class LoRAInjectionResult:
    layers: list[HiDreamO1LoRALinear]
    skipped: list[str]


def _get_parent_and_leaf(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return root, module_name
    parent_name, leaf = module_name.rsplit(".", 1)
    return root.get_submodule(parent_name), leaf


def _normal_lora_key(module_name: str) -> str:
    for prefix in ("model.model.", "model."):
        if module_name.startswith(prefix):
            module_name = module_name[len(prefix):]
    return module_name


def _is_decoder_attention(name: str, leaf: str) -> bool:
    return "language_model.layers." in name and ".self_attn." in name and leaf in ATTENTION_TARGETS


def _is_decoder_mlp(name: str, leaf: str) -> bool:
    return "language_model.layers." in name and ".mlp." in name and leaf in MLP_TARGETS


def _is_pixel_head(name: str) -> bool:
    return name.endswith(PIXEL_TARGET_SUFFIXES)


def _ignored_by_aitoolkit(name: str) -> bool:
    return any(part in name for part in AITOOLKIT_IGNORE_SUBSTRINGS)


def _wanted_module(name: str, leaf: str, target_preset: str) -> bool:
    target_preset = (target_preset or "attention+pixel").lower()
    if target_preset in {"aitoolkit", "ai-toolkit", "ostris"}:
        return not _ignored_by_aitoolkit(name)
    if name.startswith("model.visual.") or ".visual." in name:
        return False
    if _is_decoder_attention(name, leaf):
        return True
    if "mlp" in target_preset and _is_decoder_mlp(name, leaf):
        return True
    if "pixel" in target_preset and _is_pixel_head(name):
        return True
    return False


def _is_linear_like(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Linear)
        or module.__class__.__name__.endswith("Linear")
        or (
            hasattr(module, "in_features")
            and hasattr(module, "out_features")
            and hasattr(module, "weight")
        )
    )


def inject_lora_layers(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_preset: str,
) -> LoRAInjectionResult:
    layers: list[HiDreamO1LoRALinear] = []
    skipped: list[str] = []

    for name, module in list(root.named_modules()):
        if not _is_linear_like(module):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if not _wanted_module(name, leaf, target_preset):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            skipped.append(name)
            continue

        parent, parent_leaf = _get_parent_and_leaf(root, name)
        lora_key = _normal_lora_key(name)
        wrapper = HiDreamO1LoRALinear(
            module,
            lora_key=lora_key,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, parent_leaf, wrapper)
        layers.append(wrapper)

    return LoRAInjectionResult(layers=layers, skipped=skipped)


def lora_parameters(layers: list[HiDreamO1LoRALinear]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in layers:
        params.extend(layer.trainable_parameters())
    return params


def lora_state_dict(
    layers: list[HiDreamO1LoRALinear],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer in layers:
        prefix = f"diffusion_model.{layer.lora_key}"
        state[f"{prefix}.lora_down.weight"] = layer.lora_down.detach().to("cpu", dtype=dtype)
        state[f"{prefix}.lora_up.weight"] = layer.lora_up.detach().to("cpu", dtype=dtype)
        state[f"{prefix}.alpha"] = torch.tensor(float(layer.alpha), dtype=dtype)
    return state
