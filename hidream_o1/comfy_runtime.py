from __future__ import annotations

import gc
import errno
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.lora
import comfy.lora_convert
import comfy.ops
import comfy.utils
import folder_paths

from .models import qwen3_vl_transformers
from .models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

LOGGER = logging.getLogger("HiDream_O1")
NO_LORA_NAME = "None"

FLOAT8_DTYPE_NAMES = {
    getattr(torch, "float8_e4m3fn", None): "float8_e4m3fn",
    getattr(torch, "float8_e5m2", None): "float8_e5m2",
}
FLOAT8_DTYPE_NAMES = {k: v for k, v in FLOAT8_DTYPE_NAMES.items() if k is not None}

SAFETENSORS_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}
SAFETENSORS_DTYPE_MAP = {k: v for k, v in SAFETENSORS_DTYPE_MAP.items() if v is not None}

MODEL_FOLDER_LABELS = (
    ("diffusion_models", "diffusion_models"),
    ("unet", "unet"),
    ("diffusion_model", "diffusion_model"),
    ("checkpoints", "checkpoints"),
)

DOWNLOAD_TARGETS = {
    "full": {
        "bf16": ("drbaph/HiDream-O1-Image-BF16", "HiDream-O1-Image-bf16"),
        "fp16": ("drbaph/HiDream-O1-Image-FP16", "HiDream-O1-Image-fp16"),
        "fp8": ("drbaph/HiDream-O1-Image-FP8", "HiDream-O1-Image-fp8"),
    },
    "dev": {
        "bf16": ("drbaph/HiDream-O1-Image-Dev-BF16", "HiDream-O1-Image-Dev-bf16"),
        "fp16": ("drbaph/HiDream-O1-Image-Dev-FP16", "HiDream-O1-Image-Dev-fp16"),
        "fp8": ("drbaph/HiDream-O1-Image-Dev-FP8", "HiDream-O1-Image-Dev-fp8"),
    },
}

CANONICAL_MODEL_CHOICES = {
    "HiDream-O1-Image-BF16": ("full", "bf16"),
    "HiDream-O1-Image-FP16": ("full", "fp16"),
    "HiDream-O1-Image-FP8": ("full", "fp8_e4m3fn"),
    "HiDream-O1-Image-Dev-BF16": ("dev", "bf16"),
    "HiDream-O1-Image-Dev-FP16": ("dev", "fp16"),
    "HiDream-O1-Image-Dev-FP8": ("dev", "fp8_e4m3fn"),
}


def ensure_model_folders() -> None:
    for folder, _label in MODEL_FOLDER_LABELS:
        path = Path(folder_paths.models_dir) / folder
        path.mkdir(parents=True, exist_ok=True)
        try:
            folder_paths.add_model_folder_path(f"hidream_o1_{folder}", str(path))
        except Exception:
            pass


def ensure_lora_folders() -> None:
    lora_path = Path(folder_paths.models_dir) / "lora"
    lora_path.mkdir(parents=True, exist_ok=True)


def lora_root() -> Path:
    ensure_lora_folders()
    return (Path(folder_paths.models_dir) / "lora").resolve()


def model_roots() -> list[tuple[Path, str]]:
    ensure_model_folders()
    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for folder, label in MODEL_FOLDER_LABELS:
        path = (Path(folder_paths.models_dir) / folder).resolve()
        if path not in seen:
            roots.append((path, label))
            seen.add(path)
    return roots


def is_hidream_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_weights = (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()
    has_tokenizer = (path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file()
    return has_config and has_weights and has_tokenizer


def discover_models() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root, label in model_roots():
        if not root.exists():
            continue
        candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
        for path in candidates:
            if is_hidream_model_dir(path):
                try:
                    display = f"{label}/{path.relative_to(root).as_posix()}"
                except ValueError:
                    display = f"{label}/{path.name}"
                out[display] = path
    return dict(sorted(out.items(), key=lambda item: item[0].lower()))


def _casefold(value: str) -> str:
    return value.casefold()


def _target_name_for_choice(model_name: str) -> str | None:
    choice = canonical_model_choice(model_name)
    if choice is None:
        return None
    variant, precision = choice
    precision_key = _download_key_from_precision(precision)
    return DOWNLOAD_TARGETS[variant][precision_key][1]


def canonical_model_choice(model_name: str) -> tuple[str, str] | None:
    wanted = _casefold(model_name)
    for display_name, choice in CANONICAL_MODEL_CHOICES.items():
        if _casefold(display_name) == wanted:
            return choice
    return None


def canonical_model_names() -> list[str]:
    return list(CANONICAL_MODEL_CHOICES.keys())


def lora_names() -> list[str]:
    root = lora_root()
    supported = {ext.lower() for ext in folder_paths.supported_pt_extensions}
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )
    return names or [NO_LORA_NAME]


def lora_path_for_name(lora_name: str) -> str | None:
    if not lora_name or lora_name == NO_LORA_NAME:
        return None
    root = lora_root()
    path = (root / lora_name).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root) or not path.is_file():
        raise FileNotFoundError(f"HiDream O1 LoRA not found in {root}: {lora_name}")
    return str(path)


def find_existing_canonical_model(model_name: str) -> Path | None:
    target_name = _target_name_for_choice(model_name)
    if target_name is None:
        return None
    expected_names = {_casefold(model_name), _casefold(target_name)}
    for path in discover_models().values():
        if _casefold(path.name) in expected_names:
            return path
    return None


def resolve_model_name(model_name: str) -> Path:
    models = discover_models()
    if model_name in models:
        return models[model_name]
    wanted = _casefold(model_name)
    for display, path in models.items():
        if _casefold(display) == wanted or _casefold(path.name) == wanted:
            return path
    path = Path(model_name).expanduser()
    if path.exists() and is_hidream_model_dir(path):
        return path.resolve()
    raise FileNotFoundError(
        "Could not find a HiDream O1 model folder. Put the complete Hugging Face folder "
        "under ComfyUI/models/diffusion_models, models/unet, models/diffusion_model, "
        "or models/checkpoints."
    )


def _dtype_from_safetensors(model_dir: Path) -> torch.dtype | None:
    candidates: list[Path] = []
    single = model_dir / "model.safetensors"
    if single.exists():
        candidates.append(single)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for shard_name in sorted(set(index.get("weight_map", {}).values())):
                shard_path = model_dir / shard_name
                if shard_path.exists():
                    candidates.append(shard_path)
        except Exception:
            pass

    dtype_counts: dict[torch.dtype, int] = {}
    for path in candidates[:2]:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    dtype_name = handle.get_slice(key).get_dtype()
                    dtype = SAFETENSORS_DTYPE_MAP.get(dtype_name)
                    if dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype)):
                        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        except Exception:
            continue

    for dtype in FLOAT8_DTYPE_NAMES:
        if dtype_counts.get(dtype, 0) > 0:
            return dtype
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        if dtype_counts.get(dtype, 0) > 0:
            return dtype
    return None


def _dtype_from_precision_name(precision: str) -> tuple[torch.dtype | None, bool]:
    if precision == "bf16":
        return torch.bfloat16, False
    if precision == "fp16":
        return torch.float16, False
    if precision == "fp32":
        return torch.float32, False
    if precision == "fp8_e4m3fn_fast":
        LOGGER.warning("fp8_e4m3fn_fast has been removed; using fp8_e4m3fn instead.")
        precision = "fp8_e4m3fn"
    if precision == "fp8_e4m3fn":
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
        return dtype, False
    if precision == "fp8_e5m2":
        dtype = getattr(torch, "float8_e5m2", None)
        if dtype is None:
            raise RuntimeError("This PyTorch build does not expose torch.float8_e5m2.")
        return dtype, False
    return None, False


def _config_dtype_name(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return ""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_dtype = str(config.get("dtype") or config.get("torch_dtype") or "")
        text_config = config.get("text_config") or {}
        config_dtype = str(text_config.get("dtype") or text_config.get("torch_dtype") or config_dtype)
        return config_dtype.lower()
    except Exception:
        return ""


def dtype_from_precision(precision: str, model_dir: Path) -> tuple[torch.dtype, bool]:
    explicit_dtype, fp8_optimizations = _dtype_from_precision_name(precision)
    if explicit_dtype is not None:
        return explicit_dtype, fp8_optimizations

    file_dtype = _dtype_from_safetensors(model_dir)
    if file_dtype is not None:
        return file_dtype, False

    config_dtype = _config_dtype_name(model_dir)
    if "float8_e4m3fn" in config_dtype or "fp8_e4m3fn" in config_dtype:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is not None:
            return dtype, False
    if "float8_e5m2" in config_dtype or "fp8_e5m2" in config_dtype:
        dtype = getattr(torch, "float8_e5m2", None)
        if dtype is not None:
            return dtype, False
    if "bfloat16" in config_dtype:
        return torch.bfloat16, False
    if "float16" in config_dtype or "fp16" in config_dtype:
        return torch.float16, False
    return (
        torch.bfloat16 if model_management.should_use_bf16(model_management.get_torch_device()) else torch.float16,
        False,
    )


def is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT8_DTYPE_NAMES


def compute_dtype_from_weight_dtype(weight_dtype: torch.dtype) -> torch.dtype:
    if is_float8_dtype(weight_dtype):
        device = model_management.get_torch_device()
        if model_management.should_use_bf16(device):
            return torch.bfloat16
        if model_management.should_use_fp16(device=device, prioritize_performance=False):
            return torch.float16
        return torch.float32
    if weight_dtype == torch.float16 and model_management.should_use_bf16(model_management.get_torch_device()):
        return torch.bfloat16
    return weight_dtype


def add_special_tokens(tokenizer) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    img = image.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    arr = np.clip(img.numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3]).convert("RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def has_flash_attention() -> bool:
    return getattr(qwen3_vl_transformers, "_flash_attn_func", None) is not None


def has_sage_attention() -> bool:
    get_func = getattr(qwen3_vl_transformers, "get_sage_attention_func", None)
    return bool(get_func and get_func())


def save_temp_image(image: Image.Image, prefix: str = "hidream_o1", extension: str = "jpg") -> dict[str, str] | None:
    temp_dir = Path(folder_paths.get_temp_directory())
    temp_dir.mkdir(parents=True, exist_ok=True)
    extension = extension.lstrip(".").lower()
    if extension not in {"jpg", "jpeg", "png"}:
        extension = "jpg"
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}.{extension}"
    target = temp_dir / filename
    try:
        if extension in {"jpg", "jpeg"}:
            image.convert("RGB").save(target, quality=92, subsampling=0)
        else:
            image.save(target, compress_level=1)
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ENOSPC:
            LOGGER.warning("Skipping HiDream temp preview save because the disk is full: %s", target)
            return None
        raise
    return {"filename": filename, "subfolder": "", "type": "temp"}


def prune_stale_loaded_models() -> int:
    """Remove already-unloaded Comfy entries that would crash cleanup_models_gc."""
    loaded_models = getattr(model_management, "current_loaded_models", None)
    if loaded_models is None:
        return 0

    removed = 0
    for index in range(len(loaded_models) - 1, -1, -1):
        loaded_model = loaded_models[index]
        real_model = getattr(loaded_model, "real_model", None)
        if real_model is not None and callable(real_model):
            continue
        loaded_models.pop(index)
        removed += 1

    if removed:
        LOGGER.info("Removed %s stale Comfy loaded-model entries before HiDream O1 reload.", removed)
    return removed


def _loaded_model_matches_patcher(loaded_model, patcher: comfy.model_patcher.ModelPatcher) -> bool:
    model = getattr(loaded_model, "model", None)
    if model is patcher:
        return True
    if model is None:
        return False
    try:
        if hasattr(model, "is_clone") and model.is_clone(patcher):
            return True
    except Exception:
        pass
    try:
        if hasattr(patcher, "is_clone") and patcher.is_clone(model):
            return True
    except Exception:
        pass
    return False


def remove_loaded_model_entries_for_patcher(patcher: comfy.model_patcher.ModelPatcher) -> bool:
    loaded_models = getattr(model_management, "current_loaded_models", None)
    if loaded_models is None:
        return False

    removed = False
    for index in range(len(loaded_models) - 1, -1, -1):
        loaded_model = loaded_models[index]
        if not _loaded_model_matches_patcher(loaded_model, patcher):
            continue

        real_model = getattr(loaded_model, "real_model", None)
        finalizer = getattr(loaded_model, "model_finalizer", None)
        try:
            if callable(real_model) and finalizer is not None:
                loaded_model.model_unload()
            else:
                model = getattr(loaded_model, "model", None)
                if model is not None:
                    model.detach()
        except Exception as exc:
            LOGGER.debug("Ignoring HiDream O1 unload cleanup error: %s", exc)
        loaded_models.pop(index)
        removed = True

    return removed


class HiDreamTorchWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, compute_dtype: torch.dtype, weight_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.hidream_dtype = compute_dtype
        self.hidream_weight_dtype = weight_dtype
        self.manual_cast_dtype = compute_dtype if compute_dtype != weight_dtype else None
        self.device = torch.device("cpu")

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_dtype(self):
        return self.hidream_dtype

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except Exception:
            pass
        return self


@dataclass
class HiDreamO1Handle:
    patcher: comfy.model_patcher.ModelPatcher
    processor: object
    model_dir: Path
    dtype: torch.dtype
    weight_dtype: torch.dtype
    attention: str

    def load_for_inference(self, memory_required: int = 0) -> HiDreamTorchWrapper:
        prune_stale_loaded_models()
        model_management.load_models_gpu([self.patcher], memory_required=memory_required)
        model = self.patcher.model
        model.hidream_dtype = self.dtype
        model.hidream_weight_dtype = self.weight_dtype
        return model

    def should_use_flash_attention(self) -> bool:
        if self.attention == "flash":
            if not has_flash_attention():
                raise RuntimeError(
                    "Flash attention was requested, but neither flash_attn_interface nor flash_attn is installed. "
                    "Use attention=sdpa/sage/auto, or install a compatible flash-attn build."
                )
            return True
        if self.attention == "auto":
            return has_flash_attention()
        return False

    def should_use_sage_attention(self) -> bool:
        if self.attention != "sage":
            return False
        if not has_sage_attention():
            raise RuntimeError(
                "SageAttention was requested, but the sageattention package is not installed or failed to import. "
                "Use attention=sdpa/flash/auto, or install a compatible sageattention build."
            )
        return True

    def resolve_attention_backend(self) -> str:
        flash_available = has_flash_attention()
        sage_available = has_sage_attention()

        if self.attention == "auto":
            backend = "flash" if flash_available else "sdpa"
        elif self.attention == "flash":
            if not flash_available:
                raise RuntimeError(
                    "Flash attention was requested, but neither flash_attn_interface nor flash_attn is installed. "
                    "Use attention=sdpa/sage/auto, or install a compatible flash-attn build."
                )
            backend = "flash"
        elif self.attention == "sage":
            if not sage_available:
                raise RuntimeError(
                    "SageAttention was requested, but the sageattention package is not installed or failed to import. "
                    "Use attention=sdpa/flash/auto, or install a compatible sageattention build."
                )
            backend = "sage"
            try:
                text_config = self.patcher.model.config.text_config
                heads = getattr(text_config, "num_attention_heads", None)
                kv_heads = getattr(text_config, "num_key_value_heads", None)
                if heads and kv_heads and heads != kv_heads:
                    LOGGER.warning(
                        "HiDream O1 SageAttention is running on GQA attention (%s query heads, %s KV heads). "
                        "The Sage path expands KV heads for compatibility, so FlashAttention can be much faster.",
                        heads,
                        kv_heads,
                    )
            except Exception:
                pass
        else:
            backend = "sdpa"

        LOGGER.info(
            "HiDream O1 attention backend: %s (requested=%s, flash_available=%s, sage_available=%s)",
            backend,
            self.attention,
            flash_available,
            sage_available,
        )
        return backend

    def offload(self) -> None:
        try:
            unloaded = remove_loaded_model_entries_for_patcher(self.patcher)
            if not unloaded:
                self.patcher.detach()
        finally:
            prune_stale_loaded_models()
            gc.collect()
            model_management.soft_empty_cache()

    def dispose(self) -> None:
        try:
            self.offload()
        finally:
            try:
                self.patcher.detach()
            except Exception:
                pass
            self.processor = None
            gc.collect()
            model_management.soft_empty_cache()

    def clone_with_patcher(self, patcher: comfy.model_patcher.ModelPatcher) -> "HiDreamO1Handle":
        return HiDreamO1Handle(
            patcher=patcher,
            processor=self.processor,
            model_dir=self.model_dir,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            attention=self.attention,
        )


class _TorchNNProxy:
    def __init__(self, base_nn, operations):
        self._base_nn = base_nn
        self._operations = operations

    def __getattr__(self, name: str):
        if hasattr(self._operations, name):
            return getattr(self._operations, name)
        return getattr(self._base_nn, name)


def _fp8_safety_recast(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    recast = 0
    for name, param in list(model.named_parameters()):
        if not is_float8_dtype(param.dtype):
            continue
        if param.ndim >= 2 and not name.endswith(".bias"):
            continue
        comfy.utils.set_attr_param(model, name, param.detach().to(dtype=compute_dtype))
        recast += 1
    return recast


def _convert_matrix_params_to_dtype(model: torch.nn.Module, target_dtype: torch.dtype) -> int:
    converted = 0
    for name, param in list(model.named_parameters()):
        if not torch.is_floating_point(param) or param.dtype == target_dtype:
            continue
        if param.ndim < 2 or name.endswith(".bias"):
            continue
        comfy.utils.set_attr_param(model, name, param.detach().to(dtype=target_dtype))
        converted += 1
    return converted


def _lora_prefix_variants(module_name: str) -> set[str]:
    variants = {module_name}
    prefixes_to_strip = (
        "model.",
        "model.model.",
        "base_model.model.",
        "base_model.model.model.",
        "transformer.",
        "diffusion_model.",
    )
    for prefix in prefixes_to_strip:
        if module_name.startswith(prefix):
            variants.add(module_name[len(prefix):])

    expanded = set(variants)
    for name in variants:
        expanded.add(f"base_model.model.{name}")
        expanded.add(f"transformer.{name}")
        expanded.add(f"diffusion_model.{name}")
        expanded.add(f"lora_unet_{name.replace('.', '_')}")
        expanded.add(f"lycoris_{name.replace('.', '_')}")
    return expanded


def hidream_lora_key_map(model: torch.nn.Module) -> dict[str, str]:
    key_map: dict[str, str] = {}
    for key in model.state_dict().keys():
        if not key.endswith(".weight"):
            continue
        module_name = key[: -len(".weight")]
        for lora_key in _lora_prefix_variants(module_name):
            key_map.setdefault(lora_key, key)
    return key_map


def apply_hidream_lora(handle: HiDreamO1Handle, lora: dict[str, torch.Tensor], strength: float, lora_name: str = "") -> HiDreamO1Handle:
    if strength == 0:
        return handle

    key_map = hidream_lora_key_map(handle.patcher.model)
    converted_lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(converted_lora, key_map)

    new_patcher = handle.patcher.clone()
    patched_keys = set(new_patcher.add_patches(loaded, strength))
    missed_keys = [key for key in loaded if key not in patched_keys]
    if patched_keys:
        LOGGER.info("Applied HiDream O1 LoRA %s to %s weights at strength %.3f.", lora_name or "<memory>", len(patched_keys), strength)
    else:
        raise RuntimeError(
            f"HiDream O1 LoRA {lora_name or '<memory>'} did not match any model weights. "
            "This usually means the LoRA was trained for a different HiDream architecture or uses unsupported key names."
        )
    if missed_keys:
        LOGGER.warning("HiDream O1 LoRA %s had %s unmatched converted patches.", lora_name or "<memory>", len(missed_keys))
    return handle.clone_with_patcher(new_patcher)


def _load_single_safetensors_direct(model: torch.nn.Module, model_dir: Path) -> bool:
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        return False

    state_dict = comfy.utils.load_torch_file(str(weights_path), safe_load=True)
    tensor_count = len(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    missing = [key for key in missing if not key.endswith(".inv_freq")]
    LOGGER.info("Loaded %s HiDream O1 tensors directly from %s.", tensor_count, weights_path.name)
    if unexpected:
        LOGGER.warning("Ignored %s unexpected HiDream O1 weight keys while loading %s.", len(unexpected), weights_path.name)
    if missing:
        LOGGER.warning("HiDream O1 direct safetensors load missed %s model keys.", len(missing))
    return True


def load_hidream_model(model_dir: Path, precision: str = "auto", attention: str = "auto") -> HiDreamO1Handle:
    weight_dtype, fp8_optimizations = dtype_from_precision(precision, model_dir)
    file_dtype = _dtype_from_safetensors(model_dir)
    compute_dtype = compute_dtype_from_weight_dtype(weight_dtype)
    dtype_label = f"{weight_dtype} weights, {compute_dtype} compute"
    if fp8_optimizations:
        dtype_label += ", fp8 fast ops"
    LOGGER.info("Loading HiDream O1 from %s as %s", model_dir, dtype_label)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    add_special_tokens(tokenizer)

    load_device = model_management.get_torch_device()
    operations = comfy.ops.pick_operations(
        weight_dtype,
        compute_dtype,
        load_device=load_device,
        fp8_optimizations=fp8_optimizations,
    )
    nn_proxy = _TorchNNProxy(qwen3_vl_transformers.nn, operations)
    original_nn = qwen3_vl_transformers.nn

    kwargs = {}
    config_dtype_name = _config_dtype_name(model_dir)
    if is_float8_dtype(weight_dtype):
        kwargs["torch_dtype"] = compute_dtype
        if "float8" in config_dtype_name or "fp8" in config_dtype_name:
            LOGGER.warning(
                "Model config declares %s, which Transformers cannot use as a default dtype. "
                "Using %s for model init; keep config dtype as bfloat16 for native FP8 safetensors loading.",
                config_dtype_name,
                compute_dtype,
            )
    else:
        kwargs["torch_dtype"] = weight_dtype
    try:
        qwen3_vl_transformers.nn = nn_proxy
        loaded_direct = False
        if (model_dir / "model.safetensors").exists():
            config = Qwen3VLForConditionalGeneration.config_class.from_pretrained(str(model_dir))
            model = Qwen3VLForConditionalGeneration(config)
            loaded_direct = _load_single_safetensors_direct(model, model_dir)
        if not loaded_direct:
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    finally:
        qwen3_vl_transformers.nn = original_nn

    if is_float8_dtype(weight_dtype):
        fp8_params = sum(1 for param in model.parameters() if param.dtype == weight_dtype)
    else:
        fp8_params = 0

    if is_float8_dtype(weight_dtype) and (file_dtype != weight_dtype or fp8_params == 0):
        converted = _convert_matrix_params_to_dtype(model, weight_dtype)
        LOGGER.info("Converted %s large weight tensors to %s in memory.", converted, weight_dtype)

    if is_float8_dtype(weight_dtype):
        recast = _fp8_safety_recast(model, compute_dtype)
        if recast:
            LOGGER.info("Kept %s small/bias FP8 tensors in %s for safe arithmetic.", recast, compute_dtype)
    model_management.archive_model_dtypes(model)
    for module in model.modules():
        module.hidream_compute_dtype = compute_dtype

    for module in model.modules():
        if "RotaryEmbedding" not in module.__class__.__name__:
            continue
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is None or getattr(inv_freq, "is_meta", False):
            inv_freq, attention_scaling = module.compute_default_rope_parameters(module.config, device=torch.device("cpu"))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.attention_scaling = attention_scaling
        module.original_inv_freq = module.inv_freq

    wrapped = HiDreamTorchWrapper(model.eval(), compute_dtype, weight_dtype)
    patcher = comfy.model_patcher.CoreModelPatcher(
        wrapped,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
    )
    return HiDreamO1Handle(
        patcher=patcher,
        processor=processor,
        model_dir=model_dir,
        dtype=compute_dtype,
        weight_dtype=weight_dtype,
        attention=attention,
    )


def _download_key_from_precision(precision: str) -> str:
    if precision == "fp16":
        return "fp16"
    if precision.startswith("fp8"):
        return "fp8"
    return "bf16"


def _download_variant_name(model_variant: str | None) -> str:
    model_variant = (model_variant or "full").lower()
    if model_variant == "dev":
        return "dev"
    return "full"


def maybe_download_model(precision: str = "auto", model_variant: str = "full") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Install huggingface_hub to enable automatic model downloads.") from exc

    variant_key = _download_variant_name(model_variant)
    precision_key = _download_key_from_precision(precision)
    repo_id, target_name = DOWNLOAD_TARGETS[variant_key][precision_key]
    target = Path(folder_paths.models_dir) / "diffusion_models" / target_name
    if is_hidream_model_dir(target):
        LOGGER.info("Using existing HiDream O1 model folder: %s", target)
        return target
    target.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading HiDream O1 %s %s model from %s to %s", variant_key.upper(), precision_key.upper(), repo_id, target)
    try:
        snapshot_download(repo_id, local_dir=str(target), local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(repo_id, local_dir=str(target))
    return target
