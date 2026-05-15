from __future__ import annotations

import logging
import os

import comfy.model_management as model_management
import comfy.utils
import folder_paths

from .hidream_o1.comfy_runtime import (
    NO_LORA_NAME,
    apply_hidream_lora,
    canonical_model_choice,
    canonical_model_names,
    HiDreamO1Handle,
    find_existing_canonical_model,
    lora_names,
    lora_path_for_name,
    load_hidream_model,
    maybe_download_model,
    pil_to_tensor,
    save_temp_image,
    tensor_to_pil,
)
from .hidream_o1.models.pipeline import DEFAULT_TIMESTEPS, generate_image
from .hidream_o1.training import clean_output_name, create_image_caption_jsonl, run_hidream_o1_lora_training

try:
    from comfy_api.latest import IO

    _V3 = True
except ImportError:
    IO = None
    _V3 = False


MAX_IMAGE_INPUTS = 12
LOGGER = logging.getLogger("HiDream_O1")
PRECISION_CHOICES = [
    "auto",
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3fn",
    "fp8_e5m2",
]
ATTENTION_CHOICES = ["auto", "flash", "sdpa", "sage"]
_LOADED_MODEL_HANDLES: dict[str, tuple[tuple[str, str, str], HiDreamO1Handle]] = {}
SAMPLER_INPUT_DEFAULTS = {
    "dev_editing_scheduler": "flow_match",
    "layout_bboxes": "",
}

TOOLTIPS = {
    "model": "Loaded HiDream O1 model handle.",
    "conditioning": "Prompt conditioning from HiDream O1 Conditioning.",
    "model_name": "HiDream O1 model choice. Converted Full/Dev BF16, FP16, FP8, and Dev-2604 entries are always shown; missing entries download only when download_if_missing is enabled.",
    "lora_name": "HiDream O1 LoRA file from models/loras.",
    "lora_strength": "Default: 1.0. LoRA model strength from -10.0 to 10.0; 0 disables the LoRA.",
    "smoothing": "Dev/Dev-2604 only. Applies patch-grid smoothing in the last denoise steps to reduce visible tile seams.",
    "precision": "Weight precision to load. Default: auto detects the safetensors dtype. FP16/FP8 weights use BF16 compute on BF16-capable GPUs to avoid NaNs.",
    "attention": "Attention backend. Default: auto uses FlashAttention when installed, otherwise SDPA. Use sage only if sageattention is installed.",
    "download_if_missing": "Default: false. If the selected canonical model is missing locally, enabling this downloads that selected model into models/diffusion_models.",
    "prompt": "Text instruction for HiDream O1. Default is a simple cinematic portrait prompt.",
    "enhanced_prompt": "Optional STRING input from ComfyUI's bundled Prompt Enhance subgraph or any prompt-enhancer output. When connected and non-empty, this replaces the prompt textbox.",
    "negative_prompt": "Default: empty. Used as the unconditional CFG branch in full mode when guidance_scale is above 1. Dev mode ignores CFG.",
    "model_type": "Default: auto. Uses dev settings when the model folder name contains dev, otherwise full settings.",
    "width": "Requested output width. Default: 2048. HiDream snaps to its nearest supported patch-aligned resolution.",
    "height": "Requested output height. Default: 2048. HiDream snaps to its nearest supported patch-aligned resolution.",
    "steps": "Default: 0 means auto: 50 steps for full. Dev always uses the upstream fixed 28-step schedule.",
    "guidance_scale": "Default: 5.0. Classifier-free guidance for full mode; dev mode ignores this and uses 0.",
    "shift": "Default: -1 means auto: 3.0 for full, 1.0 for dev.",
    "noise_scale_start": "Default: 7.5. Initial noise scale used by the scheduler.",
    "noise_scale_end": "Default: 7.5. Final noise scale used by the scheduler.",
    "noise_clip_std": "Default: 2.5. Clips scheduler noise outliers; lower values clamp harder.",
    "dev_editing_scheduler": "Default: flow_match. Upstream uses flow_match for Dev edit mode with exactly one reference image; flash remains available.",
    "layout_bboxes": "Optional JSON string or JSON file path for upstream layout conditioning. Uses relative xxyy boxes like [[0.1, 0.45, 0.2, 0.8]].",
    "preview_every": "Default: 4. Sends a decoded preview every N steps; 0 disables previews.",
    "keep_image1_aspect": "Default: false. Only applies when image_1 is connected; output aspect follows image_1.",
    "force_offload": "Default: false. Immediately unloads the HiDream model after sampling.",
    "image": "Default: 0. Choose how many optional reference image inputs to show; 0 means text-only.",
    "num_images": "Default: 0. Number of optional reference image inputs to read; 0 means text-only.",
}


def _collect_ref_images(inputs: dict, count: int = MAX_IMAGE_INPUTS):
    refs = []
    count = max(0, min(MAX_IMAGE_INPUTS, int(count)))
    for i in range(1, count + 1):
        image = inputs.get(f"image_{i}")
        if image is None:
            continue
        if image.ndim == 4:
            for frame in image:
                refs.append(tensor_to_pil(frame))
        else:
            refs.append(tensor_to_pil(image))
    return refs


def _refs_from_dynamic_image(image):
    if not isinstance(image, dict):
        return []
    try:
        count = int(image.get("image", MAX_IMAGE_INPUTS))
    except (TypeError, ValueError):
        count = MAX_IMAGE_INPUTS
    return _collect_ref_images(image, count)


def _is_dev_model(model: HiDreamO1Handle) -> bool:
    return "dev" in model.model_dir.name.lower()


def _run_sampler(
    model: HiDreamO1Handle,
    conditioning,
    model_type: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    guidance_scale: float,
    shift: float,
    noise_scale_start: float,
    noise_scale_end: float,
    noise_clip_std: float,
    dev_editing_scheduler: str,
    layout_bboxes: str,
    preview_every: int,
    keep_image1_aspect: bool,
    force_offload: bool,
    refs=None,
    unique_id=None,
):
    prompt = conditioning["prompt"]
    negative_prompt = conditioning.get("negative_prompt", "")
    refs = refs or conditioning.get("refs") or []
    smoothing = dict(model.smoothing or {})
    keep_image1_aspect = bool(keep_image1_aspect and refs)
    resolved_type = "dev" if model_type == "auto" and "dev" in model.model_dir.name.lower() else model_type
    if resolved_type == "auto":
        resolved_type = "full"
    if smoothing and (resolved_type != "dev" or not _is_dev_model(model)):
        raise RuntimeError("HiDream O1 smoothing is only supported for Dev/Dev-2604 model folders in dev mode.")

    if resolved_type == "dev":
        if steps not in (0, len(DEFAULT_TIMESTEPS)):
            LOGGER.warning(
                "HiDream O1 dev uses the upstream fixed %s-step schedule; ignoring steps=%s.",
                len(DEFAULT_TIMESTEPS),
                steps,
            )
        num_steps = len(DEFAULT_TIMESTEPS)
        resolved_guidance = 0.0
        resolved_shift = 1.0 if shift < 0 else shift
        timesteps_list = DEFAULT_TIMESTEPS
        scheduler_name = "flow_match" if len(refs) == 1 and dev_editing_scheduler == "flow_match" else "flash"
        if scheduler_name == "flash" and (noise_scale_start, noise_scale_end, noise_clip_std) != (7.5, 7.5, 2.5):
            LOGGER.warning(
                "HiDream O1 dev upstream defaults are noise_scale_start=7.5, "
                "noise_scale_end=7.5, noise_clip_std=2.5. Different values can cause noisy or odd-color outputs."
            )
    else:
        num_steps = steps or 50
        resolved_guidance = guidance_scale
        resolved_shift = 3.0 if shift < 0 else shift
        timesteps_list = None
        scheduler_name = "default"

    LOGGER.info(
        "HiDream O1 sampler mode: %s (requested=%s, steps=%s, guidance=%s, shift=%s, scheduler=%s, negative_prompt=%s)",
        resolved_type,
        model_type,
        num_steps,
        resolved_guidance,
        resolved_shift,
        scheduler_name,
        bool(negative_prompt and resolved_guidance > 1.0),
    )

    node_id = str(unique_id) if unique_id is not None else None
    pbar = comfy.utils.ProgressBar(num_steps, node_id=node_id)

    def callback(step_idx, total_steps, get_preview=None):
        model_management.throw_exception_if_processing_interrupted()
        preview = None
        if get_preview is not None and preview_every > 0 and (
            (step_idx + 1) % preview_every == 0 or step_idx + 1 == total_steps
        ):
            preview = ("JPEG", get_preview().convert("RGB"), 768)
        pbar.update_absolute(step_idx + 1, total_steps, preview)

    try:
        inference_model = model.load_for_inference()
        attention_backend = model.resolve_attention_backend()
        if resolved_type == "dev" and attention_backend != "flash":
            LOGGER.warning(
                "HiDream O1 upstream dev inference uses FlashAttention; current backend is %s.",
                attention_backend,
            )
        image = generate_image(
            model=inference_model,
            processor=model.processor,
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_image_paths=refs,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=resolved_guidance,
            shift=resolved_shift,
            timesteps_list=timesteps_list,
            scheduler_name=scheduler_name,
            seed=seed,
            noise_scale_start=noise_scale_start,
            noise_scale_end=noise_scale_end,
            noise_clip_std=noise_clip_std,
            layout_bboxes=layout_bboxes or None,
            seam_smooth_steps=int(smoothing.get("steps", 0)),
            seam_smooth_strength=float(smoothing.get("strength", 0.5)),
            seam_smooth_schedule=str(smoothing.get("schedule", "constant")),
            seam_smooth_shift_mode=str(smoothing.get("shift_mode", "rotate")),
            seam_smooth_adaptive_threshold=float(smoothing.get("adaptive_threshold", 0.0)),
            seam_smooth_multiscale=bool(smoothing.get("multiscale", False)),
            seam_smooth_cfg_aware=bool(smoothing.get("cfg_aware", False)),
            keep_original_aspect=keep_image1_aspect,
            use_flash_attn=attention_backend == "flash",
            use_sage_attn=attention_backend == "sage",
            callback=callback,
        )
    finally:
        if force_offload:
            model.offload()

    result = pil_to_tensor(image).to(model_management.intermediate_device())
    final_ui = save_temp_image(image, prefix="hidream_o1")
    if final_ui is not None:
        return {"ui": {"images": [final_ui]}, "result": (result,)}
    return {"result": (result,)}


class HiDreamO1ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = canonical_model_names()
        return {
            "required": {
                "model_name": (models, {"default": "HiDream-O1-Image-BF16", "tooltip": TOOLTIPS["model_name"]}),
                "precision": (PRECISION_CHOICES, {"default": "auto", "tooltip": TOOLTIPS["precision"]}),
                "attention": (ATTENTION_CHOICES, {"default": "auto", "tooltip": TOOLTIPS["attention"]}),
                "download_if_missing": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["download_if_missing"]}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HiDream O1"

    @staticmethod
    def _cache_key(unique_id) -> str:
        return str(unique_id) if unique_id is not None else "global"

    @classmethod
    def _dispose_cached_handle(cls, key: str) -> None:
        cached = _LOADED_MODEL_HANDLES.pop(key, None)
        if cached is None:
            return
        _signature, handle = cached
        LOGGER.info("Unloading previous HiDream O1 model for loader node %s.", key)
        try:
            handle.dispose()
        except Exception as exc:
            LOGGER.warning("HiDream O1 previous model cleanup hit an error: %s", exc)

    @classmethod
    def _load_cached(cls, key: str, model_dir, precision: str, attention: str):
        signature = (str(model_dir.resolve()), precision, attention)
        cached = _LOADED_MODEL_HANDLES.get(key)
        if cached is not None:
            cached_signature, cached_handle = cached
            if cached_signature == signature:
                return cached_handle
            cls._dispose_cached_handle(key)

        handle = load_hidream_model(model_dir, precision=precision, attention=attention)
        _LOADED_MODEL_HANDLES[key] = (signature, handle)
        return handle

    def load_model(
        self,
        model_name: str,
        precision: str,
        attention: str,
        download_if_missing: bool,
        unique_id=None,
    ):
        key = self._cache_key(unique_id)
        canonical_choice = canonical_model_choice(model_name)
        if canonical_choice is not None:
            model_variant, model_precision = canonical_choice
            model_dir = find_existing_canonical_model(model_name)
            if model_dir is None:
                if not download_if_missing:
                    raise FileNotFoundError(
                        f"{model_name} is not installed. Enable download_if_missing to download it, "
                        "or place the complete model folder in ComfyUI/models/diffusion_models."
                    )
                model_dir = maybe_download_model(precision=model_precision, model_variant=model_variant)
            return (self._load_cached(key, model_dir, model_precision, attention),)

        from .hidream_o1.comfy_runtime import resolve_model_name

        model_dir = resolve_model_name(model_name)
        return (self._load_cached(key, model_dir, precision, attention),)


class HiDreamO1Conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cinematic portrait with detailed lighting.",
                        "tooltip": TOOLTIPS["prompt"],
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": TOOLTIPS["negative_prompt"],
                    },
                ),
            },
            "optional": {
                "enhanced_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "default": "",
                        "tooltip": TOOLTIPS["enhanced_prompt"],
                    },
                ),
            },
        }

    RETURN_TYPES = ("HIDREAM_O1_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "condition"
    CATEGORY = "HiDream O1"

    def condition(self, prompt: str, negative_prompt: str = "", enhanced_prompt: str | None = None):
        active_prompt = enhanced_prompt.strip() if isinstance(enhanced_prompt, str) and enhanced_prompt.strip() else prompt
        return ({"prompt": active_prompt, "negative_prompt": negative_prompt},)


class HiDreamO1Lora:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_O1_MODEL", {"tooltip": TOOLTIPS["model"]}),
                "lora_name": (lora_names(), {"tooltip": TOOLTIPS["lora_name"]}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": TOOLTIPS["lora_strength"],
                    },
                ),
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_lora"
    CATEGORY = "HiDream O1"

    def apply_lora(self, model: HiDreamO1Handle, lora_name: str, strength: float):
        if lora_name == NO_LORA_NAME or strength == 0:
            return (model,)

        lora_path = lora_path_for_name(lora_name)
        if lora_path is None:
            return (model,)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return (apply_hidream_lora(model, lora, strength, lora_name=lora_name),)


class HiDreamO1DevSmoothing:
    SCHEDULES = ["constant", "linear", "cosine", "front_loaded", "late"]
    SHIFT_MODES = ["rotate", "static", "all"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_O1_MODEL", {"tooltip": TOOLTIPS["model"]}),
                "steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Final denoise steps to run patch-grid smoothing. 0 disables smoothing.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend strength for the shifted patch prediction.",
                    },
                ),
                "schedule": (cls.SCHEDULES, {"default": "constant"}),
                "shift_mode": (cls.SHIFT_MODES, {"default": "rotate"}),
                "adaptive_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Skip smoothing when estimated seam intensity is below this value. 0 disables the skip check.",
                    },
                ),
                "multiscale": ("BOOLEAN", {"default": False}),
                "cfg_aware": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also smooth the unconditional branch when CFG is active. Costs extra forwards.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_smoothing"
    CATEGORY = "HiDream O1"

    def apply_smoothing(
        self,
        model: HiDreamO1Handle,
        steps: int,
        strength: float,
        schedule: str,
        shift_mode: str,
        adaptive_threshold: float,
        multiscale: bool,
        cfg_aware: bool,
    ):
        steps = int(steps)
        if steps <= 0:
            return (model.clone_with_smoothing(None),)
        if not _is_dev_model(model):
            raise RuntimeError("HiDream O1 smoothing only supports Dev/Dev-2604 model folders.")
        smoothing = {
            "steps": steps,
            "strength": float(strength),
            "schedule": schedule,
            "shift_mode": shift_mode,
            "adaptive_threshold": float(adaptive_threshold),
            "multiscale": bool(multiscale),
            "cfg_aware": bool(cfg_aware),
        }
        return (model.clone_with_smoothing(smoothing),)


class HiDreamO1DatasetMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder containing image files and matching .txt captions.",
                    },
                ),
                "output_filename": (
                    "STRING",
                    {
                        "default": "train.jsonl",
                        "tooltip": "Dataset manifest filename to write inside the image folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "create_dataset"
    CATEGORY = "HiDream O1/training"

    def create_dataset(self, image_directory: str, output_filename: str):
        return (create_image_caption_jsonl(image_directory, output_filename),)


class HiDreamO1TrainConfig:
    TARGET_PRESETS = ["aitoolkit", "attention+mlp+pixel", "attention+pixel", "attention"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "lora_rank": ("INT", {"default": 32, "min": 4, "max": 256, "step": 4}),
                "lora_alpha": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 256.0, "step": 1.0}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "weight_decay": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 0.2, "step": 0.0001}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 1}),
                "grad_accum_steps": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "caption_dropout": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "timestep_type": (["linear", "sigmoid", "shift"], {"default": "linear"}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "min_sigma": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.95, "step": 0.001}),
                "max_sigma": ("FLOAT", {"default": 0.999, "min": 0.05, "max": 0.9999, "step": 0.001}),
                "noise_scale": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "loss_target": (["velocity", "x0"], {"default": "velocity"}),
                "max_loss": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "target_preset": (cls.TARGET_PRESETS, {"default": "aitoolkit"}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "save_dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("HIDREAM_O1_TRAIN_CONFIG",)
    RETURN_NAMES = ("train_config",)
    FUNCTION = "configure"
    CATEGORY = "HiDream O1/training"

    def configure(self, **kwargs):
        return (kwargs,)


class HiDreamO1LoraTrainer:
    TRAIN_MODEL_CHOICES = [
        "HiDream-O1-Image-BF16",
        "HiDream-O1-Image-FP16",
        "HiDream-O1-Image-FP8",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_name": (cls.TRAIN_MODEL_CHOICES, {"default": "HiDream-O1-Image-BF16"}),
                "precision": (PRECISION_CHOICES, {"default": "auto", "tooltip": TOOLTIPS["precision"]}),
                "attention": (ATTENTION_CHOICES, {"default": "auto", "tooltip": TOOLTIPS["attention"]}),
                "download_if_missing": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["download_if_missing"]}),
                "train_config": ("HIDREAM_O1_TRAIN_CONFIG",),
                "dataset_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "JSONL manifest from HiDream O1 Dataset Maker.",
                    },
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": "hidream_o1_lora",
                        "tooltip": "Subfolder name in ComfyUI/models/loras.",
                    },
                ),
                "max_steps": ("INT", {"default": 3000, "min": 1, "max": 100000, "step": 1}),
                "save_every_steps": ("INT", {"default": 250, "min": 1, "max": 10000, "step": 1}),
                "num_workers": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_output_dir",)
    FUNCTION = "train"
    CATEGORY = "HiDream O1/training"

    def train(
        self,
        base_model_name: str,
        precision: str,
        attention: str,
        download_if_missing: bool,
        train_config,
        dataset_path: str,
        output_name: str,
        max_steps: int,
        save_every_steps: int,
        num_workers: int,
    ):
        lora_root = folder_paths.get_folder_paths("loras")[0]
        output_name = clean_output_name(output_name)
        output_dir = os.path.join(lora_root, output_name)
        return (
            run_hidream_o1_lora_training(
                base_model_name=base_model_name,
                precision=precision,
                attention=attention,
                train_config=train_config,
                dataset_path=dataset_path,
                output_dir=output_dir,
                output_name=output_name,
                max_steps=max_steps,
                save_every_steps=save_every_steps,
                num_workers=num_workers,
                download_if_missing=download_if_missing,
            ),
        )


def _sampler_required_inputs():
    return {
        "model": ("HIDREAM_O1_MODEL", {"tooltip": TOOLTIPS["model"]}),
        "conditioning": ("HIDREAM_O1_CONDITIONING", {"tooltip": TOOLTIPS["conditioning"]}),
        "model_type": (["auto", "full", "dev"], {"default": "auto", "tooltip": TOOLTIPS["model_type"]}),
        "width": ("INT", {"default": 2048, "min": 512, "max": 3104, "step": 32, "tooltip": TOOLTIPS["width"]}),
        "height": ("INT", {"default": 2048, "min": 512, "max": 3104, "step": 32, "tooltip": TOOLTIPS["height"]}),
        "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": TOOLTIPS["steps"]}),
        "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": TOOLTIPS["guidance_scale"]}),
        "shift": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.1, "tooltip": TOOLTIPS["shift"]}),
        "noise_scale_start": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": TOOLTIPS["noise_scale_start"]}),
        "noise_scale_end": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": TOOLTIPS["noise_scale_end"]}),
        "noise_clip_std": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": TOOLTIPS["noise_clip_std"]}),
        "dev_editing_scheduler": (["flow_match", "flash"], {"default": "flow_match", "tooltip": TOOLTIPS["dev_editing_scheduler"]}),
        "layout_bboxes": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["layout_bboxes"]}),
        "preview_every": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": TOOLTIPS["preview_every"]}),
        "keep_image1_aspect": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["keep_image1_aspect"]}),
        "force_offload": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["force_offload"]}),
    }


if _V3:

    def _dynamic_image_options():
        return [
            IO.DynamicCombo.Option(
                key=str(count),
                inputs=[
                    IO.Image.Input(
                        f"image_{i}",
                        optional=True,
                        tooltip=f"Reference image {i}.",
                    )
                    for i in range(1, count + 1)
                ],
            )
            for count in range(0, MAX_IMAGE_INPUTS + 1)
        ]

    class HiDreamO1Sampler(IO.ComfyNode):
        @classmethod
        def define_schema(cls):
            return IO.Schema(
                node_id="HiDreamO1Sampler",
                display_name="HiDream O1 Sampler",
                category="HiDream O1",
                inputs=[
                    IO.Custom("HIDREAM_O1_MODEL").Input("model", tooltip=TOOLTIPS["model"]),
                    IO.Custom("HIDREAM_O1_CONDITIONING").Input("conditioning", tooltip=TOOLTIPS["conditioning"]),
                    IO.Combo.Input("model_type", options=["auto", "full", "dev"], default="auto", tooltip=TOOLTIPS["model_type"]),
                    IO.Int.Input("width", default=2048, min=512, max=3104, step=32, tooltip=TOOLTIPS["width"]),
                    IO.Int.Input("height", default=2048, min=512, max=3104, step=32, tooltip=TOOLTIPS["height"]),
                    IO.Int.Input("steps", default=0, min=0, max=100, tooltip=TOOLTIPS["steps"]),
                    IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                    IO.Float.Input("guidance_scale", default=5.0, min=0.0, max=30.0, step=0.1, tooltip=TOOLTIPS["guidance_scale"]),
                    IO.Float.Input("shift", default=-1.0, min=-1.0, max=10.0, step=0.1, tooltip=TOOLTIPS["shift"]),
                    IO.Float.Input("noise_scale_start", default=7.5, min=0.0, max=30.0, step=0.1, tooltip=TOOLTIPS["noise_scale_start"]),
                    IO.Float.Input("noise_scale_end", default=7.5, min=0.0, max=30.0, step=0.1, tooltip=TOOLTIPS["noise_scale_end"]),
                    IO.Float.Input("noise_clip_std", default=2.5, min=0.0, max=20.0, step=0.1, tooltip=TOOLTIPS["noise_clip_std"]),
                    IO.Combo.Input("dev_editing_scheduler", options=["flow_match", "flash"], default="flow_match", tooltip=TOOLTIPS["dev_editing_scheduler"]),
                    IO.String.Input("layout_bboxes", default="", multiline=True, tooltip=TOOLTIPS["layout_bboxes"]),
                    IO.Int.Input("preview_every", default=4, min=0, max=100, tooltip=TOOLTIPS["preview_every"]),
                    IO.Boolean.Input("keep_image1_aspect", default=False, tooltip=TOOLTIPS["keep_image1_aspect"]),
                    IO.Boolean.Input("force_offload", default=False, tooltip=TOOLTIPS["force_offload"]),
                    IO.DynamicCombo.Input(
                        "image",
                        options=_dynamic_image_options(),
                        display_name="image",
                        tooltip=TOOLTIPS["image"],
                    ),
                ],
                outputs=[IO.Image.Output(display_name="image")],
                hidden=[IO.Hidden.unique_id],
            )

        @classmethod
        def execute(
            cls,
            model: HiDreamO1Handle,
            conditioning,
            model_type: str,
            width: int,
            height: int,
            steps: int,
            seed: int,
            guidance_scale: float,
            shift: float,
            noise_scale_start: float,
            noise_scale_end: float,
            noise_clip_std: float,
            dev_editing_scheduler: str,
            layout_bboxes: str,
            preview_every: int,
            keep_image1_aspect: bool,
            force_offload: bool,
            image: dict,
        ):
            return _run_sampler(
                model=model,
                conditioning=conditioning,
                model_type=model_type,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                guidance_scale=guidance_scale,
                shift=shift,
                noise_scale_start=noise_scale_start,
                noise_scale_end=noise_scale_end,
                noise_clip_std=noise_clip_std,
                dev_editing_scheduler=dev_editing_scheduler,
                layout_bboxes=layout_bboxes,
                preview_every=preview_every,
                keep_image1_aspect=keep_image1_aspect,
                force_offload=force_offload,
                refs=_refs_from_dynamic_image(image),
                unique_id=cls.hidden.unique_id,
            )

else:

    class HiDreamO1Sampler:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    **_sampler_required_inputs(),
                    "num_images": ("INT", {"default": 0, "min": 0, "max": MAX_IMAGE_INPUTS, "step": 1, "tooltip": TOOLTIPS["num_images"]}),
                },
                "optional": {f"image_{i}": ("IMAGE",) for i in range(1, MAX_IMAGE_INPUTS + 1)},
                "hidden": {"unique_id": "UNIQUE_ID"},
            }

        RETURN_TYPES = ("IMAGE",)
        RETURN_NAMES = ("image",)
        FUNCTION = "generate"
        CATEGORY = "HiDream O1"

        def generate(self, num_images: int, unique_id=None, **kwargs):
            refs = _collect_ref_images(kwargs, num_images)
            sampler_kwargs = {
                key: kwargs[key] if key in kwargs else SAMPLER_INPUT_DEFAULTS[key]
                for key in _sampler_required_inputs()
            }
            return _run_sampler(**sampler_kwargs, refs=refs, unique_id=unique_id)


NODE_CLASS_MAPPINGS = {
    "HiDreamO1ModelLoader": HiDreamO1ModelLoader,
    "HiDreamO1Conditioning": HiDreamO1Conditioning,
    "HiDreamO1Lora": HiDreamO1Lora,
    "HiDreamO1DevSmoothing": HiDreamO1DevSmoothing,
    "HiDreamO1DatasetMaker": HiDreamO1DatasetMaker,
    "HiDreamO1TrainConfig": HiDreamO1TrainConfig,
    "HiDreamO1LoraTrainer": HiDreamO1LoraTrainer,
    "HiDreamO1Sampler": HiDreamO1Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamO1ModelLoader": "HiDream O1 Model Loader",
    "HiDreamO1Conditioning": "HiDream O1 Conditioning",
    "HiDreamO1Lora": "HiDream O1 LoRA",
    "HiDreamO1DevSmoothing": "HiDream O1 Dev Smoothing",
    "HiDreamO1DatasetMaker": "HiDream O1 Dataset Maker",
    "HiDreamO1TrainConfig": "HiDream O1 Train Config",
    "HiDreamO1LoraTrainer": "HiDream O1 LoRA Trainer",
    "HiDreamO1Sampler": "HiDream O1 Sampler",
}
