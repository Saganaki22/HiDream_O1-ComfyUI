# HiDream_O1-ComfyUI

[English](README.md)

**HiDream O1 图像生成 ComfyUI 自定义节点**：在本地运行 HiDream O1，支持文本生成、参考图像、BF16/FP16/FP32/FP8 模型加载、FlashAttention、SageAttention、生成预览，以及 ComfyUI DynamicVRAM/Aimdo 集成。

[![Demo](https://img.shields.io/badge/Demo-HiDream--O1--Image-green)](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
[![GitHub](https://img.shields.io/badge/GitHub-Saganaki22%2FHiDream__O1--ComfyUI-black)](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## 功能特性

- 在 ComfyUI 中直接运行 HiDream O1 图像生成
- 支持纯文本生成和参考图像工作流
- 采样器节点支持动态显示 `image_1` 到 `image_12`
- 支持 Dev 布局条件控制，可通过 JSON bbox 输入
- `keep_image1_aspect` 可根据 `image_1` 自动匹配输出宽高比
- 支持 BF16、FP16、FP32、FP8 E4M3FN、FP8 E5M2 加载选项
- 支持混合 FP8 权重，并使用 ComfyUI manual-cast 风格的安全计算路径
- 支持 FlashAttention、SageAttention、PyTorch SDPA 注意力后端
- 通过 ComfyUI 进度条显示生成预览
- Dev/Dev-2604 patch 网格平滑节点，用于减少明显 tile 接缝
- 支持与 AI Toolkit 对齐的 HiDream O1 LoRA 训练节点
- 支持 ComfyUI 模型管理、卸载、DynamicVRAM、Aimdo/VBAR

## 安装

### 方式一：ComfyUI Manager

在 ComfyUI Manager 中搜索 `HiDream O1` 或 `HiDream_O1-ComfyUI` 并安装。

### 方式二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/HiDream_O1-ComfyUI.git
cd HiDream_O1-ComfyUI
python -m pip install -r requirements.txt
```

安装或更新后请重启 ComfyUI。

**建议 `transformers` 版本：4.57.1 到 5.3。** 更高版本可能出现兼容性问题。

HiDream 2026 年 5 月 13 日的上游更新说明不建议使用 PyTorch 2.9.x，因为 Qwen3-VL 存在相关问题。本节点检测到 2.9.x 时会输出警告。

## 模型配置

下载完整模型文件夹，并放入 `ComfyUI/models/diffusion_models/`：

| 模型版本 | 显存参考 | 下载链接 |
|----------|----------|----------|
| Full BF16 | ~18-20 GB | [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) |
| Full FP16 | ~18-20 GB | [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) |
| Full FP8 | ~10-11 GB | [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) |
| Dev 2604 BF16 | ~18-20 GB | [drbaph/HiDream-O1-Image-Dev-2604-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-BF16) |
| Dev 2604 FP16 | ~18-20 GB | [drbaph/HiDream-O1-Image-Dev-2604-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP16) |
| Dev 2604 FP8 | ~10-11 GB | [drbaph/HiDream-O1-Image-Dev-2604-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP8) |
| Dev BF16 | ~18-20 GB | [drbaph/HiDream-O1-Image-Dev-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16) |
| Dev FP16 | ~18-20 GB | [drbaph/HiDream-O1-Image-Dev-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16) |
| Dev FP8 | ~10-11 GB | [drbaph/HiDream-O1-Image-Dev-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8) |

**示例：FP8，最低显存占用**

1. 打开 [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
2. 下载**完整模型文件夹**，不要只下载 `model.safetensors`
3. 放到 `ComfyUI/models/diffusion_models/HiDream-O1-Image-fp8/`

模型文件夹必须包含 Hugging Face 支持文件：

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

原始分片格式也支持，但文件夹中必须同时包含 `model.safetensors.index.json` 和所有分片文件。

模型加载器始终显示内置转换模型选择：Full/Dev BF16、FP16、FP8，以及 Dev-2604 BF16、FP16、FP8。如果选中的模型已经在本地存在，就直接使用本地文件夹。如果本地不存在，开启 `download_if_missing` 后会把选中的模型下载到 `ComfyUI/models/diffusion_models`。

本地文件夹匹配不区分大小写，所以 `HiDream-O1-Image-Dev-FP8`、`hidream-o1-image-dev-fp8` 和默认目标文件夹大小写都会匹配到同一个内置模型。加载器下拉框只显示内置 HiDream O1 模型选择。

### 上游伪影说明

原始 Full 版 HiDream O1 可能出现网格伪影或参考图生成伪影。上游 issue 中，HiDream 开发者建议尝试 Dev 模型，因为 Dev 应该有更少的网格伪影，同时说明参考图生成还会继续改进：[HiDream-ai/HiDream-O1-Image issue #1](https://github.com/HiDream-ai/HiDream-O1-Image/issues/1#issuecomment-4412738522)。

一般来说，Full 模型更适合真实感、摄影感和细节表现。Dev 模型速度更快，通常更适合插画、数字设计，也更容易减少网格/伪影问题，但它对采样器和分辨率设置更敏感。

| 版本 | 精度选择 | Hugging Face 仓库 | 目标文件夹 |
|------|----------|-------------------|------------|
| Full | `auto`、`bf16`、`fp32` | [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) | `HiDream-O1-Image-bf16` |
| Full | `fp16` | [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) | `HiDream-O1-Image-fp16` |
| Full | `fp8_e4m3fn`、`fp8_e5m2` | [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) | `HiDream-O1-Image-fp8` |
| Dev 2604 | `auto`、`bf16`、`fp32` | [drbaph/HiDream-O1-Image-Dev-2604-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-BF16) | `HiDream-O1-Image-Dev-2604-bf16` |
| Dev 2604 | `fp16` | [drbaph/HiDream-O1-Image-Dev-2604-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP16) | `HiDream-O1-Image-Dev-2604-fp16` |
| Dev 2604 | `fp8_e4m3fn`、`fp8_e5m2` | [drbaph/HiDream-O1-Image-Dev-2604-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP8) | `HiDream-O1-Image-Dev-2604-fp8` |
| Dev | `auto`、`bf16`、`fp32` | [drbaph/HiDream-O1-Image-Dev-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16) | `HiDream-O1-Image-Dev-bf16` |
| Dev | `fp16` | [drbaph/HiDream-O1-Image-Dev-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16) | `HiDream-O1-Image-Dev-fp16` |
| Dev | `fp8_e4m3fn`、`fp8_e5m2` | [drbaph/HiDream-O1-Image-Dev-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8) | `HiDream-O1-Image-Dev-fp8` |

## 节点

### HiDream O1 Model Loader

加载本地 HiDream O1 模型文件夹，并返回由 ComfyUI 管理的模型句柄。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `HiDream-O1-Image-BF16` | 内置 HiDream O1 模型选择 |
| `precision` | `auto` | 自动检测 safetensors dtype，或强制使用 `bf16`、`fp16`、`fp32`、`fp8_e4m3fn`、`fp8_e5m2` |
| `attention` | `auto` | 可选 `auto`、`flash`、`sdpa`、`sage` |
| `download_if_missing` | `false` | 当选中的内置模型本地不存在时，下载该模型 |

### HiDream O1 Conditioning

为采样器创建提示词条件。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 电影感人像提示词 | 正向提示词 |
| `negative_prompt` | 空 | Full 模式下，当 `guidance_scale > 1.0` 时作为无条件 CFG 分支；Dev 模式忽略 CFG |

### HiDream O1 LoRA

在模型加载器和采样器之间应用 LoRA：

```text
HiDream O1 Model Loader -> HiDream O1 LoRA -> HiDream O1 Sampler
```

LoRA 下拉框会读取 `ComfyUI/models/loras/`，包括符号链接文件夹中的受支持 LoRA 文件。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_name` | 没有 LoRA 时为 `None` | LoRA 文件 |
| `strength` | `1.0` | 模型强度，范围 `-10.0` 到 `10.0`；`0` 表示不启用 LoRA |

### HiDream O1 Dev Smoothing

在模型加载器或 LoRA 节点与采样器之间加入 patch 网格平滑：

```text
HiDream O1 Model Loader -> HiDream O1 Dev Smoothing -> HiDream O1 Sampler
HiDream O1 Model Loader -> HiDream O1 LoRA -> HiDream O1 Dev Smoothing -> HiDream O1 Sampler
```

该节点只支持 Dev 和 Dev-2604 模型文件夹。它会在最后几步降噪中运行额外的 shifted patch 预测，再混合回 latent patch 网格，以减少可见接缝。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `steps` | `4` | 最后多少个降噪步启用平滑；`0` 表示关闭 |
| `strength` | `0.5` | shifted patch 预测的混合强度 |
| `schedule` | `constant` | 平滑强度随 step 变化的方式 |
| `shift_mode` | `rotate` | patch 网格偏移模式 |
| `adaptive_threshold` | `0.0` | seam intensity 低于该值时跳过平滑；`0` 表示关闭跳过检查 |
| `multiscale` | `false` | 额外加入更小的 patch 网格偏移 |
| `cfg_aware` | `false` | CFG 启用时也平滑 uncond 分支；会增加额外 forward |

### HiDream O1 LoRA 训练

ComfyUI 内置实验性的文本到图像 LoRA 训练节点：

```text
HiDream O1 Dataset Maker -> HiDream O1 Train Config -> HiDream O1 LoRA Trainer
```

当前训练器只支持图片/字幕数据集，尚未接入参考图、编辑或 IP 训练流程。

数据集文件夹格式：

```text
my_dataset/
  image_001.png
  image_001.txt
  image_002.jpg
  image_002.txt
```

每张图片需要同名 `.txt` 字幕文件。Dataset Maker 会写出 `train.jsonl`，Trainer 会读取该 manifest。

训练默认值与 AI Toolkit 2026 年 5 月的 HiDream O1 训练路径对齐：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_model_name` | `HiDream-O1-Image-BF16` | Full O1 BF16 权重 |
| `resolution` | `1024` | 图片会缩放/裁剪到 patch 对齐的训练尺寸 |
| `target_preset` | `aitoolkit` | 训练 linear-like 层，但排除 `lm_head`、`patch_embed`、`visual` |
| `loss_target` | `velocity` | 将模型的 x0 预测转换为 flow velocity 后计算 loss |
| `noise_scale` | `8.0` | 与 AI Toolkit HiDream O1 flow scheduler 的噪声缩放一致 |
| `timestep_type` | `linear` | AI Toolkit O1 默认值 |
| `max_loss` | `1.0` | 像 AI Toolkit 一样限制极端 loss spike |
| `lora_rank` / `lora_alpha` | `32` / `32` | AI Toolkit 风格的 linear LoRA 默认值 |
| `weight_decay` | `0.0001` | AdamW 权重衰减默认值 |
| `save_dtype` | `bf16` | LoRA checkpoint tensor dtype |
| `max_steps` | `3000` | 总训练步数 |
| `save_every_steps` | `250` | checkpoint 保存间隔 |

训练器会使用 `noise_scale=8.0` 添加 scaled noise，将 noisy image patches 输入 Qwen-VL 模型，把模型的 x0 预测转换成 velocity-equivalent prediction，并用 `noise * noise_scale - image` 作为训练目标。训练会阻塞 ComfyUI 队列。训练请使用 Full 模型；Dev/Dev-2604 是蒸馏推理模型，不在训练节点中开放。

更完整的设置和调参说明见英文文档：[HiDream O1 training notes](docs/HIDREAM_O1_TRAINING.md)。

### HiDream O1 Sampler

运行模型并输出 ComfyUI `IMAGE`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_type` | `auto` | 如果模型文件夹名包含 `dev`，自动使用 Dev 配方，否则使用 Full 配方 |
| `width` | `2048` | 请求输出宽度；内部会对齐到支持的 patch 分辨率 |
| `height` | `2048` | 请求输出高度；内部会对齐到支持的 patch 分辨率 |
| `steps` | `0` | `0` 表示自动：Full 为 50 步；Dev 始终使用上游固定 28 步时间表 |
| `seed` | `42` | 随机种子 |
| `guidance_scale` | `5.0` | Full 模式 CFG 强度；Dev 模式忽略此项并使用 `0.0` |
| `shift` | `-1.0` | `-1` 表示自动：Full 为 `3.0`，Dev 为 `1.0` |
| `noise_scale_start` | `7.5` | 初始噪声缩放 |
| `noise_scale_end` | `7.5` | 最终噪声缩放 |
| `noise_clip_std` | `2.5` | 噪声裁剪标准差 |
| `dev_editing_scheduler` | `flow_match` | Dev 单参考图编辑模式默认 scheduler；仍可选择 `flash` |
| `layout_bboxes` | 空 | 可选 JSON 字符串或 JSON 文件路径，用于参考图布局控制 |
| `preview_every` | `4` | 每 N 步发送一次解码预览；`0` 禁用预览 |
| `keep_image1_aspect` | `false` | 仅在连接 `image_1` 时生效 |
| `force_offload` | `false` | 生成完成后立即卸载模型 |
| `image` | `0` | 动态参考图数量，范围 `0` 到 `12` |

参考图输入是可选的。`image=0` 表示纯文本生成；增大数值会显示 `image_1`、`image_2`，最多到 `image_12`。

## 精度说明

`auto` 会从 safetensors 文件检测模型存储 dtype。原生混合 FP8 文件夹中，大型矩阵权重应为 `float8_e4m3fn`，归一化、偏置等小张量应保留 BF16/FP16。

不要把 `config.json` 写成 `float8_e4m3fn`。Transformers 可能会尝试把 FP8 设为 PyTorch 全局默认 dtype，从而加载失败。请保持 config dtype 为 `bfloat16`；本节点会直接从 safetensors 张量检测 FP8。

加载器只显示普通 FP8 选项。

## 采样器

采样器根据模型类型自动选择调度器：

| 模型类型 | 调度器 | 说明 |
|----------|--------|------|
| Full | `FlowUniPCMultistepScheduler` | 高阶求解器，细节更多 |
| Dev 文生图/主体参考 | `FlashFlowMatchEulerDiscreteScheduler` | 自定义 Euler，带噪声缩放，按少步数蒸馏模型调校 |
| Dev 单参考图编辑 | 默认 `FlowMatchEulerDiscreteScheduler` | 对齐 2026 年 5 月 13 日上游 Dev editing scheduler 更新；仍可选择 `flash` |

当 `model_type=auto` 时，节点会检查模型文件夹名是否包含 `dev`。包含则使用 Dev 配方，否则使用 Full 配方。

Dev 配方与上游保持一致：固定 28 步时间表、guidance `0.0`、shift `1.0`，使用 `flash` 时噪声默认值为 `7.5 / 7.5 / 2.5`。如果 Dev 结果在最后几步附近出现噪点、颜色异常或发白/褪色，请先把 `noise_scale_start`、`noise_scale_end`、`noise_clip_std` 恢复为这些默认值，使用 `flash` 或 `auto` 注意力后端，并把输出固定到内部支持的分辨率之一：`2048x2048`、`2304x1728`、`1728x2304`、`2560x1440`、`1440x2560`、`2496x1664`、`1664x2496`、`3104x1312`、`1312x3104`、`2304x1792`、`1792x2304`。上游建议编辑任务使用 Full 模型。

## 注意力后端

| 选项 | 说明 |
|------|------|
| `auto` | FlashAttention 可用时使用 FlashAttention，否则使用 SDPA |
| `flash` | 强制使用 FlashAttention，需要已安装可用的 FlashAttention |
| `sage` | 强制使用 SageAttention，需要 `sageattention` 包 |
| `sdpa` | 使用 PyTorch scaled dot-product attention |

## 链接

- 在线演示：[HiDream-O1-Image](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
- Dev 2604 BF16 模型：[drbaph/HiDream-O1-Image-Dev-2604-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-BF16)
- Dev 2604 FP16 模型：[drbaph/HiDream-O1-Image-Dev-2604-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP16)
- Dev 2604 FP8 模型：[drbaph/HiDream-O1-Image-Dev-2604-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-2604-FP8)
- Full BF16 模型：[drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16)
- Full FP16 模型：[drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16)
- Full FP8 模型：[drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
- Dev BF16 模型：[drbaph/HiDream-O1-Image-Dev-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-BF16)
- Dev FP16 模型：[drbaph/HiDream-O1-Image-Dev-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP16)
- Dev FP8 模型：[drbaph/HiDream-O1-Image-Dev-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-Dev-FP8)
- 上游项目：[HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image)
- 节点仓库：[Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## 许可证

本自定义节点基于 MIT 许可证发布。HiDream O1 模型有独立的许可证和使用条款；再分发或商业使用前，请查看对应 Hugging Face 模型页面。
