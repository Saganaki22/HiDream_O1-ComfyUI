# HiDream_O1-ComfyUI

[English](README.md)

**HiDream O1 图像生成 ComfyUI 节点** — 本地 HiDream O1 生成，支持文本提示词、可选参考图像、BF16/FP16/FP32/FP8 模型加载、FlashAttention、SageAttention、预览更新，以及 ComfyUI DynamicVRAM/Aimdo 集成。

[![HiDream O1 Model](https://img.shields.io/badge/HuggingFace-HiDream--O1--Image-blue)](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
[![Demo](https://img.shields.io/badge/Demo-HiDream--O1--Image-green)](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
[![GitHub](https://img.shields.io/badge/GitHub-Saganaki22%2FHiDream__O1--ComfyUI-black)](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## 功能特性

- 在 ComfyUI 中直接生成 HiDream O1 图像
- 纯文本和参考图像两种工作流
- 采样器节点支持动态 `image_1` 到 `image_12` 输入
- `keep_image1_aspect` 开关，根据参考图自动适配输出宽高比
- BF16、FP16、FP32、FP8 E4M3FN 和 FP8 E5M2 加载器选项
- FP8 混合权重加载，使用 ComfyUI manual-cast 方式计算
- FlashAttention、SageAttention 和 PyTorch SDPA 注意力后端
- 通过 ComfyUI 采样器进度条实时预览
- ComfyUI 模型管理、卸载、DynamicVRAM 和 Aimdo/VBAR 支持

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

**建议 `transformers` 版本：4.57.1 – 5.3**（更高版本可能存在兼容性问题）。

## 模型配置

从下方链接下载完整的模型文件夹，放置到 `ComfyUI/models/diffusion_models/` 中：

| 精度 | 显存占用 | 下载链接 |
|------|----------|----------|
| BF16 / FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) |
| FP16 | ~18–20 GB | [drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) |
| FP8 | ~10–11 GB | [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) |

> **Dev 模型量化版本即将推出** — 正在处理中。

**示例 — FP8（最低显存）：**

1. 前往 [drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
2. 下载**完整的模型文件夹**（所有文件，不只是 safetensors）
3. 放置到 `ComfyUI/models/diffusion_models/HiDream-O1-Image-fp8/`

文件夹必须包含完整的 Hugging Face 支持文件：

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

如果文件夹包含 `model.safetensors.index.json` 和所有分片文件，原始分片格式也支持。

如果启用了 `download_if_missing` 并指定了精度，加载器会使用或下载 `ComfyUI/models/diffusion_models` 下对应的文件夹。`precision=auto` 时，仅在本地不存在 HiDream O1 文件夹时下载 BF16。

| 精度 | Hugging Face 仓库 | 目标文件夹 |
|------|-------------------|------------|
| `auto`、`bf16`、`fp32` | [`drbaph/HiDream-O1-Image-BF16`](https://huggingface.co/drbaph/HiDream-O1-Image-BF16) | `HiDream-O1-Image-bf16` |
| `fp16` | [`drbaph/HiDream-O1-Image-FP16`](https://huggingface.co/drbaph/HiDream-O1-Image-FP16) | `HiDream-O1-Image-fp16` |
| `fp8_e4m3fn`、`fp8_e4m3fn_fast`、`fp8_e5m2` | [`drbaph/HiDream-O1-Image-FP8`](https://huggingface.co/drbaph/HiDream-O1-Image-FP8) | `HiDream-O1-Image-fp8` |

## 节点

### HiDream O1 模型加载器

加载本地 HiDream O1 模型文件夹，返回 ComfyUI 管理的模型句柄。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | 第一个发现的文件夹 | 完整的 HiDream O1 模型文件夹 |
| `precision` | `auto` | 自动检测 safetensors 数据类型，或强制 `bf16`、`fp16`、`fp32`、`fp8_e4m3fn`、`fp8_e4m3fn_fast`、`fp8_e5m2` |
| `attention` | `auto` | `auto`、`flash`、`sdpa` 或 `sage` |
| `download_if_missing` | `false` | 指定精度时，使用或下载对应的 drbaph BF16/FP16/FP8 仓库；`auto` 时仅在本地无模型时下载 BF16 |

### HiDream O1 条件节点

为采样器创建提示词条件。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 电影感人像提示词 | 生成文本指令 |
| `negative_prompt` | 空 | 全量模式下 `guidance_scale` 大于 `1.0` 时作为无条件 CFG 分支使用的负向提示词；Dev 模式忽略 CFG |

### HiDream O1 采样器

运行模型并输出 ComfyUI `IMAGE`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_type` | `auto` | 如果模型文件夹名包含 `dev` 则使用 dev 设置，否则使用 full 设置 |
| `width` | `2048` | 请求的输出宽度；内部会对齐到支持的 patch 分辨率 |
| `height` | `2048` | 请求的输出高度；内部会对齐到支持的 patch 分辨率 |
| `steps` | `0` | `0` 表示自动：full 为 50，dev 为 28 |
| `seed` | `42` | 随机种子 |
| `guidance_scale` | `5.0` | 全量模式的 CFG 缩放；dev 模式忽略 CFG |
| `shift` | `-1.0` | `-1` 表示自动：full 为 3.0，dev 为 1.0 |
| `noise_scale_start` | `7.5` | 初始噪声缩放 |
| `noise_scale_end` | `7.5` | 最终噪声缩放 |
| `noise_clip_std` | `2.5` | 噪声裁剪标准差 |
| `preview_every` | `4` | 每 N 步发送解码预览；`0` 禁用预览 |
| `keep_image1_aspect` | `false` | 仅在 `image_1` 已连接时生效 |
| `force_offload` | `false` | 生成完成后立即卸载模型 |
| `image` | `0` | 动态参考图像数量，从 `0` 到 `12` |

参考图像输入为可选项。设置 `image` 为 `0` 进行纯文本生成，增加数值以显示 `image_1`、`image_2`，最多到 `image_12`。

## 精度说明

`auto` 从 safetensors 文件检测模型存储数据类型。对于原生混合 FP8 文件夹，大型矩阵权重应为 `float8_e4m3fn`，而小型张量（如归一化和偏置）保持 BF16/FP16。

不要将 `config.json` 设置为 `float8_e4m3fn`。Transformers 可能会尝试将 FP8 用作 PyTorch 全局默认数据类型，导致失败。请保持 config 数据类型为 `bfloat16`；本节点从 safetensors 张量本身检测 FP8。

使用 `fp8_e4m3fn_fast` 请求 ComfyUI 的快速 FP8 运算路径（需要当前 GPU、PyTorch 和 ComfyUI 构建支持）。否则模型仍以 FP8 存储大型权重，并使用 ComfyUI manual-cast 方式安全计算。

## 采样器

采样器根据模型类型自动选择调度器：

| 模型类型 | 调度器 | 说明 |
|----------|--------|------|
| Full（`auto`） | `FlowUniPCMultistepScheduler` | 高阶求解器，生成更多细节 |
| Dev | `FlashFlowMatchEulerDiscreteScheduler` | 自定义 Euler，内置噪声缩放，针对少步数优化 |

当 `model_type` 为 `auto` 时，会检查文件夹名是否包含 `dev` — 如果未找到，则使用 UniPC 的全量模型路径。

## 注意力后端

| 选项 | 说明 |
|------|------|
| `auto` | FlashAttention 可用时使用，否则使用 SDPA |
| `flash` | 需要 FlashAttention |
| `sage` | 需要 `sageattention` 包 |
| `sdpa` | 使用 PyTorch 缩放点积注意力 |

## 链接

- 演示：[HiDream-O1-Image](https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image)
- BF16 模型：[drbaph/HiDream-O1-Image-BF16](https://huggingface.co/drbaph/HiDream-O1-Image-BF16)
- FP16 模型：[drbaph/HiDream-O1-Image-FP16](https://huggingface.co/drbaph/HiDream-O1-Image-FP16)
- FP8 模型：[drbaph/HiDream-O1-Image-FP8](https://huggingface.co/drbaph/HiDream-O1-Image-FP8)
- 上游项目：[HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image)
- 节点仓库：[Saganaki22/HiDream_O1-ComfyUI](https://github.com/Saganaki22/HiDream_O1-ComfyUI)

## 许可证

本自定义节点基于 MIT 许可证发布。HiDream O1 模型有其独立的许可证和使用条款；在再分发或商业使用前，请查阅上游 Hugging Face 模型页面。
