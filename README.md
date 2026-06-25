# 🧠 PixelMind

<p align="center">
  <b>Train a Tiny VLM from Scratch — 65M params, 2 hours, $3, Single GPU</b>
</p>

<p align="center">
  <a href="https://github.com/jingyaogong/minimind">← MiniMind (LLM)</a> &nbsp;·&nbsp;
  <a href="https://github.com/jingyaogong/minimind-v">← MiniMind-V (VLM inspiration)</a> &nbsp;·&nbsp;
  <a href="https://pixelmind.readthedocs.io">Docs (WIP)</a>
</p>

PixelMind is a **minimal, educational Vision-Language Model** trained entirely from scratch. It proves that a useful multimodal model can be built with ~65M parameters.

Built by merging and refining the [MiniMind](https://github.com/jingyaogong/minimind) (pure LLM) and [MiniMind-V](https://github.com/jingyaogong/minimind-v) (VLM) projects into a single, clean codebase.

---

## ✨ Features

- 🧩 **Pluggable Visual Encoders** — SigLIP2 / DINOv2 / InternViT via abstract interface, with ablation comparison framework
- 📊 **OCR / Document Understanding** — Data-driven: reads text in images via DocVQA / ChartQA training data
- 🚀 **VLM GRPO ** — Group Relative Policy Optimization for VLM: image-conditioned reinforcement learning alignment
- 📝 **Custom Native `generate()`** — Full PyTorch implementation with GQA, RoPE, top-p, temperature, streamer support
- ⚡ **Efficient Training** — DDP multi-GPU, bfloat16, torch.compile, flash attention (via PyTorch SDPA)
- 🏗️ **Clean Package Structure** — Single `pixelmind` package (not two repos), well-organized modules
- 📦 **HuggingFace Compatible** — Save/load as native `.pth` or HuggingFace Transformers format

---

## 🗺️ Training Pipeline

```
LLM Pretrain       LLM SFT         VLM Pretrain       VLM SFT         VLM GRPO ⭐
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ next-token │ ──→ │ instruct │ ──→ │ visual   │ ──→ │ image    │ ──→ │ RL       │
│ prediction │     │ tuning   │     │ align    │     │ instruct │     │ align    │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
  2 epochs         2 epochs        1 epoch          1 epoch          1 epoch
  lr=5e-4          lr=1e-5         lr=4e-4          lr=5e-6          lr=3e-7
  seq=340          seq=768         seq=450          seq=768          seq=768+gen
```

| Stage | What Gets Trained | Output |
|-------|-------------------|--------|
| 1. LLM Pretrain | All LLM layers (from scratch) | `pretrain_768.pth` |
| 2. LLM SFT | All LLM layers | `sft_768.pth` |
| 3. VLM Pretrain | Projector only | `pretrain_vlm_768.pth` |
| 4. VLM SFT | Projector + LLM first/last layers | `sft_vlm_768.pth` |
| 5. VLM GRPO ⭐ | All LLM layers + Projector (RL) | `grpo_vlm_768.pth` |

---

## 🚀 Quick Start

### Installation

```bash
cd pixelmind
pip install -e .
```

### Download Resources

```bash
# LLM tokenizer (from MiniMind)
# Vision encoder (SigLIP2 from HuggingFace)
# Training data (ALLaVA-4V parquet or MiniMind JSONL)

mkdir -p model/tokenizer model/siglip2-base-p32-256-ve data/text data/multimodal
# Download files from ModelScope or HuggingFace...
```

### Train

```bash
# Stage 1: LLM Pretrain
python -m pixelmind.trainer.llm_pretrain --data_path data/text/pretrain.jsonl

# Stage 2: LLM SFT
python -m pixelmind.trainer.llm_sft --from_weight pretrain --data_path data/text/sft.jsonl

# Stage 3: VLM Pretrain (projector alignment)
python -m pixelmind.trainer.vlm_pretrain --from_weight sft --data_path data/multimodal/pretrain_i2t.parquet

# Stage 4: VLM SFT
python -m pixelmind.trainer.vlm_sft --from_weight pretrain_vlm --data_path data/multimodal/sft_i2t.parquet

# Stage 5: VLM GRPO ⭐
python -m pixelmind.trainer.vlm_grpo --from_weight sft_vlm --data_path data/multimodal/grpo.parquet
```

### Chat

```bash
# LLM Chat
python -m pixelmind.eval.chat_llm --weight sft

# VLM Chat (evaluates all images in a directory)
python -m pixelmind.eval.chat_vlm --weight sft_vlm --image_dir ./dataset/eval_images/

# Web Demo
python -m pixelmind.scripts.web_demo --weight sft_vlm
```

---

## 🏗️ Architecture

```
Image (256×256)
    │
    ▼
┌──────────────────────┐
│  Vision Encoder      │  ← SigLIP2 / DINOv2 / InternViT (frozen)
│  ~95M params         │
└──────────┬───────────┘
           │ [B, 64, 768] visual features
           ▼
┌──────────────────────┐
│  MMVisionProjector   │  ← LayerNorm → Linear → GELU → Linear
│  (2-layer MLP)       │
└──────────┬───────────┘
           │ [B, 64, 768] projected tokens
           │
           │  Replace <|image_pad|> markers in hidden states
           ▼
┌──────────────────────────────────────────────────┐
│  PixelMind LLM (Decoder-Only Transformer)        │
│  ┌────────────┐  ┌────────────┐      ┌────────┐ │
│  │ Embedding  │→ │ Block × 8  │ ... →│  Norm  │→│ lm_head
│  │ + Dropout  │  │ (GQA+RoPE  │      │        │ │ → text
│  └────────────┘  │  +SwiGLU)  │      └────────┘ │
│                  └────────────┘                   │
│  ~64M params (trainable)                         │
└──────────────────────────────────────────────────┘
```

**Key specs:**
- 8 decoder layers, d_model=768, 8 Q heads / 4 KV heads (GQA 2:1)
- RoPE position encoding (theta=1e6) with YaRN extrapolation
- SwiGLU activation, Pre-Norm RMSNorm
- BPE tokenizer, vocab=6400
- Image: 64 tokens per image (256×256, patch_size=32)

---

## 📋 Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- transformers ≥ 4.45
- datasets, pyarrow, Pillow

Full list: `requirements.txt`

---

## 🙏 Acknowledgments

PixelMind is built upon the excellent work of:

- [MiniMind](https://github.com/jingyaogong/minimind) — The original 64M LLM from scratch
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — The original VLM extension
- [SigLIP2](https://arxiv.org/abs/2505.14315) — Visual encoder
- [GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) — Training data

## 📄 License

Apache License 2.0 — same as the upstream projects.

## 📝 Citation

```bibtex
@misc{pixelmind,
  title = {PixelMind: Train a Tiny VLM from Scratch},
  author = {},
  year = {2025},
  url = {https://github.com/your/pixelmind},
}
```
