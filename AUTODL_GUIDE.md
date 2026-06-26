# PixelMind on AutoDL — Complete Training Guide

> Tested on: RTX 4090 24GB | PyTorch 2.3.0 | CUDA 12.1 | Python 3.10

---

## 0. Recommended AutoDL Configuration

| Item | Selection |
|------|-----------|
| GPU | RTX 4090 24GB (1×) |
| Image | PyTorch 2.3.0 / CUDA 12.1 / Python 3.10 |
| System Disk | ≥ 50GB (models + data ~15GB) |
| Data Disk | optional, mount to `./data` if provided |

After SSH login, verify:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.3.0+cu121  True
```

---

## 1. Environment Setup

```bash
# Mirror for faster downloads (China)
export HF_ENDPOINT=https://hf-mirror.com

# Install core dependencies
pip install transformers==4.45.0 datasets pyarrow Pillow swanlab numpy accelerate \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Optional: HuggingFace Hub client for dataset download
pip install huggingface_hub modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verify
python -c "import torch, transformers, datasets; print('Environment OK')"
```

---

## 2. Clone & Install PixelMind

```bash
git clone https://github.com/toolowen/pixelmind.git
cd pixelmind
pip install -e .
```

---

## 3. Download Model Resources

### 3.1 Tokenizer (MiniMind BPE, vocab 6400)

```bash
mkdir -p model/tokenizer

python -c "
from modelscope import snapshot_download
snapshot_download('gongjb/minimind', local_dir='./model',
                  allow_patterns=['tokenizer.json', 'tokenizer_config.json'])
# Move files into place
import os, shutil
for f in os.listdir('./model'):
    if 'tokenizer' in f:
        shutil.move(f'./model/{f}', f'./model/tokenizer/{f}')
print('Tokenizer ready')
"
# Verify
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./model/tokenizer')
print(f'Tokenizer OK: vocab_size={tok.vocab_size}')
# Expected: 6400
"
```

### 3.2 SigLIP2 Visual Encoder (~95M, frozen)

```bash
mkdir -p model/siglip2-base-p32-256-ve

python -c "
from transformers import SiglipVisionModel, SiglipImageProcessor
model = SiglipVisionModel.from_pretrained('google/siglip2-base-patch32-256')
model.save_pretrained('./model/siglip2-base-p32-256-ve')
processor = SiglipImageProcessor.from_pretrained('google/siglip2-base-patch32-256')
processor.save_pretrained('./model/siglip2-base-p32-256-ve')
print('SigLIP2 ready')
"
# Verify
python -c "
from transformers import SiglipVisionModel
model = SiglipVisionModel.from_pretrained('./model/siglip2-base-p32-256-ve')
print(f'SigLIP2 OK: hidden_size={model.config.hidden_size}')
# Expected: 768
"
```

---

## 4. Download Training Data

### 4.1 LLM Data (JSONL, from MiniMind ModelScope)

```bash
mkdir -p data/text

python -c "
from modelscope import snapshot_download
import shutil, os

# Download mini datasets (~3GB total)
snapshot_download('gongjb/minimind', local_dir='./_dl',
                  allow_patterns=['pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl'])

# Move to data/text/
for f in ['pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl']:
    shutil.move(f'./_dl/{f}', f'./data/text/{f}')
shutil.rmtree('./_dl', ignore_errors=True)
print('LLM data ready')
"
# Verify
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='./data/text/pretrain_t2t_mini.jsonl', split='train')
print(f'Pretrain samples: {len(ds)}')
ds = load_dataset('json', data_files='./data/text/sft_t2t_mini.jsonl', split='train')
print(f'SFT samples: {len(ds)}')
"
```

### 4.2 VLM Data (Parquet, from ALLaVA-4V HuggingFace)

```bash
mkdir -p data/multimodal

python -c "
from huggingface_hub import snapshot_download
snapshot_download('FreedomIntelligence/ALLaVA-4V',
                  repo_type='dataset',
                  allow_patterns=['pretrain_i2t.parquet', 'sft_i2t.parquet'],
                  local_dir='./data/multimodal')
print('VLM data ready')
"
# Verify
python -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('./data/multimodal/pretrain_i2t.parquet')
print(f'Pretrain rows: {pf.metadata.num_rows}')
pf = pq.ParquetFile('./data/multimodal/sft_i2t.parquet')
print(f'SFT rows: {pf.metadata.num_rows}')
# Expected: ~1.27M pretrain, ~2.9M SFT
"
```

### 4.3 Reward Model (for GRPO, ~3.6GB)

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('Shanghai_AI_Laboratory/internlm2-1_8b-reward',
                  local_dir='./internlm2-1_8b-reward')
print('Reward model ready')
"
# Verify
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('./internlm2-1_8b-reward', trust_remote_code=True)
print('Reward model OK')
"
```

---

## 5. Smoke Test Before Training

```bash
cd ~/pixelmind

# Create log directory
mkdir -p logs

# 5.1 LLM import test
python -c "
from pixelmind.config import PixelMindConfig
from pixelmind.model.llm import PixelMindForCausalLM
cfg = PixelMindConfig(hidden_size=768, num_hidden_layers=8)
m = PixelMindForCausalLM(cfg)
print(f'LLM OK: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')
"

# 5.2 VLM import test
python -c "
from pixelmind.config import PixelMindConfig
from pixelmind.model.vlm import PixelMind
cfg = PixelMindConfig(hidden_size=768, num_hidden_layers=8)
m = PixelMind(cfg, vision_encoder_path='./model/siglip2-base-p32-256-ve')
print(f'VLM OK: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')
"

# 5.3 Data loader test (1 batch)
python -c "
import torch
from transformers import AutoTokenizer
from pixelmind.config import PixelMindConfig
from pixelmind.trainer.model_init import init_vlm_model
from pixelmind.data.mm_dataset import VLMDataset
from pixelmind.data.collate import vlm_collate_fn
from torch.utils.data import DataLoader

cfg = PixelMindConfig(hidden_size=768, num_hidden_layers=8)
model, tokenizer, preprocess = init_vlm_model(
    cfg, from_weight='none',
    vision_encoder_path='./model/siglip2-base-p32-256-ve',
    device='cuda', freeze_llm=2,
)
ds = VLMDataset('./data/multimodal/pretrain_i2t.parquet', tokenizer,
                preprocess=preprocess, max_length=450,
                image_special_token=cfg.image_special_token,
                image_token_len=cfg.image_token_len)
loader = DataLoader(ds, batch_size=2, collate_fn=vlm_collate_fn)
batch = next(iter(loader))
print(f'Data OK: ids={batch[0].shape}, labels={batch[1].shape}')
"
```

---

## 6. Full Training Pipeline

All commands use `nohup` for background execution. Logs written to `logs/`.

Expected VRAM peaks on 4090 24GB:

| Stage | VRAM | Time (mini data) |
|-------|------|------------------|
| 1. LLM Pretrain | ~3GB | ~30 min |
| 2. LLM SFT | ~5GB | ~20 min |
| 3. VLM Pretrain | ~8GB | ~1 h |
| 4. VLM SFT | ~10GB | ~1 h |
| 5. VLM GRPO ⭐ | ~18GB | ~2 h |

### 6.1 LLM Pretrain (from scratch)

```bash
nohup python -m pixelmind.trainer.llm_pretrain \
    --data_path data/text/pretrain_t2t_mini.jsonl \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 340 \
    --accumulation_steps 8 \
    --log_interval 100 \
    --save_interval 500 \
    --use_wandb \
    --wandb_project PixelMind-Pipeline \
    > logs/llm_pretrain.log 2>&1 &

# Monitor
tail -f logs/llm_pretrain.log
```

### 6.2 LLM SFT

```bash
nohup python -m pixelmind.trainer.llm_sft \
    --data_path data/text/sft_t2t_mini.jsonl \
    --from_weight pretrain \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_seq_len 768 \
    --log_interval 100 \
    --save_interval 500 \
    --use_wandb \
    --wandb_project PixelMind-Pipeline \
    > logs/llm_sft.log 2>&1 &
```

### 6.3 VLM Pretrain (projector alignment)

```bash
nohup python -m pixelmind.trainer.vlm_pretrain \
    --data_path data/multimodal/pretrain_i2t.parquet \
    --from_weight sft \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 4e-4 \
    --max_seq_len 450 \
    --freeze_llm 2 \
    --log_interval 100 \
    --save_interval 500 \
    --use_wandb \
    --wandb_project PixelMind-Pipeline \
    > logs/vlm_pretrain.log 2>&1 &
```

### 6.4 VLM SFT

```bash
nohup python -m pixelmind.trainer.vlm_sft \
    --data_path data/multimodal/sft_i2t.parquet \
    --from_weight pretrain_vlm \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --max_seq_len 768 \
    --freeze_llm 1 \
    --log_interval 100 \
    --save_interval 500 \
    --use_wandb \
    --wandb_project PixelMind-Pipeline \
    > logs/vlm_sft.log 2>&1 &
```

### 6.5 VLM GRPO ⭐ (core innovation)

```bash
nohup python -m pixelmind.trainer.vlm_grpo \
    --data_path data/multimodal/sft_i2t.parquet \
    --from_weight sft_vlm \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 3e-7 \
    --num_generations 4 \
    --max_gen_len 256 \
    --max_seq_len 768 \
    --beta 0.1 \
    --loss_type cispo \
    --epsilon 0.2 \
    --epsilon_high 5.0 \
    --reward_model_path ./internlm2-1_8b-reward \
    --log_interval 1 \
    --save_interval 10 \
    --use_wandb \
    --wandb_project PixelMind-Pipeline \
    > logs/vlm_grpo.log 2>&1 &
```

---

## 7. Monitor Training

```bash
# Check all running processes
jobs -l

# Watch a specific log
tail -f logs/llm_pretrain.log

# Check GPU usage
nvidia-smi

# SwanLab dashboard (open in browser)
# The script will print a URL in the log
grep -i "swanlab\|dashboard\|view" logs/llm_pretrain.log
```

---

## 8. Evaluate Results

### 8.1 LLM Evaluation

```bash
# Check convergence: loss should drop from ~9 to ~4 (pretrain), ~3 (SFT)
grep "loss:" logs/llm_pretrain.log | tail -5
grep "loss:" logs/llm_sft.log | tail -5

# Interactive chat
python -m pixelmind.eval.chat_llm --weight sft
```

### 8.2 VLM Evaluation

```bash
# Prepare test images
mkdir -p dataset/eval_images
# Upload your test images (scp, wget, etc.)

# Test VLM (SFT weight)
python -m pixelmind.eval.chat_vlm \
    --weight sft_vlm \
    --image_dir dataset/eval_images/ \
    --temperature 0.7

# Test VLM (GRPO weight, after RL)
python -m pixelmind.eval.chat_vlm \
    --weight grpo_vlm \
    --image_dir dataset/eval_images/ \
    --temperature 0.7
```

---

## 9. OOM Troubleshooting

If you hit CUDA out-of-memory, try these in order:

### VLM Pretrain
```bash
# Halve batch
--batch_size 2 --accumulation_steps 2
```

### VLM SFT
```bash
# Halve batch, reduce seq len
--batch_size 2 --max_seq_len 512
```

### VLM GRPO (most memory-hungry)
```bash
# Reduce generation count
--num_generations 2
# Reduce generation length
--max_gen_len 128
# Halve batch
--batch_size 1
# Remove thinking tag overhead (skip <think> format)
# Run without reward model initially (text-only heuristics)
```

---

## 10. Expected Results

After full pipeline:

| Stage | Expected Loss | Output |
|-------|--------------|--------|
| LLM Pretrain | ~3.5-4.5 | `out/pretrain_768.pth` |
| LLM SFT | ~2.5-3.5 | `out/sft_768.pth` |
| VLM Pretrain | ~4.0-5.0 | `out/pretrain_vlm_768.pth` |
| VLM SFT | ~3.0-4.0 | `out/sft_vlm_768.pth` |
| VLM GRPO | Reward ↑ over steps | `out/grpo_vlm_768.pth` |

The 65M model will produce **basic but meaningful** image descriptions. Hallucinations and repetition are expected — this is known behavior documented in MiniMind-V.

---

## 11. Files to Collect for Your Resume

After training completes:

```bash
# 1. SwanLab loss curves (screenshot from dashboard)
# 2. Generated outputs before/after GRPO (side-by-side comparison)
# 3. Model weights (for demo)
ls -lh out/*.pth

# 4. Training log summary
grep -E "Epoch.*loss" logs/llm_pretrain.log | tail -3
grep -E "Epoch.*loss" logs/llm_sft.log | tail -3
grep -E "Epoch.*loss" logs/vlm_pretrain.log | tail -3
grep -E "Epoch.*loss" logs/vlm_sft.log | tail -3
grep -E "Reward" logs/vlm_grpo.log | tail -10
```

---

## Quick Reference: One-liner Pipeline

```bash
# Run all 5 stages sequentially (when you're confident)
cd ~/pixelmind && mkdir -p logs

for stage in llm_pretrain llm_sft vlm_pretrain vlm_sft vlm_grpo; do
    echo "=== Starting $stage at $(date) ==="
    # ... (copy the command for each stage above)
    echo "=== $stage done at $(date) ==="
done
```
