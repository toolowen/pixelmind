# PixelMind on AutoDL — Complete Training Guide

> Tested for: RTX 4090 24GB | PyTorch 2.3.0+ | CUDA 12.1+ | Python 3.10

---

## 0. Recommended AutoDL Configuration

| Item | Selection |
|------|-----------|
| GPU | RTX 4090 24GB (单卡) |
| Image | PyTorch 2.3.0 / CUDA 12.1 / Python 3.10 |
| System Disk | ≥ 50GB (系统 + 模型 ~15GB + 数据 ~10GB) |
| Data Disk | 可选, 有的话挂到 `./data` |

登录后先验证环境：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.3.0+cu121  True
```

---

## 1. Environment Setup

```bash
# 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 安装核心依赖
pip install transformers==4.45.0 datasets pyarrow Pillow swanlab numpy accelerate \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# HuggingFace Hub + ModelScope 客户端 (下载数据用)
pip install huggingface_hub modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证
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
import shutil, os

snapshot_download('gongjb/minimind', local_dir='./_tmp',
                  allow_patterns=['tokenizer.json', 'tokenizer_config.json'])
for f in os.listdir('./_tmp'):
    if 'tokenizer' in f:
        shutil.move(f'./_tmp/{f}', f'./model/tokenizer/{f}')
shutil.rmtree('./_tmp', ignore_errors=True)
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

### 3.3 Reward Model (for GRPO stage, ~3.6GB)

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

## 4. Download Training Data

### 4.1 LLM Data (JSONL, from MiniMind ModelScope)

```bash
mkdir -p data/text

python -c "
from modelscope import snapshot_download
import shutil, os

snapshot_download('gongjb/minimind', local_dir='./_dl',
                  allow_patterns=['pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl'])

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

**注意：** ALLaVA-4V 总计 ~8.7GB，下载较慢。如果时间紧张，可以先用 MiniMind-V 自带的小样本跳过 Stage 3-4，直接到 Stage 5 GRPO（但 GRPO 也需要 Parquet 格式的 VLM 数据）。

---

## 5. Smoke Test Before Training

```bash
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

# 5.3 Data loader test (1 batch, 验证 Parquet 数据可读)
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

预计 VRAM 峰值 (RTX 4090 24GB):

| Stage | VRAM | Time | 关键数据 |
|-------|------|------|----------|
| 1. LLM Pretrain | ~3GB | ~30 min | pretrain_t2t_mini.jsonl |
| 2. LLM SFT | ~5GB | ~20 min | sft_t2t_mini.jsonl |
| 3. VLM Pretrain | ~8GB | ~1-2 h | pretrain_i2t.parquet |
| 4. VLM SFT | ~10GB | ~2-3 h | sft_i2t.parquet |
| 5. VLM GRPO ⭐ | ~18GB | ~3-5 h | sft_i2t.parquet |

### 6.1 Stage 1: LLM Pretrain

**训练内容**：Next-token prediction, from scratch  
**超参**：lr=5e-4, batch=32, max_seq=340, accum=8

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

tail -f logs/llm_pretrain.log
```

**预期 loss**：9.x → 3.5-4.5  
**输出**：`out/pretrain_768.pth`

### 6.2 Stage 2: LLM SFT

**训练内容**：Instruction tuning on multi-turn conversations  
**超参**：lr=1e-5, batch=16, max_seq=768

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

tail -f logs/llm_sft.log
```

**预期 loss**：3.x → 2.5-3.5  
**输出**：`out/sft_768.pth`

### 6.3 Stage 3: VLM Pretrain (Projector Alignment)

**训练内容**：视觉投影层对齐, LLM + vision encoder 冻结  
**超参**：lr=4e-4, batch=4, max_seq=450, freeze_llm=2

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

tail -f logs/vlm_pretrain.log
```

**预期 loss**：9.x → 4.0-5.0  
**输出**：`out/pretrain_vlm_768.pth` (不含 vision_encoder, ~130MB)

### 6.4 Stage 4: VLM SFT

**训练内容**：多模态指令微调, projector + LLM 首尾层  
**超参**：lr=5e-6, batch=4, max_seq=768, freeze_llm=1

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

tail -f logs/vlm_sft.log
```

**预期 loss**：5.x → 3.0-4.0  
**输出**：`out/sft_vlm_768.pth`

### 6.5 Stage 5: VLM GRPO ⭐ (Core Innovation)

**训练内容**：Group Relative Policy Optimization with visual inputs  
**超参**：lr=3e-7, batch=2, num_generations=4, max_gen_len=256, beta=0.1, loss_type=cispo

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

tail -f logs/vlm_grpo.log
```

**预期 Reward ↑**：训练过程中 reward 均值缓慢上升  
**输出**：`out/grpo_vlm_768.pth`

---

## 7. VLM GRPO Reward Design (How It Works)

### 7.1 Pipeline Flow (per step)

```
Dataset → VLMRLDataset (image + prompt, no answer)
   → Collate (pad images + text)
   → Re-tokenize (left-padded for generation)
   → RolloutEngine.rollout() WITH pixel_values
        → model.generate() generates 4 responses per image
        → compute_per_token_logps() records probabilities
   → Reward computation:
        (a) Text quality: response length ±0.5
        (b) Format: <think> tag validity ±1.0
        (c) Repetition penalty: trigram overlap ≤0.5
        (d) Reward Model: internlm2-1_8b-reward ∈ [-3,3]
   → Policy forward WITH pixel_values → log-probs
   → Reference forward WITH pixel_values → ref log-probs
   → GRPO: group-adv normalized, KL penalty, ratio clipping
   → CISPO loss → backward → optimizer step
```

### 7.2 Reward Components

| Reward Signal | Type | Score Range | Source |
|--------------|------|-------------|--------|
| **Response Length** | 文本质量 | ±0.5 | 规则 (20~800 chars +0.5, else -0.5) |
| **Thinking Format** | 结构规范 | ±1.25 | 规则 (<think> 标签检查) |
| **Repetition Penalty** | 文本质量 | −0~0.5 | 规则 (trigram overlap) |
| **Reward Model** | 文本质量 | −3.0~+3.0 | internlm2-1_8b-reward |

**为什么没有视觉 reward (CLIPScore / Qwen2.5-VL)？**

- **CLIPScore** 只度量图文的粗粒度相似度，语义理解弱
- **Qwen2.5-VL-3B 打分** 是可行的增强方向，但会显著增加 GRPO 的推理开销 (~8× per step)
- 当前设计是**可运行的 MVP**——先用文本 reward 跑通训练流程，视觉 reward 作为后续升级项
- 这也正是简历上的技术亮点：**"设计了多维度 reward 函数用于 VLM 强化学习对齐，并在单卡上完成了训练框架验证"**

### 7.3 GRPO Hyperparameters Explained

| 参数 | 值 | 解释 |
|------|-----|------|
| `num_generations=4` | 每组 4 个回答 | 越多统计越稳, 但显存×生成次数 |
| `beta=0.1` | KL 惩罚系数 | 防止策略偏离 reference 太远 |
| `loss_type=cispo` | CISPO loss | 仅裁剪上界, 允许策略激进提升 |
| `epsilon_high=5.0` | CISPO 上界 | 5× ratio cap, 比标准 PPO (2.0) 宽松 |
| `learning_rate=3e-7` | 学习率 | RL 阶段极低 lr, 避免破坏 SFT 能力 |
| `max_gen_len=256` | 生成上限 | 控制显存和训练速度 |
| `temperature=0.8` | 采样温度 | 平衡多样性和质量 |

---

## 8. Monitor Training

```bash
# 查看所有后台进程
jobs -l

# 实时查看某个阶段的 log
tail -f logs/llm_pretrain.log
tail -f logs/llm_sft.log
tail -f logs/vlm_pretrain.log
tail -f logs/vlm_sft.log
tail -f logs/vlm_grpo.log

# GPU 使用率
nvidia-smi

# SwanLab 可视化 (log 里会打印 URL)
grep -i "swanlab\|dashboard\|view" logs/llm_pretrain.log
```

**Attention 关键字**（说明训练正常）:
```
"Epoch:[1/2](100/xxx), loss: x.xxxx, lr: x.xxxx"  ← 正常
"CUDA out of memory"                                ← 需要减小 batch/gen_len
"NaN" in loss                                       ← 降低 lr 或增大 beta
```

---

## 9. Evaluate Results

### 9.1 检查 Loss 收敛

```bash
grep "loss:" logs/llm_pretrain.log | tail -5
grep "loss:" logs/llm_sft.log | tail -5
grep "loss:" logs/vlm_pretrain.log | tail -5
grep "loss:" logs/vlm_sft.log | tail -5
grep "Reward:" logs/vlm_grpo.log | tail -10
```

### 9.2 LLM Chat 测试

```bash
python -m pixelmind.eval.chat_llm --weight sft
```

### 9.3 VLM Chat 测试

```bash
mkdir -p dataset/eval_images
# 上传几张测试图片 (scp / wget / curl)

python -m pixelmind.eval.chat_vlm \
    --weight sft_vlm \
    --image_dir dataset/eval_images/ \
    --temperature 0.7
```

### 9.4 GRPO 前后对比

```bash
# SFT only
python -m pixelmind.eval.chat_vlm --weight sft_vlm --image_dir dataset/eval_images/

# GRPO (RL aligned)
python -m pixelmind.eval.chat_vlm --weight grpo_vlm --image_dir dataset/eval_images/
```

---

## 10. OOM Troubleshooting

### Stage 1-2 (LLM): 基本不会 OOM
4090 24GB 跑 64M 纯 LLM 绰绰有余。如遇到 OOM：
```bash
--batch_size 16 --accumulation_steps 2     # 等效 batch 32
```

### Stage 3-4 (VLM SFT): 可能 OOM
```bash
--batch_size 2 --accumulation_steps 2     # 等效 batch 4
--max_seq_len 512                          # 减短序列
```

### Stage 5 (VLM GRPO): 最可能 OOM
按以下顺序依次尝试：
```bash
# 1. 减少生成数量
--num_generations 2              # 从 4 减到 2

# 2. 减少生成长度
--max_gen_len 128               # 从 256 减到 128

# 3. 减小 batch
--batch_size 1                  # 从 2 减到 1

# 4. 缩短 prompt
--max_seq_len 512               # 从 768 减到 512

# 5. 关闭 torch.compile (如果开着)
--use_compile 0
```

---

## 11. Expected Results

| Stage | Expected Loss Range | Output File |
|-------|--------------------|-------------|
| LLM Pretrain | 3.5 - 4.5 | `out/pretrain_768.pth` (~132MB) |
| LLM SFT | 2.5 - 3.5 | `out/sft_768.pth` (~132MB) |
| VLM Pretrain | 4.0 - 5.0 | `out/pretrain_vlm_768.pth` (~134MB) |
| VLM SFT | 3.0 - 4.0 | `out/sft_vlm_768.pth` (~134MB) |
| VLM GRPO ⭐ | Reward ↑ | `out/grpo_vlm_768.pth` (~134MB) |

65M 模型的对话能力有限，会有重复和幻觉——这是 MiniMind-V 原论文已知的局限。简历上反而可以作为 **"已知局限 + 改进分析"** 的素材。

---

## 12. Files to Collect for Your Resume

训练结束后收集以下素材：

```bash
# 1. SwanLab loss 曲线 (从 dashboard 截图保存)
# 2. 生成效果对比 (SFT vs GRPO) 截图
ls -lh out/*.pth

# 3. 训练日志摘要
grep -E "Epoch.*loss" logs/llm_pretrain.log | tail -3
grep -E "Epoch.*loss" logs/vlm_sft.log | tail -3
grep -E "Reward" logs/vlm_grpo.log | tail -10

# 4. 编码器对比实验 (optional)
#    python -m pixelmind.eval.benchmarks --compare_encoders
```

---

## Technical Notes

### 为什么用 ALLaVA-4V SFT 数据做 GRPO？

`VLMRLDataset` 和 `VLMDataset` 的**关键区别**：
- `VLMDataset`：`add_generation_prompt=False`，输出 `(input_ids, labels, pixel_values)`
- `VLMRLDataset`：`add_generation_prompt=True`，输出 `{prompt, prompt_ids, pixel_values}`

GRPO 阶段模型需要自己**生成**回答——所以只有图片+问题，没有正确答案。Reward 模型对生成的回答打分，文本质量高 → 高奖励，重复/空洞 → 低奖励。

### 视觉编码器在整个 GRPO 中冻结

Policy model 和 Reference model **共享同一个 vision encoder 实例**。`pixel_values` 在 rollout → policy forward → reference forward 全链路传递，但 encoder 权重不变。Gradient 只更新 LLM + projector。

### CISPO vs GRPO

CISPO (`loss_type=cispo`):
- 仅裁剪 ratio 上界 (`max=epsilon_high`)，保留下界不做 pessimistic clipping
- 用 `clamped_ratio * log_probs` 代替 `min(ratio*adv, clipped*adv)`
- 对小模型更友好——允许策略在 good tokens 上大胆提升概率

标准 GRPO (`loss_type=grpo`):
- 对称裁剪 `±epsilon`，取 min 防止过度更新
- 更保守，适合大模型
