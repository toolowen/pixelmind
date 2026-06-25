---
license: apache-2.0
language:
- zh
- en
base_model:
- google/siglip2-base-patch32-256
---
# SigLIP2 ViT-B/32 Vision Encoder (256)

模型来源：[google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256)

仅裁剪保留了 **Vision Encoder** 部分，去除了 Text Encoder 及其他无关权重，并以 **float16** 精度保存为 HuggingFace `SiglipVisionModel` 兼容的 safetensors 格式。

---

Source: [google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256)

Only the **Vision Encoder** is retained. The Text Encoder and all unrelated weights have been removed. Weights are saved in **float16** precision as HuggingFace `SiglipVisionModel`-compatible safetensors format.

---
## Model Info

| Item | Value |
|------|-------|
| Architecture | SigLIP2 ViT-B/32 (fixed resolution) |
| Parameters | 94.6M |
| Hidden Size | 768 |
| Layers | 12 |
| Patch Size | 32 |
| Input Resolution | 256×256 |
| Output Tokens | 64 (8×8) |
| Weight Precision | float16 |
| File Size | ~180MB |

## Usage

```python
from transformers import SiglipVisionModel, SiglipImageProcessor

model = SiglipVisionModel.from_pretrained("path_to/siglip2-base-p32-256-ve")
processor = SiglipImageProcessor.from_pretrained("path_to/siglip2-base-p32-256-ve")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)  # outputs.last_hidden_state: [B, 64, 768]
```