"""
Benchmarks for PixelMind VLM
=============================
Encoder comparison experiments + OCR/VQA benchmark evaluation.

This is the framework for the "pluggable encoder" innovation:
  - Compare different vision encoders (SigLIP2 vs DINOv2 vs InternViT)
  - Evaluate OCR accuracy, description quality, inference speed
  - Generate comparison tables suitable for a resume or paper
"""

import time
import warnings

import torch
from tqdm import tqdm

from ..config import PixelMindConfig
from ..model.vision import build_vision_encoder
from ..model.vlm import PixelMind

warnings.filterwarnings("ignore")


def evaluate_model(model, tokenizer, preprocess, test_samples, args):
    """
    Evaluate a VLM on a set of test samples.

    Args:
        model: PixelMind VLM
        tokenizer: BPE tokenizer
        preprocess: image preprocessor function
        test_samples: list of {"prompt": str, "image": PIL.Image, "gt": str or None}
        args: argparse namespace

    Returns:
        dict of metrics
    """
    device = next(model.parameters()).device
    model.eval()

    total_time = 0
    total_tokens = 0
    successful = 0

    for sample in tqdm(test_samples, desc="Evaluating"):
        try:
            # Preprocess image
            pixel_values = {
                k: v.to(device) for k, v in preprocess(sample["image"]).items()
            }

            # Tokenize prompt
            messages = [
                {
                    "role": "user",
                    "content": sample["prompt"].replace(
                        "<image>",
                        model.config.image_special_token * model.config.image_token_len,
                    ),
                }
            ]
            inputs_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                inputs_text, return_tensors="pt", truncation=True,
                max_length=args.max_seq_len,
            ).to(device)

            # Generate
            st = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generated_ids = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pixel_values=pixel_values,
                )

            gen_time = time.time() - st
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])

            total_time += gen_time
            total_tokens += gen_tokens
            successful += 1

        except Exception as e:
            print(f"Error on sample: {e}")

    metrics = {
        "samples": successful,
        "total_time_s": total_time,
        "tokens_per_second": total_tokens / max(total_time, 0.001),
        "avg_latency_s": total_time / max(successful, 1),
    }
    return metrics


def compare_encoders(
    vlm_weights_path,
    encoders,
    benchmark_data,
    config=None,
    tokenizer=None,
    args=None,
):
    """
    Run the same VLM (same LLM weights + projector) with different visual
    encoders on a shared benchmark suite.

    This is the core of the "Pluggable Encoder" innovation — same model,
    different visual backbones, direct comparison.

    Args:
        vlm_weights_path: path to trained VLM weights (LLM + projector only)
        encoders: dict of {name: path} for encoders to test
        benchmark_data: list of test samples
        config: PixelMindConfig
        tokenizer: tokenizer instance
        args: argparse namespace with generation parameters

    Returns:
        dict: {encoder_name: metrics_dict}
    """
    results = {}

    for enc_name, enc_path in encoders.items():
        print(f"\n{'='*60}")
        print(f"Evaluating encoder: {enc_name} ({enc_path})")
        print(f"{'='*60}")

        # Load config if not provided
        cfg = config or PixelMindConfig()

        # Build encoder
        encoder = build_vision_encoder(enc_name, enc_path)
        preprocess = encoder.preprocess

        # Build model with this encoder
        model = PixelMind(cfg, vision_encoder=encoder)
        state_dict = torch.load(vlm_weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.half().eval().to(args.device if args else "cuda")

        # Evaluate
        metrics = evaluate_model(
            model, tokenizer, preprocess, benchmark_data, args
        )
        results[enc_name] = metrics

    # ── Print comparison table ──
    print(f"\n{'='*60}")
    print("Encoder Comparison Results")
    print(f"{'='*60}")
    print(f"{'Encoder':<16} {'Tokens/s':<12} {'Avg Latency':<12} {'Samples':<10}")
    print("-" * 50)
    for enc_name, metrics in results.items():
        print(
            f"{enc_name:<16} "
            f"{metrics['tokens_per_second']:<12.1f} "
            f"{metrics['avg_latency_s']:<12.3f} "
            f"{metrics['samples']:<10}"
        )

    return results


if __name__ == "__main__":
    print("PixelMind Benchmarks — use import for evaluation pipeline.")
