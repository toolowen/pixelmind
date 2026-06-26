"""
PixelMind Pretrain Evaluation
=============================
Comprehensive evaluation for the pretrained LLM checkpoint.

Metrics:
  1. Perplexity on validation text (next-token prediction quality)
  2. Sample text completion (qualitative generation check)
  3. Token generation speed (tokens/s)
  4. Top-k / Top-p accuracy on known patterns

Usage:
    # Quick test
    python -m pixelmind.eval.eval_pretrain --weight pretrain --mode quick

    # Full evaluation with perplexity
    python -m pixelmind.eval.eval_pretrain --weight pretrain --mode full --data_path data/text/pretrain_t2t_mini.jsonl

    # Only generation samples
    python -m pixelmind.eval.eval_pretrain --weight pretrain --mode generate
"""

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import PixelMindConfig
from ..model.llm import PixelMindForCausalLM
from ..trainer.utils import setup_seed, get_model_params

warnings.filterwarnings("ignore")


# ── Model loading ──────────────────────────────────────────────────────

def load_pretrained_model(args):
    """Load the pretrained LLM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = PixelMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        inference_rope_scaling=args.inference_rope_scaling,
    )
    model = PixelMindForCausalLM(config)

    ckp = f"{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
    if not Path(ckp).exists():
        raise FileNotFoundError(
            f"Weight file not found: {ckp}\n"
            f"Run LLM pretrain first: python -m pixelmind.trainer.llm_pretrain"
        )

    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    get_model_params(model, config)
    model = model.half().eval().to(args.device)

    return model, tokenizer, config


# ── 1. Perplexity evaluation ──────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, data_path, args):
    """
    Compute perplexity on validation text.

    Perplexity = exp(average cross-entropy loss).
    Lower is better — the model is less "surprised" by the text.
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print("1. Perplexity Evaluation")
    print(f"{'='*60}")

    ds = load_dataset("json", data_files=data_path, split="train")
    # Use last 10% as validation if large enough
    if len(ds) > 100:
        ds = ds.select(range(max(1, int(len(ds) * 0.9)), len(ds)))
    if len(ds) > args.max_eval_samples:
        ds = ds.select(range(args.max_eval_samples))

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for sample in tqdm(ds, desc="Computing perplexity"):
        text = str(sample.get("text", sample.get("content", "")))
        if not text.strip():
            continue

        # Tokenize with truncation
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            max_length=args.max_seq_len - 2,
            truncation=True,
        ).input_ids
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

        # Pad to max_seq_len
        pad_len = args.max_seq_len - len(tokens)
        tokens = tokens + [tokenizer.pad_token_id] * pad_len
        input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                # Count non-masked tokens for weighted average
                n_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        except Exception as e:
            print(f"  Skipping sample (error: {e})")
            continue

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    print(f"\n  Validation samples : {len(ds)}")
    print(f"  Total tokens       : {total_tokens}")
    print(f"  Average loss       : {avg_loss:.4f}")
    print(f"  Perplexity         : {perplexity:.2f}")
    print(f"\n  Interpretation:")
    if perplexity < 10:
        print(f"    ✓ Excellent — model predicts very confidently")
    elif perplexity < 50:
        print(f"    ○ Good — model captures most patterns")
    elif perplexity < 200:
        print(f"    △ Decent for a 64M model from scratch")
    else:
        print(f"    ✗ High — model struggles with this text distribution")

    return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}


# ── 2. Text completion samples ─────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, tokenizer, args):
    """
    Generate completions for a diverse set of prompts.
    Covers: factual, creative, code, reasoning, Chinese text.
    """
    print(f"\n{'='*60}")
    print("2. Generation Samples")
    print(f"{'='*60}")

    prompts = [
        # Chinese prompts
        "人工智能的未来发展趋势是",
        "光合作用的基本原理是：",
        "今天天气真好，适合",
        "学习编程的第一步是",
        # English prompts
        "The capital of France is",
        "Machine learning is a subset of",
        "To solve this problem, we need to",
        # Code / structure
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "import numpy as np\n\n# Create a 3x3 identity matrix\n",
        # Knowledge recall
        "水的化学分子式是",
        "地球绕太阳公转一周的时间是",
        "世界上最深的海洋是",
        # Reasoning
        "如果明天下雨，我应该带上",
        "一只猫和一只狗的区别在于",
    ]

    model.eval()
    for i, prompt in enumerate(prompts):
        setup_seed(42 + i)

        # Pretrain model: raw text continuation (no chat template)
        inputs_text = tokenizer.bos_token + prompt
        inputs = tokenizer(
            inputs_text, return_tensors="pt", truncation=True
        ).to(args.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.gen_max_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
            )

        completion = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )

        print(f"\n  [{i+1}] Prompt: {prompt.strip()}")
        print(f"      Completion: {completion.strip()[:200]}")
        if len(completion) > 200:
            print(f"      ... (truncated, total {len(completion)} chars)")


# ── 3. Speed benchmark ─────────────────────────────────────────────────

@torch.no_grad()
def benchmark_speed(model, tokenizer, args):
    """
    Measure token generation speed (tokens/second).
    Runs multiple batches and reports average.
    """
    print(f"\n{'='*60}")
    print("3. Speed Benchmark")
    print(f"{'='*60}")

    test_prompts = [
        "请介绍一下中国的四大发明。",
        "Write a Python function to sort a list of numbers.",
        "人工智能是计算机科学的一个分支，",
        "The history of deep learning dates back to",
    ]

    total_time = 0.0
    total_tokens = 0

    for i, prompt in enumerate(test_prompts):
        inputs_text = tokenizer.bos_token + prompt
        inputs = tokenizer(
            inputs_text, return_tensors="pt", truncation=True
        ).to(args.device)

        torch.cuda.synchronize()
        st = time.time()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - st
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        speed = gen_tokens / elapsed

        total_time += elapsed
        total_tokens += gen_tokens
        print(f"  [{i+1}] {gen_tokens:3d} tokens in {elapsed:.2f}s → {speed:7.1f} tok/s")

    avg_speed = total_tokens / max(total_time, 0.001)
    print(f"\n  Average: {avg_speed:.1f} tokens/s")
    return {"tokens_per_second": avg_speed, "total_tokens": total_tokens}


# ── 4. Loss on known patterns ──────────────────────────────────────────

@torch.no_grad()
def test_known_patterns(model, tokenizer, args):
    """
    Test the model's loss on common patterns:
    - Numbers sequence
    - Repeated characters
    - Common phrases
    """
    print(f"\n{'='*60}")
    print("4. Pattern Recognition Tests")
    print(f"{'='*60}")

    patterns = {
        "Number sequence": "1 2 3 4 5 6 7 8 9 10",
        "Alphabet": "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "Common Chinese": "你好，今天天气很好，我们去公园散步吧。",
        "Python code": "def add(a, b):\n    return a + b",
        "Math": "1 + 1 = 2, 2 + 2 = 4, 3 + 3 = 6",
    }

    model.eval()
    for name, text in patterns.items():
        tokens = tokenizer(
            tokenizer.bos_token + text + tokenizer.eos_token,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)

        labels = tokens["input_ids"].clone()
        labels[:, :1] = -100  # don't predict BOS

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(tokens["input_ids"], labels=labels)
            loss = outputs["loss"].item()

        ppl = math.exp(loss) if loss < 100 else float("inf")
        verdict = "✓ confident" if loss < 2.0 else ("△ OK" if loss < 5.0 else "✗ struggling")
        print(f"  {name:<20s}: loss={loss:.4f}  ppl={ppl:8.1f}  [{verdict}]")

    return {"patterns_tested": len(patterns)}


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PixelMind Pretrain Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="quick", type=str,
        choices=["quick", "full", "generate", "perplexity", "speed"],
        help="quick=generate+speed | full=all tests | generate/speed/perplexity=only that",
    )
    parser.add_argument(
        "--weight", default="pretrain", type=str,
        help="weight prefix (e.g., pretrain, sft)",
    )
    parser.add_argument("--save_dir", default="./out", type=str)
    parser.add_argument("--tokenizer_path", default="./model/tokenizer", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument(
        "--inference_rope_scaling", default=False, action="store_true",
        help="enable YaRN rope scaling for long context",
    )
    parser.add_argument("--max_seq_len", default=340, type=int)
    parser.add_argument("--max_eval_samples", default=200, type=int)
    parser.add_argument("--data_path", default="data/text/pretrain_t2t_mini.jsonl", type=str)
    parser.add_argument("--gen_max_tokens", default=80, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
    )
    parser.add_argument("--output", default=None, type=str, help="save results as JSON")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"PixelMind Pretrain Evaluation")
    print(f"{'='*60}")
    print(f"  Weight : {args.weight}")
    print(f"  Config : hidden={args.hidden_size}, layers={args.num_hidden_layers}")
    print(f"  Device : {args.device}")
    print(f"  Mode   : {args.mode}")

    # ── Load model ──
    print(f"\nLoading model from {args.save_dir}/{args.weight}_{args.hidden_size}.pth ...")
    try:
        model, tokenizer, config = load_pretrained_model(args)
        print(f"  Model loaded — {config.hidden_size}d, {config.num_hidden_layers} layers")
        print(f"  Vocab size: {config.vocab_size}")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"\n  Available modes that don't require weights:")
        print(f"    None — the model must be trained first.")
        print(f"\n  To train: python -m pixelmind.trainer.llm_pretrain --data_path data/text/pretrain_t2t_mini.jsonl")
        return

    results = {"config": f"{args.hidden_size}d_{args.num_hidden_layers}L", "mode": args.mode}

    # ── Run evaluations ──
    if args.mode in ("quick", "generate", "full"):
        generate_samples(model, tokenizer, args)
        results["generation"] = "done"

    if args.mode in ("quick", "speed", "full"):
        speed = benchmark_speed(model, tokenizer, args)
        results["speed"] = speed

    if args.mode in ("full", "perplexity"):
        # Check if data exists
        if not Path(args.data_path).exists():
            print(f"\n  Data file not found: {args.data_path}")
            print(f"  Skipping perplexity evaluation.")
            print(f"  Download data first: see AUTODL_GUIDE.md Section 4.1")
        else:
            ppl = compute_perplexity(model, tokenizer, args.data_path, args)
            results["perplexity"] = ppl

    if args.mode == "full":
        test_known_patterns(model, tokenizer, args)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")

    if results.get("perplexity"):
        print(f"  Perplexity : {results['perplexity']['perplexity']:.2f}")
    if results.get("speed"):
        print(f"  Speed      : {results['speed']['tokens_per_second']:.1f} tok/s")

    # ── Save results ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
