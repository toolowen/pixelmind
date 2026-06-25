"""
Model format conversion for PixelMind.

Converts between:
  - Native .pth format (PixelMind torch weights)
  - HuggingFace Transformers format

Usage:
    # Native → HuggingFace
    python -m pixelmind.scripts.convert --mode to_hf --weight sft

    # HuggingFace → Native
    python -m pixelmind.scripts.convert --mode to_native --hf_path ./pixelmind-hf
"""

import argparse
import os
import warnings

import torch
from transformers import AutoTokenizer

from ..config import PixelMindConfig
from ..model import PixelMindForCausalLM, PixelMind

warnings.filterwarnings("ignore")


def convert_to_hf(args):
    """Convert native .pth weights to HuggingFace format."""
    os.makedirs(args.hf_output, exist_ok=True)

    if args.model_type == "llm":
        config = PixelMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
        )
        model = PixelMindForCausalLM(config)
    else:
        config = PixelMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            vision_encoder_name=args.vision_encoder_name,
        )
        model = PixelMind(config, vision_encoder_path=args.vision_encoder_path)

    # Load weights
    ckp = f"{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
    state_dict = torch.load(ckp, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Save in HF format
    model.half()
    model.save_pretrained(args.hf_output)
    print(f"Saved HF model to {args.hf_output}")

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.save_pretrained(args.hf_output)
    print(f"Saved tokenizer to {args.hf_output}")


def convert_to_native(args):
    """Convert HuggingFace format to native .pth."""
    # Load from HF
    if args.model_type == "llm":
        from ..model.llm import PixelMindForCausalLM
        model = PixelMindForCausalLM.from_pretrained(args.hf_path)
    else:
        model = PixelMind.from_pretrained(args.hf_path, trust_remote_code=True)

    # Save as .pth
    os.makedirs(args.save_dir, exist_ok=True)
    ckp = f"{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, ckp)
    print(f"Saved native weights to {ckp}")


def main():
    parser = argparse.ArgumentParser(description="PixelMind Model Converter")
    parser.add_argument(
        "--mode", type=str, default="to_hf",
        choices=["to_hf", "to_native"],
        help="to_hf: native→HF, to_native: HF→native",
    )
    parser.add_argument(
        "--model_type", default="vlm", type=str,
        choices=["llm", "vlm"],
    )
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument("--weight", default="sft_vlm", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--hf_output", default="./pixelmind-hf", type=str)
    parser.add_argument("--hf_path", default="./pixelmind-hf", type=str)
    parser.add_argument("--tokenizer_path", default="./model/tokenizer", type=str)
    parser.add_argument(
        "--vision_encoder_path", default="./model/siglip2-base-p32-256-ve", type=str,
    )
    parser.add_argument("--vision_encoder_name", default="siglip2", type=str)
    args = parser.parse_args()

    if args.mode == "to_hf":
        convert_to_hf(args)
    else:
        convert_to_native(args)


if __name__ == "__main__":
    main()
