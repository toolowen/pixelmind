"""
PixelMind VLM Chat/Evaluation
=============================
Command-line evaluation for PixelMind VLM models.
Processes all images in a directory and generates descriptions.

Usage:
    python -m pixelmind.eval.chat_vlm --weight sft_vlm --image_dir ./dataset/eval_images/
"""

import argparse
import os
import random
import time
import warnings

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from ..config import PixelMindConfig
from ..model.vlm import PixelMind
from ..trainer.utils import setup_seed, get_model_params

warnings.filterwarnings("ignore")


def init_model(args):
    """Load VLM from native .pth or HuggingFace format."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.load_from, trust_remote_code=True
    )

    if "model" in args.load_from:
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
        model = PixelMind(
            PixelMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                vision_encoder_name=args.vision_encoder_name,
            ),
            vision_encoder_path=args.vision_encoder_path,
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if "mask" not in k},
            strict=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )
        model.vision_encoder, model.processor = PixelMind.get_vision_model(
            args.vision_encoder_path
        )

    get_model_params(model, model.config, ignore_patterns={"vision_encoder"})
    return model.half().eval().to(args.device), tokenizer, model.vision_encoder.preprocess


def main():
    parser = argparse.ArgumentParser(description="PixelMind VLM Chat")
    parser.add_argument(
        "--load_from", default="model", type=str,
        help="model= native .pth, else HF path",
    )
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument(
        "--weight", default="sft_vlm", type=str,
        help="weight prefix (pretrain_vlm, sft_vlm)",
    )
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument(
        "--vision_encoder_path", type=str,
        default="./model/siglip2-base-p32-256-ve",
    )
    parser.add_argument(
        "--vision_encoder_name", type=str, default="siglip2",
    )
    parser.add_argument(
        "--image_dir", default="./dataset/eval_images/", type=str,
    )
    parser.add_argument("--show_speed", default=1, type=int)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
    )
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt = "<image>\n请描述这张图中的主要物体和场景。"

    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert("RGB")

            # Preprocess image
            pixel_values = {
                k: v.to(args.device)
                for k, v in preprocess(image).items()
            }

            # Build prompt with image pads
            messages = [
                {
                    "role": "user",
                    "content": prompt.replace(
                        "<image>",
                        model.config.image_special_token * model.config.image_token_len,
                    ),
                }
            ]
            inputs_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=bool(args.open_thinking),
            )
            inputs = tokenizer(
                inputs_text, return_tensors="pt", truncation=True
            ).to(args.device)

            print(f"[Image]: {image_file}")
            print(f"💬: {repr(prompt)}")
            print("🤖: ", end="")
            st = time.time()

            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                pixel_values=pixel_values,
            )

            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            if args.show_speed:
                print(
                    f"\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n"
                )
            else:
                print("\n\n")


if __name__ == "__main__":
    main()
