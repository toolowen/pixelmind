"""
PixelMind LLM Chat/Evaluation
=============================
Command-line interface for evaluating PixelMind LLM models.

Supports:
  - Auto-test mode with preset prompts
  - Interactive chat
  - History management
  - Speed measurement

Usage:
    python -m pixelmind.eval.chat_llm --weight sft --save_dir ./out
"""

import argparse
import random
import time
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from ..config import PixelMindConfig
from ..model.llm import PixelMindForCausalLM
from ..trainer.utils import setup_seed, get_model_params

warnings.filterwarnings("ignore")


def init_model(args):
    """Load model from either native .pth or HuggingFace format."""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if "model" in args.load_from:
        # Native PixelMind weights
        model = PixelMindForCausalLM(
            PixelMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
        model.load_state_dict(
            torch.load(ckp, map_location=args.device), strict=True
        )
    else:
        # HuggingFace Transformers format
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )

    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="PixelMind LLM Chat")
    parser.add_argument(
        "--load_from", default="model", type=str,
        help="model= native .pth, else HF path",
    )
    parser.add_argument("--save_dir", default="out", type=str, help="weights directory")
    parser.add_argument(
        "--weight", default="sft", type=str,
        help="weight prefix (pretrain, sft)",
    )
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument(
        "--inference_rope_scaling", default=False, action="store_true",
        help="enable YaRN rope scaling",
    )
    parser.add_argument("--max_new_tokens", default=8192, type=int)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument("--historys", default=0, type=int, help="history turns")
    parser.add_argument("--show_speed", default=1, type=int, help="show tokens/s")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
    )
    args = parser.parse_args()

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        "解释一下光合作用的基本过程",
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = (
        prompts
        if input_mode == 0
        else iter(lambda: input("💬: "), "")
    )

    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f"💬: {prompt}")

        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        if "pretrain" in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=bool(args.open_thinking),
            )

        inputs = tokenizer(
            inputs, return_tensors="pt", truncation=True
        ).to(args.device)

        print("🧠: ", end="")
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
            repetition_penalty=1.0,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        conversation.append({"role": "assistant", "content": response})

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f"\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n")
        else:
            print("\n\n")


if __name__ == "__main__":
    main()
