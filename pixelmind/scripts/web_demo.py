#!/usr/bin/env python3
"""
PixelMind VLM Web Demo
=======================
Gradio-based interactive demo for PixelMind VLM.

Usage:
    python -m pixelmind.scripts.web_demo --model_path ./pixelmind-hf
"""

import argparse
import warnings

import torch
from PIL import Image

from ..config import PixelMindConfig
from ..model.vlm import PixelMind

warnings.filterwarnings("ignore")


def launch_demo(args):
    """Launch Gradio web demo for VLM chat."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        return

    # Load model
    tokenizer_path = args.tokenizer_path
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = PixelMind(
        PixelMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            vision_encoder_name=args.vision_encoder_name,
        ),
        vision_encoder_path=args.vision_encoder_path,
    )

    if args.weight:
        ckp = f"{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    model = model.half().eval().to(args.device)
    preprocess = model.vision_encoder.preprocess

    def chat_fn(image, prompt, history, temperature, top_p, max_new_tokens):
        if image is None:
            return history + [["", "Please upload an image."]]

        # Preprocess
        pixel_values = {
            k: v.to(args.device) for k, v in preprocess(image).items()
        }

        # Build chat
        messages = []
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": prompt})

        inputs_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            inputs_text, return_tensors="pt", truncation=True,
            max_length=args.max_seq_len,
        ).to(args.device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pixel_values=pixel_values,
            )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        history.append([prompt, response])
        return history

    with gr.Blocks(title="PixelMind VLM Demo") as demo:
        gr.Markdown("# 🧠 PixelMind VLM Demo")
        gr.Markdown(
            "A tiny Vision-Language Model (~65M params) trained from scratch. "
            "Upload an image and ask questions!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.85, label="Top-P")
                max_tokens = gr.Slider(64, 512, value=256, step=16, label="Max Tokens")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation")
                prompt_input = gr.Textbox(
                    placeholder="Ask about the image...", label="Your Question"
                )
                submit_btn = gr.Button("Send")

        submit_btn.click(
            chat_fn,
            inputs=[image_input, prompt_input, chatbot, temperature, top_p, max_tokens],
            outputs=[chatbot],
        )

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(description="PixelMind VLM Web Demo")
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument("--weight", default="sft_vlm", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--tokenizer_path", default="./model/tokenizer", type=str)
    parser.add_argument(
        "--vision_encoder_path", default="./model/siglip2-base-p32-256-ve", type=str,
    )
    parser.add_argument("--vision_encoder_name", default="siglip2", type=str)
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
    )
    args = parser.parse_args()
    launch_demo(args)


if __name__ == "__main__":
    main()
