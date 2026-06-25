"""
PixelMind VLM Pretraining (Projector Alignment)
================================================
Stage 3: Align the vision encoder's output to the LLM's embedding space.

Only the MMVisionProjector is trained; the LLM and vision encoder are frozen.

Usage:
    python -m pixelmind.trainer.vlm_pretrain --from_weight sft --data_path ../data/multimodal/pretrain_i2t.parquet

Recommended: lr=4e-4, epochs=1, batch=4, max_seq=450, freeze_llm=2
"""

import os
import sys
import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..config import PixelMindConfig
from ..data.mm_dataset import VLMDataset
from ..data.collate import vlm_collate_fn
from ..trainer.utils import (
    Logger,
    is_main_process,
    get_lr,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
)
from ..trainer.checkpoint import vlm_checkpoint
from ..trainer.model_init import init_vlm_model

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, args, model, optimizer, scaler,
                autocast_ctx, vlm_config, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(args.device)
        last_step = step

        lr = get_lr(
            epoch * iters + step, args.epochs * iters, args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res["loss"] / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = (
                spend_time / max(step - start_step, 1) * (iters - step) // 60
            )
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, lr: {current_lr:.8f}, "
                f"epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f"{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            # Strip vision encoder (frozen, not needed in ckp)
            clean_state_dict = {
                k: v.half().cpu()
                for k, v in state_dict.items()
                if not k.startswith("vision_encoder.")
            }
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(
                vlm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description="PixelMind VLM Pretrain")
    parser.add_argument("--save_dir", type=str, default="./out", help="output dir")
    parser.add_argument("--save_weight", default="pretrain_vlm", type=str, help="weight prefix")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="peak learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision type")
    parser.add_argument("--num_workers", type=int, default=2, help="data loader workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="save interval")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden dimension")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="num layers")
    parser.add_argument("--max_seq_len", default=450, type=int, help="max sequence length")
    parser.add_argument("--data_path", type=str, default="../data/multimodal/pretrain_i2t.parquet")
    parser.add_argument("--from_weight", default="sft", type=str, help="LLM base weight")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="resume")
    parser.add_argument("--freeze_llm", default=2, type=int, choices=[0, 1, 2],
                        help="0=all, 1=first+last layers, 2=projector only")
    parser.add_argument("--vision_encoder_path", type=str,
                        default="./model/siglip2-base-p32-256-ve")
    parser.add_argument("--vision_encoder_name", type=str, default="siglip2",
                        help="siglip2, dinov2, or intervit")
    parser.add_argument("--use_wandb", action="store_true", help="enable SwanLab")
    parser.add_argument("--wandb_project", type=str, default="PixelMind-VLM-Pretrain")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile")
    args = parser.parse_args()

    # ── 1. Init environment ──
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ── 2. Config + checkpoint ──
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = PixelMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vision_encoder_name=args.vision_encoder_name,
    )
    ckp_data = (
        vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ── 3. Mixed precision ──
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ── 4. Wandb ──
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"PixelMind-VLM-Pretrain-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id,
            resume=resume,
        )

    # ── 5. Model + data + optimizer ──
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        vision_encoder_path=args.vision_encoder_path,
        device=args.device,
        freeze_llm=args.freeze_llm,
    )
    train_ds = VLMDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        max_length=args.max_seq_len,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )

    # ── 6. Resume ──
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ── 7. Compile + DDP ──
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ── 8. Train ──
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        # VLMDataset perm_map already shuffled — use sequential sampling
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or list(range(len(train_ds))), args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=vlm_collate_fn,
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: "
                f"skipping {start_step} steps, resuming from step {start_step + 1}"
            )
            train_epoch(
                epoch, loader, len(loader) + skip, args, model, optimizer,
                scaler, autocast_ctx, vlm_config, start_step, wandb,
            )
        else:
            train_epoch(
                epoch, loader, len(loader), args, model, optimizer,
                scaler, autocast_ctx, vlm_config, 0, wandb,
            )

    # ── 9. Cleanup ──
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
