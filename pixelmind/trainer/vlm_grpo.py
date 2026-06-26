"""
PixelMind VLM GRPO — Group Relative Policy Optimization for VLM ★ CORE INNOVATION
====================================================================================
Stage 5: Align the VLM with reinforcement learning using visual inputs.

This is a novel combination:
  - GRPO (single-network RL, no critic needed, memory-efficient)
  - Vision-Language Model (images processed during rollout AND optimization)

Based on text-only GRPO, extended to handle pixel_values throughout
the entire optimization pipeline: rollout → log-prob → reference → loss.

Usage:
    python -m pixelmind.trainer.vlm_grpo \
        --from_weight sft_vlm --data_path ../data/multimodal/grpo.parquet \
        --num_generations 4 --batch_size 2

Key design decisions:
  - Vision encoder FROZEN and SHARED across policy + reference models
  - pixel_values carried through RolloutResult for reward computation
  - Policy and reference forward passes both receive pixel_values
"""

import argparse
import gc
import math
import os
import re
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import PixelMindConfig
from ..data.mm_dataset import VLMRLDataset
from ..data.collate import vlm_rl_collate_fn
from ..trainer.utils import (
    Logger,
    is_main_process,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
)
from ..trainer.checkpoint import vlm_checkpoint
from ..trainer.model_init import init_vlm_model
from ..trainer.reward import rep_penalty, calculate_rewards, LMForRewardModel, VLMJudgeRewardModel
from ..trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings("ignore")


# ── Combined reward function for VLM GRPO ──

def calculate_vlm_rewards(
    prompts, responses, reward_model, num_generations, device,
    raw_image_list=None,
):
    """
    Compute rewards for VLM GRPO.

    Passes raw image bytes to the reward model (VLMJudgeRewardModel)
    when available, so the judge can see the actual image.
    """
    return calculate_rewards(
        prompts, responses, reward_model, num_generations, device,
        raw_image_list=raw_image_list,
    )


# ── GRPO training epoch for VLM ──

def vlm_grpo_train_epoch(
    epoch, loader, iters, args, model, rollout_engine,
    ref_model, reward_model, tokenizer, autocast_ctx,
    vlm_config, optimizer, scheduler, start_step=0, wandb=None,
):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompts"]       # list[str], length B
        pixel_values = batch["pixel_values"]  # image tensors
        raw_images_list = batch.get("raw_images_list", None)  # raw bytes for VLM judge

        # ── Move pixel_values to device ──
        pv_device = {}
        if hasattr(pixel_values, "keys"):
            pv_device = {
                k: v.to(args.device) for k, v in pixel_values.items()
            }
        else:
            pv_device = pixel_values.to(args.device)

        # ── Tokenize prompts ──
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            padding_side="left",
            add_special_tokens=False,
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -args.max_seq_len:
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -args.max_seq_len:
            ]

        # ── Rollout WITH pixel_values ★ ──
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
            pixel_values=pv_device,  # ★ VLM: pass images to rollout
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = (
            prompt_lens.unsqueeze(1)
            - 1
            + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        )

        # ── Repeat pixel_values for num_generations (for forward) ──
        if hasattr(pv_device, "keys"):
            pv_repeated = {
                k: v.repeat_interleave(args.num_generations, dim=0)
                for k, v in pv_device.items()
            }
        else:
            pv_repeated = pv_device.repeat_interleave(args.num_generations, dim=0)

        # ── Rewards ──
        rewards = calculate_vlm_rewards(
            prompts, completions, reward_model,
            args.num_generations, args.device,
            raw_image_list=raw_images_list,
        )

        # ── Policy forward WITH vision ★ ──
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(
                outputs, attention_mask=full_mask, pixel_values=pv_repeated,
            )
            per_token_logps = (
                F.log_softmax(res["logits"][:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        # ── Reference forward WITH vision ★ ──
        with torch.no_grad():
            ref_res = ref_model(
                outputs, attention_mask=full_mask, pixel_values=pv_repeated,
            )
            ref_per_token_logps = (
                F.log_softmax(ref_res["logits"][:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        # ── Debug sampling ──
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger("-" * 100)
                Logger(f"{'=' * 30} [DEBUG] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger("=" * 100)

        # ── GRPO advantage computation ──
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(
            args.num_generations
        )
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ── Completion mask (up to EOS) ──
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1) - 1,
            dtype=torch.long, device=args.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            (
                torch.arange(is_eos.size(1), device=args.device)
                .expand(is_eos.size(0), -1)
                <= eos_idx.unsqueeze(1)
            )
            & completion_pad_mask
        ).int()

        # ── KL divergence + clipped loss ──
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(
                clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                - args.beta * per_token_kl
            )
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(
                torch.min(per_token_loss1, per_token_loss2)
                - args.beta * per_token_kl
            )

        policy_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()
        loss = policy_loss / args.accumulation_steps
        loss.backward()

        # ── Optimizer step ──
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ──
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = (
                ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item()
                / max(completion_mask.sum().item(), 1)
            )
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                f"Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, "
                f"Actor Loss: {policy_loss_val:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, "
                f"Learning Rate: {current_lr:.8f}"
            )
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr,
                })

        # ── Save + update policy ──
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f"{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
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
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scheduler=scheduler,
            )
            model.train()
            del state_dict, clean_state_dict

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        # ── Memory cleanup ──
        del (
            prompt_inputs, outputs, completion_ids, per_token_logps,
            ref_per_token_logps, completions, rewards,
            grouped_rewards, mean_r, std_r, advantages,
            completion_mask, completion_pad_mask, prompt_lens, logp_pos,
            pv_device, pv_repeated,
        )
        gc.collect()

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="PixelMind VLM GRPO")
    parser.add_argument("--save_dir", type=str, default="./out", help="output dir")
    parser.add_argument("--save_weight", default="grpo_vlm", type=str, help="weight prefix")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="peak learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision type")
    parser.add_argument("--num_workers", type=int, default=2, help="data loader workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip threshold")
    parser.add_argument("--log_interval", type=int, default=1, help="logging interval")
    parser.add_argument("--save_interval", type=int, default=10, help="save interval")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden dimension")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="num layers")
    parser.add_argument("--max_seq_len", default=768, type=int, help="max prompt length")
    parser.add_argument("--max_gen_len", type=int, default=512, help="max generation length")
    parser.add_argument("--data_path", type=str, default="../data/multimodal/grpo.parquet")
    parser.add_argument("--num_generations", type=int, default=4, help="generations per prompt")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon upper bound (CISPO)")
    parser.add_argument("--from_weight", default="sft_vlm", type=str, help="VLM base weight")
    parser.add_argument("--reward_model_path", type=str,
                        default="../../internlm2-1_8b-reward", help="reward model path")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="resume")
    parser.add_argument("--vision_encoder_path", type=str,
                        default="./model/siglip2-base-p32-256-ve")
    parser.add_argument("--vision_encoder_name", type=str, default="siglip2")
    parser.add_argument("--rollout_engine", type=str, default="torch",
                        choices=["torch", "sglang"], help="rollout engine type")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="./model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo")
    parser.add_argument("--use_wandb", action="store_true", help="enable SwanLab")
    parser.add_argument("--wandb_project", type=str, default="PixelMind-VLM-GRPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="print debug samples")
    parser.add_argument("--debug_interval", type=int, default=20)
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
        max_seq_len=args.max_seq_len + args.max_gen_len,
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
            name=f"PixelMind-VLM-GRPO-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id,
            resume=resume,
        )

    # ── 5. Init models + data + optimizer ──
    # Policy model
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        vision_encoder_path=args.vision_encoder_path,
        device=args.device,
        freeze_llm=0,  # ★ GRPO trains all LLM layers (projector already trainable)
    )

    # Reference model: share vision encoder, clone LLM+projector weights
    ref_model, _, _ = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        vision_encoder_path=args.vision_encoder_path,
        device=args.device,
        freeze_llm=0,
    )
    ref_model = ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # ── Init reward model: auto-detect VLM judge ──
    def _is_vlm_judge(path):
        return any(kw in path.lower() for kw in ("qwen2.5-vl", "qwen25-vl", "qwen2.5vl", "qwen-vl"))
    if _is_vlm_judge(args.reward_model_path):
        reward_model = VLMJudgeRewardModel(
            args.reward_model_path, device=args.device, dtype=torch.float16
        )
        Logger(f"Using VLM judge reward model: {args.reward_model_path}")
    else:
        reward_model = LMForRewardModel(
            args.reward_model_path, device=args.device, dtype=torch.float16
        )

    # Rollout engine
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # Data + optimizer
    train_ds = VLMRLDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        max_length=vlm_config.max_seq_len,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=vlm_rl_collate_fn,
    )
    iters = len(loader_for_count)
    total_optimizer_steps = (
        math.ceil(iters / args.accumulation_steps) * args.epochs
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps,
        eta_min=args.learning_rate / 10,
    )

    # ── 6. Resume ──
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scheduler" in ckp_data:
            scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ── 7. Compile + DDP ──
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ── 8. Train ──
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=vlm_rl_collate_fn,
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: "
                f"skipping {start_step} steps, resuming from step {start_step + 1}"
            )
            vlm_grpo_train_epoch(
                epoch, loader, len(loader) + skip, args, model,
                rollout_engine, ref_model, reward_model, tokenizer,
                autocast_ctx, vlm_config, optimizer, scheduler,
                start_step, wandb,
            )
        else:
            vlm_grpo_train_epoch(
                epoch, loader, len(loader), args, model,
                rollout_engine, ref_model, reward_model, tokenizer,
                autocast_ctx, vlm_config, optimizer, scheduler,
                0, wandb,
            )

    # ── 9. Cleanup ──
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
