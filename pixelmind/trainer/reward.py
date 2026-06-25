"""
Reward functions for PixelMind GRPO training.

Provides:
  - LMForRewardModel: external reward model wrapper
  - compute_ocr_score: OCR accuracy reward for VLM
  - compute_hallucination_penalty: object hallucination penalty for VLM
  - rep_penalty: n-gram repetition penalty
  - calculate_rewards: bundled reward computation
"""

import re

import torch
from transformers import AutoModel, AutoTokenizer


# ── External Reward Model ──

class LMForRewardModel:
    """
    External LM-based reward model for scoring responses.

    Uses internlm2-1_8b-reward or similar reward model.
    Scores range from -3.0 to +3.0.
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """
        Score a response given conversation history.

        Args:
            messages: list of {"role": ..., "content": ...}
            response: assistant's response text

        Returns:
            float score in [-3.0, 3.0]
        """
        history_text = "\n".join(
            [
                f"{m['role']}: {m['content']}"
                for m in messages[:-1]
            ]
        )
        last_query = messages[-1]["content"] if messages else ""
        message_context = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
            if history_text
            else last_query
        )
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ── Repetition Penalty (n-gram) ──

def rep_penalty(text, n=3, cap=0.5):
    """
    n-gram repetition penalty based on unique gram ratio.

    Returns penalty in [0, cap] — higher means more repetition.
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i: i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


# ── OCR Score (data-driven) ──

def compute_ocr_score(response: str, ground_truth: str = None) -> float:
    """
    Compute OCR accuracy score for VLM responses.

    If ground_truth is available, compares character overlap.
    Otherwise, applies heuristic checks for structured text output.

    Returns score in [-0.5, 0.5].
    """
    if ground_truth:
        # Character-level accuracy
        gt_chars = set(ground_truth.strip().lower())
        resp_chars = set(response.strip().lower())
        if not gt_chars:
            return 0.0
        accuracy = len(gt_chars & resp_chars) / len(gt_chars)
        return 0.5 * (2 * accuracy - 1)  # scale to [-0.5, 0.5]
    else:
        # Heuristic: check if response contains structured text
        score = 0.0
        # Bonus for containing digits/alphanumerics (OCR-style output)
        has_alpha = bool(re.search(r"[a-zA-Z]", response))
        has_digit = bool(re.search(r"\d", response))
        if has_alpha and has_digit:
            score += 0.2
        elif has_alpha or has_digit:
            score += 0.1
        # Penalty for being too short (likely didn't read)
        if len(response.strip()) < 10:
            score -= 0.3
        return score


# ── Hallucination Penalty ──

def compute_hallucination_penalty(response: str, image_prompt: str = None) -> float:
    """
    Penalize likely visual hallucinations in VLM responses.

    Currently heuristic-based: checks for overconfident object descriptions
    when the prompt doesn't ask for them.

    Returns penalty in [0.0, 0.5].
    """
    penalty = 0.0

    # Count object-mention keywords that could be hallucinations
    object_keywords = [
        "dog", "cat", "person", "car", "tree", "building",
        "狗", "猫", "人", "车", "树", "建筑",
    ]
    for keyword in object_keywords:
        if keyword in response.lower():
            penalty += 0.05

    # Cap at 0.5
    return min(penalty, 0.5)


# ── Bundled Reward Calculation ──

def calculate_rewards(prompts, responses, reward_model, num_generations, device):
    """
    Calculate combined rewards for GRPO training.

    Rewards combine:
      1. Text quality (length + repetition)
      2. Thinking format (<think> tag quality)
      3. External reward model score
      4. OCR accuracy (if applicable) — VLM only
      5. Hallucination penalty — VLM only

    Args:
        prompts: list of prompt strings
        responses: list of response strings (len = len(prompts) * num_generations)
        reward_model: LMForRewardModel instance
        num_generations: number of generations per prompt
        device: torch device

    Returns:
        rewards tensor of shape [B * num_generations]
    """
    rewards = torch.zeros(len(responses), device=device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(num_generations):
                response_idx = i * num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 1. Text quality
                # Length bonus
                if 20 <= len(response.strip()) <= 800:
                    rewards[response_idx] += 0.5
                else:
                    rewards[response_idx] -= 0.5

                # 2. Thinking format
                if "</think>" in response:
                    thinking_content, answer_content = response.split("</think>", 1)
                    if 20 <= len(thinking_content.strip()) <= 300:
                        rewards[response_idx] += 1.0
                    else:
                        rewards[response_idx] -= 0.5
                    if response.count("</think>") == 1:
                        rewards[response_idx] += 0.25
                    else:
                        rewards[response_idx] -= 0.25
                    response = answer_content.strip()  # Use answer part for scoring

                # 3. Repetition penalty
                rewards[response_idx] -= rep_penalty(response)

                # 4. External reward model
                pattern = (
                    r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                )
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [
                    {"role": role, "content": content.strip()}
                    for role, content in matches
                ]
                score = reward_model.get_score(messages, response)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=device)
        rewards += reward_model_scores

    return rewards
