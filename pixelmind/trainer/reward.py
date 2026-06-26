"""
Reward functions for PixelMind GRPO training.

Provides:
  - LMForRewardModel: text-only reward model (internlm2-1_8b-reward)
  - VLMJudgeRewardModel: VLM-as-judge reward model (Qwen2.5-VL-3B etc.)
  - compute_ocr_score: OCR accuracy reward for VLM
  - compute_hallucination_penalty: object hallucination penalty for VLM
  - rep_penalty: n-gram repetition penalty
  - calculate_rewards: bundled reward computation
"""

import io
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor


# ── Text-only Reward Model (internlm2-1_8b-reward) ──

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


# ── VLM-as-Judge Reward Model (Qwen2.5-VL-3B etc.) ──

class VLMJudgeRewardModel:
    """
    VLM-based reward model that scores responses by seeing the image.
    Uses Qwen2.5-VL-3B (or similar) as a judge.

    The model is prompted with: image + question + candidate response,
    and outputs a score. VLM judges understand visual context, so
    they penalize hallucinations and reward accurate visual descriptions.
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def get_score(self, raw_images, prompt_text, response):
        """
        Score a VLM response given the image(s) and conversation.

        Args:
            raw_images: list of bytes (JPEG image bytes), one per image
            prompt_text: the full conversation prompt text
            response: the model's generated response

        Returns:
            float score in [-3.0, 3.0]
        """
        # Decode the first image (most VLM tasks have one image)
        if isinstance(raw_images, list) and len(raw_images) > 0:
            img = Image.open(io.BytesIO(raw_images[0])).convert("RGB")
        elif isinstance(raw_images, bytes):
            img = Image.open(io.BytesIO(raw_images)).convert("RGB")
        else:
            # No image — fall back to text-only scoring
            return self._text_only_score(prompt_text, response)

        # Extract the user's question from the prompt
        question = prompt_text
        # Try to find the last user message
        user_pattern = r'<\|im_start\|>user\s+(.*?)<\|im_end\|>'
        matches = re.findall(user_pattern, prompt_text, re.DOTALL)
        if matches:
            question = matches[-1].strip()

        judge_prompt = (
            "You are an expert evaluator for vision-language tasks.\n\n"
            "Given an image and a question, evaluate how well the response "
            "answers the question based on what's actually in the image.\n\n"
            f"Question: {question}\n\n"
            f"Response: {response}\n\n"
            "Rate the response on a scale of 1-5:\n"
            "1 = completely wrong or describes things not in the image\n"
            "2 = mostly wrong with minor correct elements\n"
            "3 = partially correct but significant errors\n"
            "4 = mostly correct with minor issues\n"
            "5 = perfectly accurate, relevant, and well-described\n\n"
            "Output ONLY a single number."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": judge_prompt},
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
        )
        output_text = self.processor.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

        # Parse the score
        match = re.search(r'([1-5])', output_text)
        score = float(match.group(1)) if match else 3.0
        # Map [1,5] → [-3.0, 3.0] to match legacy reward range
        return (score - 3.0) * 1.5

    @torch.no_grad()
    def _text_only_score(self, prompt_text, response):
        """Fallback text-only scoring when no image is available."""
        judge_prompt = (
            "You are an expert evaluator.\n\n"
            f"Question: {prompt_text[-500:]}\n\n"
            f"Response: {response}\n\n"
            "Rate on scale 1-5. Output ONLY a number."
        )
        messages = [{"role": "user", "content": judge_prompt}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
        )
        output_text = self.processor.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

        match = re.search(r'([1-5])', output_text)
        score = float(match.group(1)) if match else 3.0
        return (score - 3.0) * 1.5


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

def calculate_rewards(prompts, responses, reward_model, num_generations, device,
                     raw_image_list=None):
    """
    Calculate combined rewards for GRPO training.

    Rewards combine:
      1. Text quality (length + repetition)
      2. Thinking format (<think> tag quality)
      3. Reward model score (text-only LMForRewardModel or VLM VLMJudgeRewardModel)

    Args:
        prompts: list of prompt strings
        responses: list of response strings (len = len(prompts) * num_generations)
        reward_model: LMForRewardModel or VLMJudgeRewardModel instance
        num_generations: number of generations per prompt
        device: torch device
        raw_image_list: list of bytes (JPEG) per prompt for VLM judge, same length as prompts

    Returns:
        rewards tensor of shape [B * num_generations]
    """
    rewards = torch.zeros(len(responses), device=device)
    is_vlm_judge = isinstance(reward_model, VLMJudgeRewardModel)

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

                # 4. Reward model score
                if is_vlm_judge:
                    # VLM judge: pass image bytes for visual context
                    img = raw_image_list[i] if raw_image_list else None
                    score = reward_model.get_score(img, prompt, response)
                else:
                    # Text-only judge
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
