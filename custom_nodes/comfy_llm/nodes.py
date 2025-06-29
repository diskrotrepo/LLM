# Node implementations for ComfyUI LLM prototype
from __future__ import annotations

from typing import List, Tuple, Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simple in-memory conversation store
_CONVERSATIONS = {}


def _get_conv(conv_id: str) -> List[int]:
    return _CONVERSATIONS.setdefault(conv_id, [])


def _append_token(conv_id: str, token_id: int):
    _get_conv(conv_id).append(token_id)


def _tokens_to_markdown(tokens: List[int], tokenizer) -> str:
    if not tokens:
        return ""
    text = tokenizer.decode(tokens)
    return text


class ChatInput:
    """Receives a user message and optional conversation id."""

    INPUT_TYPES = {
        "required": {
            "message": ("STRING", {"forceInput": True}),
            "conversation_id": ("STRING", {"default": ""}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_msg", "conv_id")
    CATEGORY = "LLM/IO"

    def execute(self, message: str, conversation_id: str) -> Tuple[str, str]:
        conv_id = conversation_id or os.urandom(4).hex()
        return message, conv_id


class ChatHistory:
    """Returns recent conversation as markdown."""

    INPUT_TYPES = {
        "required": {
            "conv_id": ("STRING", {"forceInput": True}),
            "n_turns": ("INT", {"default": 5}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("history_md",)
    OUTPUT_NODE = True
    CATEGORY = "LLM/Display"

    def execute(self, conv_id: str, n_turns: int) -> Tuple[str]:
        tokens = _get_conv(conv_id)[-n_turns:]
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        md = _tokens_to_markdown(tokens, tokenizer)
        return (md,)


class PromptBuilder:
    """Concatenate system prompt, history, and user message."""

    INPUT_TYPES = {
        "required": {
            "system_msg": ("STRING", {}),
            "history_md": ("STRING", {}),
            "user_msg": ("STRING", {}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "LLM/Pre-proc"

    def execute(self, system_msg: str, history_md: str, user_msg: str) -> Tuple[str]:
        prompt = f"{system_msg}\n{history_md}\n{user_msg}".strip()
        return (prompt,)


class HFInference:
    """Run a HF causal LM and return probabilities for next token."""

    INPUT_TYPES = {
        "required": {
            "prompt": ("STRING", {"forceInput": True}),
            "model_name": ("STRING", {"default": os.environ.get("HF_MODEL", "sshleifer/tiny-gpt2")}),
            "max_new_tokens": ("INT", {"default": 1}),
        }
    }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("prob_batch",)
    CATEGORY = "LLM/Model"

    def execute(self, prompt: str, model_name: str, max_new_tokens: int):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return (probs,)


class AverageProbs:
    """Mean of tensors in list."""

    INPUT_TYPES = {
        "required": {
            "probs": ("TENSOR", {"forceInput": True, "is_list": True}),
        }
    }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("avg_probs",)
    CATEGORY = "LLM/Math"
    INPUT_IS_LIST = True

    def execute(self, probs: List[torch.Tensor]):
        stack = torch.stack(probs)
        avg = stack.mean(dim=0)
        avg = torch.clamp(avg, min=1e-7)
        avg = avg / avg.sum(dim=-1, keepdim=True)
        return (avg,)


class RatioProbs:
    """Element-wise division of tensors with renorm."""

    INPUT_TYPES = {
        "required": {
            "positive_probs": ("TENSOR", {"forceInput": True}),
            "negative_probs": ("TENSOR", {"forceInput": True}),
            "epsilon": ("FLOAT", {"default": 1e-7}),
        }
    }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("ratio_probs",)
    CATEGORY = "LLM/Math"

    def execute(self, positive_probs: torch.Tensor, negative_probs: torch.Tensor, epsilon: float):
        ratio = positive_probs / (negative_probs + epsilon)
        ratio = torch.clamp(ratio, min=0)
        ratio = ratio / ratio.sum(dim=-1, keepdim=True)
        return (ratio,)


class TemperatureScaler:
    """Adjust logits or probs by temperature."""

    INPUT_TYPES = {
        "required": {
            "logits_or_probs": ("TENSOR", {"forceInput": True}),
            "temperature": ("FLOAT", {"default": 1.0}),
        }
    }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("scaled",)
    CATEGORY = "LLM/Math"

    def execute(self, logits_or_probs: torch.Tensor, temperature: float):
        scaled = logits_or_probs / max(temperature, 1e-5)
        scaled = torch.softmax(scaled, dim=-1)
        return (scaled,)


class TokenSampler:
    """Sample token id from probability batch."""

    INPUT_TYPES = {
        "required": {
            "prob_batch": ("TENSOR", {"forceInput": True}),
            "top_k": ("INT", {"default": 50}),
            "top_p": ("FLOAT", {"default": 0.95}),
            "seed": ("INT", {"default": None}),
        }
    }
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("token_id", "token_prob")
    CATEGORY = "LLM/Sampler"

    def execute(self, prob_batch: torch.Tensor, top_k: int, top_p: float, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
        probs = prob_batch.squeeze(0)
        topk_probs, topk_indices = torch.topk(probs, k=top_k)
        cumulative = torch.cumsum(topk_probs, dim=0)
        mask = cumulative <= top_p
        if mask.sum() == 0:
            mask[0] = True
        filtered_probs = topk_probs[mask]
        filtered_indices = topk_indices[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()
        dist = torch.distributions.Categorical(filtered_probs)
        idx = dist.sample()
        token_id = filtered_indices[idx].item()
        token_prob = filtered_probs[idx].item()
        return token_id, token_prob


class ChatUpdate:
    """Append sampled token to conversation."""

    INPUT_TYPES = {
        "required": {
            "conv_id": ("STRING", {"forceInput": True}),
            "token_id": ("INT", {}),
            "token_prob": ("FLOAT", {}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("conv_id",)
    CATEGORY = "LLM/IO"

    def execute(self, conv_id: str, token_id: int, token_prob: float):
        _append_token(conv_id, token_id)
        return (conv_id,)


class TensorViewer:
    """Return string slice of tensor for debug."""

    INPUT_TYPES = {
        "required": {
            "tensor": ("TENSOR", {"forceInput": True}),
            "slice_spec": ("STRING", {"default": ":"}),
        }
    }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_dump",)
    CATEGORY = "Debug"

    def execute(self, tensor: torch.Tensor, slice_spec: str) -> Tuple[str]:
        try:
            sliced = eval(f"tensor[{slice_spec}]")
        except Exception:
            sliced = tensor
        text = repr(sliced)
        return (text,)


def get_classes():
    return [
        ChatInput,
        ChatHistory,
        PromptBuilder,
        HFInference,
        AverageProbs,
        RatioProbs,
        TemperatureScaler,
        TokenSampler,
        ChatUpdate,
        TensorViewer,
    ]
