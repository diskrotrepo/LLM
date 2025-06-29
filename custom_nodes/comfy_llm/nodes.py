# Node implementations for ComfyUI LLM prototype
from __future__ import annotations

from typing import List, Tuple, Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModelLoader:
    """Load a HuggingFace causal language model and tokenizer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": os.environ.get("HF_MODEL", "sshleifer/tiny-gpt2")}),
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = ("MODEL", "TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    CATEGORY = "LLM/Model"

    def execute(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return model, tokenizer


def _tokens_to_markdown(tokens: List[int], tokenizer) -> str:
    if not tokens:
        return ""
    text = tokenizer.decode(tokens)
    return text


class ChatInput:
    """Receives a user message and existing conversation tokens."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {}),
                "conversation": ("LIST", {"default": []}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("user_msg", "conversation")
    CATEGORY = "LLM/IO"

    def execute(self, message: str, conversation: List[int]) -> Tuple[str, List[int]]:
        return message, list(conversation)


class ChatHistory:
    """Returns recent conversation as markdown."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokens": ("LIST", {"default": []}),
                "n_turns": ("INT", {"default": 5}),
                "tokenizer": ("TOKENIZER", {}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("history_md",)
    OUTPUT_NODE = True
    CATEGORY = "LLM/Display"

    def execute(self, tokens: List[int], n_turns: int, tokenizer) -> Tuple[str]:
        md = _tokens_to_markdown(tokens[-n_turns:], tokenizer)
        return (md,)


class PromptBuilder:
    """Concatenate system prompt, history, and user message."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_msg": ("STRING", {}),
                "history_md": ("STRING", {}),
                "user_msg": ("STRING", {}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    CATEGORY = "LLM/Pre-proc"

    def execute(self, system_msg: str, history_md: str, user_msg: str) -> Tuple[str]:
        prompt = f"{system_msg}\n{history_md}\n{user_msg}".strip()
        return (prompt,)


class HFInference:
    """Run a HF causal LM and return probabilities for next token."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "model": ("MODEL", {}),
                "tokenizer": ("TOKENIZER", {}),
                "max_new_tokens": ("INT", {"default": 1}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("prob_batch",)
    CATEGORY = "LLM/Model"

    def execute(self, prompt: str, model, tokenizer, max_new_tokens: int):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return (probs,)


class AverageProbs:
    """Mean of tensors in list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probs": ("FLOAT", {}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT",)
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_probs": ("FLOAT", {}),
                "negative_probs": ("FLOAT", {}),
                "epsilon": ("FLOAT", {"default": 1e-7}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ratio_probs",)
    CATEGORY = "LLM/Math"

    def execute(self, positive_probs: torch.Tensor, negative_probs: torch.Tensor, epsilon: float):
        ratio = positive_probs / (negative_probs + epsilon)
        ratio = torch.clamp(ratio, min=0)
        ratio = ratio / ratio.sum(dim=-1, keepdim=True)
        return (ratio,)


class TemperatureScaler:
    """Adjust logits or probs by temperature."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "logits_or_probs": ("FLOAT", {}),
                "temperature": ("FLOAT", {"default": 1.0}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("scaled",)
    CATEGORY = "LLM/Math"

    def execute(self, logits_or_probs: torch.Tensor, temperature: float):
        scaled = logits_or_probs / max(temperature, 1e-5)
        scaled = torch.softmax(scaled, dim=-1)
        return (scaled,)


class TokenSampler:
    """Sample token id from probability batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prob_batch": ("FLOAT", {}),
                "top_k": ("INT", {"default": 50}),
                "top_p": ("FLOAT", {"default": 0.95}),
                "seed": ("INT", {"default": None}),
            }
        }
    FUNCTION = "execute"
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conversation": ("LIST", {"default": []}),
                "token_id": ("INT", {}),
                "token_prob": ("FLOAT", {}),
            }
        }
    FUNCTION = "execute"
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("conversation",)
    CATEGORY = "LLM/IO"

    def execute(self, conversation: List[int], token_id: int, token_prob: float):
        conversation.append(token_id)
        return (conversation,)


class TensorViewer:
    """Return string slice of tensor for debug."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("FLOAT", {}),
                "slice_spec": ("STRING", {"default": ":"}),
            }
        }
    FUNCTION = "execute"
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
        HFModelLoader,
        HFInference,
        AverageProbs,
        RatioProbs,
        TemperatureScaler,
        TokenSampler,
        ChatUpdate,
        TensorViewer,
    ]
