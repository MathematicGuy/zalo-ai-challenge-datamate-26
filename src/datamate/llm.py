"""Chat model interface. The agent depends on ChatModel only, so tests can pass a fake one."""
from __future__ import annotations

import os
from typing import Protocol

MODEL = os.environ.get("DATAMATE_MODEL", "Qwen/Qwen3.5-9B")


class ChatModel(Protocol):
    def chat(self, conversations: list[list[dict]], n: int = 1, temperature: float = 0.0,
             thinking: bool = False) -> list[list[str]]:
        """One list of n replies per conversation."""


class VLLMChatModel:
    """Offline vLLM engine. Every knob is an environment variable so the Docker image and the dev box
    (one shared 24 GB card) run the same code."""

    def __init__(self, model: str = MODEL, seed: int = 0):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model,
            quantization=os.environ.get("DATAMATE_QUANT", "fp8") or None,
            max_model_len=int(os.environ.get("DATAMATE_MAX_LEN", 32768)),
            gpu_memory_utilization=float(os.environ.get("DATAMATE_GPU_UTIL", 0.6)),
            tensor_parallel_size=int(os.environ.get("DATAMATE_TP", 1)),
            enable_prefix_caching=True,
            # gpu_memory_utilization does not cap the profiling peak (~248k-vocab logits); fewer seqs keeps it down.
            max_num_seqs=int(os.environ.get("DATAMATE_MAX_SEQS", 16)),
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=seed,
        )
        self._sampling = SamplingParams
        self.seed = seed

    def chat(self, conversations: list[list[dict]], n: int = 1, temperature: float = 0.0,
             thinking: bool = False) -> list[list[str]]:
        """Fixed seed: the organizers' re-run must reproduce our upload within 0.03 PA (Rule 8)."""
        default_max = 8192 if thinking else 4096
        params = self._sampling(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                                max_tokens=int(os.environ.get("DATAMATE_MAX_TOKENS", default_max)), seed=self.seed)
        outs = self.llm.chat(conversations, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking})
        return [[c.text for c in o.outputs] for o in outs]
