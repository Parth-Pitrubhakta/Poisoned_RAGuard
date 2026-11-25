"""vLLM offline target.

Faithfulness for Vicuna (upstream src/models/Vicuna.py via FastChat):
  * prompt = FastChat vicuna_v1.1 single-turn string (prag.llm.chat_formats)
  * tokenized on our side and passed as token ids. transformers-5 LlamaTokenizer == [BOS]+SentencePiece on all
    100 NQ poisoned prompts (verified 2026-11-19), i.e. identical to upstream's use_fast=False tokenizer.
  * fp16 (FastChat load_model on cuda), do_sample with temperature 0.1, max_new_tokens 150, repetition_penalty 1.0,
    plus HF generate's implicit defaults top_k=50, top_p=1.0 (Vicuna's generation_config sets none of these).
  * decode with skip_special_tokens=True, spaces_between_special_tokens=False.
  * seed: upstream model config seed=100. We give every request seed=100, so sampling is deterministic per prompt
    regardless of batching. The exact random draws still differ from HF generate (different RNG), so sampled outputs
    can differ token-for-token from an HF run. At temperature 0.1 this rarely changes the answer.
"""
from __future__ import annotations

import time

from .base import Generation
from .chat_formats import CHAT_FORMATS
from .parsers import PARSERS


class VLLMTarget:
    def __init__(self, hf_id: str, revision: str | None, chat_format: str, *, dtype: str = "auto",
                 temperature: float = 0.1, top_p: float = 1.0, top_k: int = 50, max_new_tokens: int = 150,
                 repetition_penalty: float = 1.0, seed: int = 100, max_model_len: int | None = None,
                 gpu_memory_utilization: float = 0.30, enforce_eager: bool = True, tensor_parallel_size: int = 1,
                 chat_template_kwargs: dict | None = None, output_parser: str = "none", engine_kwargs: dict | None = None):
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM, SamplingParams

        self.name = hf_id
        self.chat_format = chat_format
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.parser_name, self.parser = output_parser, PARSERS[output_parser]
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.llm = LLM(model=hf_id, revision=revision, tokenizer=hf_id, tokenizer_revision=revision, dtype=dtype,
                       seed=seed, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size,
                       generation_config="vllm",  # ignore model generation_config; params below are explicit
                       **(engine_kwargs or {}))
        # ...except its EOS ids: gpt-oss ends a turn with <|return|>, Qwen with <|im_end|>/<|endoftext|>.
        # Vicuna's generation_config eos (2) equals the tokenizer eos, so this changes nothing for the baseline.
        try:
            eos = GenerationConfig.from_pretrained(hf_id, revision=revision).eos_token_id
        except OSError:
            eos = None
        self.stop_token_ids = sorted({eos} if isinstance(eos, int) else set(eos or []))
        self.sampling = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                                       repetition_penalty=repetition_penalty, max_tokens=max_new_tokens, seed=seed,
                                       stop_token_ids=self.stop_token_ids or None,
                                       # harmony channels are special tokens: keep them so the parser can split
                                       skip_special_tokens=output_parser != "harmony", spaces_between_special_tokens=False)
        self.params = dict(hf_id=hf_id, revision=revision, chat_format=chat_format, dtype=dtype,
                           temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens,
                           repetition_penalty=repetition_penalty, seed=seed, max_model_len=max_model_len,
                           tensor_parallel_size=tensor_parallel_size, chat_template_kwargs=self.chat_template_kwargs,
                           output_parser=output_parser, engine="vllm", stop_token_ids=self.stop_token_ids,
                           engine_kwargs=dict(engine_kwargs or {}))

    def format_prompt(self, user_msg: str) -> str:
        if self.chat_format == "hf_chat_template":
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user_msg}],
                                                      tokenize=False, add_generation_prompt=True,
                                                      **self.chat_template_kwargs)
        return CHAT_FORMATS[self.chat_format](user_msg)

    def _token_ids(self, text: str) -> list[int]:
        # chat templates already contain BOS/special tokens as text
        add_special = self.chat_format != "hf_chat_template"
        return self.tokenizer(text, add_special_tokens=add_special).input_ids

    def generate(self, user_msgs: list[str], max_new_tokens: int | None = None) -> list[Generation]:
        from vllm.inputs import TokensPrompt

        ids = [self._token_ids(self.format_prompt(m)) for m in user_msgs]
        limit = self.params["max_model_len"]
        too_long = [len(x) for x in ids if limit and len(x) >= limit]
        if too_long:
            raise ValueError(f"{len(too_long)} prompt(s) exceed max_model_len={limit} (longest {max(too_long)} tokens)")
        prompts = [TokensPrompt(prompt_token_ids=x) for x in ids]
        sp = self.sampling if max_new_tokens is None else self.sampling.clone()
        if max_new_tokens is not None:
            sp.max_tokens = max_new_tokens
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        per = (time.perf_counter() - t0) / max(1, len(prompts))
        gens = []
        for m, o in zip(user_msgs, outs):
            raw = o.outputs[0].text
            to_parse = raw
            # Qwen3/3.5 templates may open '<think>' inside the prompt; restore it so a truncated thought
            # (no '</think>') parses as reasoning with an empty answer instead of being scored as an answer.
            if self.parser_name == "think_tags" and self.format_prompt(m).rstrip().endswith("<think>"):
                to_parse = "<think>" + raw
            text, reasoning = self.parser(to_parse)
            gens.append(Generation(text=text, prompt_tokens=len(o.prompt_token_ids),
                                   completion_tokens=len(o.outputs[0].token_ids), finish_reason=o.outputs[0].finish_reason,
                                   latency_s=per, raw_text=raw if raw != text else None, reasoning=reasoning))
        return gens
