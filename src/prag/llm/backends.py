"""Non-vLLM targets + the factory.

* EchoLLM: CPU test double. It answers with the first context passage verbatim, so it "believes"
  whatever ranks first. That makes attack success a pure function of retrieval, which is useful for testing.
* OllamaLLM: the system Ollama service at 127.0.0.1:11434. It is a *shared* service, so only use it when approved.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .base import Generation, TargetLLM
from .parsers import PARSERS


class EchoLLM:
    def __init__(self, name: str = "echo"):
        self.name, self.params = name, {"engine": "echo"}

    def format_prompt(self, user_msg: str) -> str:
        return user_msg

    def generate(self, user_msgs, max_new_tokens=None):
        out = []
        for m in user_msgs:
            ctx = m.split("Contexts:", 1)[-1].split("Query:", 1)[0].strip()
            text = ctx.split("\n", 1)[0].strip()
            out.append(Generation(text=text, prompt_tokens=len(m.split()), completion_tokens=len(text.split()),
                                  finish_reason="stop", latency_s=0.0))
        return out


class OllamaLLM:
    def __init__(self, tag: str, *, temperature: float, top_p: float, top_k: int, max_new_tokens: int,
                 repetition_penalty: float, seed: int, output_parser: str = "none", host: str = "http://127.0.0.1:11434"):
        self.name, self.host, self.parser = tag, host, PARSERS[output_parser]
        self.options = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "num_predict": max_new_tokens,
                        "repeat_penalty": repetition_penalty, "seed": seed}
        self.params = {"engine": "ollama", "tag": tag, **self.options}

    def format_prompt(self, user_msg):
        return user_msg  # Ollama applies the model's chat template server-side

    def generate(self, user_msgs, max_new_tokens=None):
        opts = self.options if max_new_tokens is None else {**self.options, "num_predict": max_new_tokens}
        out = []
        for m in user_msgs:
            body = json.dumps({"model": self.name, "messages": [{"role": "user", "content": m}], "stream": False,
                               "options": opts}).encode()
            t0 = time.perf_counter()
            req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            msg = d.get("message", {})
            text, reasoning = self.parser(msg.get("content", ""))
            out.append(Generation(text=text, prompt_tokens=d.get("prompt_eval_count", 0), completion_tokens=d.get("eval_count", 0),
                                  finish_reason=d.get("done_reason"), latency_s=time.perf_counter() - t0,
                                  reasoning=msg.get("thinking") or reasoning))
        return out


def build_llm(llm_cfg, registry: dict) -> TargetLLM:
    spec = registry["llms"][llm_cfg.key]
    engine = spec.get("engine")
    temp = llm_cfg.temperature if llm_cfg.do_sample else 0.0
    if engine == "echo":
        return EchoLLM(llm_cfg.key)
    if engine == "ollama":
        return OllamaLLM(spec["ollama_tag"], temperature=temp, top_p=llm_cfg.top_p, top_k=llm_cfg.top_k,
                         max_new_tokens=llm_cfg.max_new_tokens, repetition_penalty=llm_cfg.repetition_penalty,
                         seed=llm_cfg.seed, output_parser=llm_cfg.output_parser)
    if engine == "vllm":
        from .vllm_backend import VLLMTarget

        return VLLMTarget(spec["hf_id"], spec.get("revision"), spec["chat_format"], dtype=spec.get("dtype", "auto"),
                          temperature=temp, top_p=llm_cfg.top_p, top_k=llm_cfg.top_k,
                          max_new_tokens=llm_cfg.max_new_tokens, repetition_penalty=llm_cfg.repetition_penalty,
                          seed=llm_cfg.seed, max_model_len=llm_cfg.max_model_len or spec.get("max_model_len"),
                          gpu_memory_utilization=llm_cfg.gpu_memory_utilization,
                          tensor_parallel_size=llm_cfg.tensor_parallel_size,
                          chat_template_kwargs=llm_cfg.chat_template_kwargs, output_parser=llm_cfg.output_parser,
                          engine_kwargs=spec.get("engine_kwargs"))
    raise ValueError(f"unknown llm engine {engine!r} for {llm_cfg.key}")
