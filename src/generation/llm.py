"""Local Ollama backend; no cloud credentials or SDK required."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import CLAIM_SCHEMA


class LLMUnavailable(RuntimeError):
    pass


class OllamaBackend:
    def __init__(self, model="qwen2.5:7b", base_url="http://localhost:11434", timeout=180,
                 num_ctx=4096, num_predict=1536, keep_alive=0):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must use HTTP or HTTPS.")
        if not model.strip():
            raise ValueError("Model name must not be empty.")
        if not math.isfinite(timeout) or timeout <= 0 or min(num_ctx, num_predict) < 1:
            raise ValueError("Timeout and token limits must be positive.")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.num_ctx, self.num_predict, self.keep_alive = num_ctx, num_predict, keep_alive
        self.last_generation = None

    def describe(self):
        return {"provider": "ollama", "model": self.model, "base_url": self.base_url,
                "num_ctx": self.num_ctx, "num_predict": self.num_predict,
                "timeout_seconds": self.timeout, "keep_alive": self.keep_alive,
                "last_generation": self.last_generation}

    def generate(self, prompt):
        return self.generate_structured(prompt, CLAIM_SCHEMA)

    def generate_structured(self, prompt, schema):
        self.last_generation = None
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "format": schema,
                           "keep_alive": self.keep_alive,
                           "options": {"temperature": 0, "seed": 42, "num_ctx": self.num_ctx,
                                       "num_predict": self.num_predict}}).encode("utf-8")
        request = Request(self.base_url + "/api/generate", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise LLMUnavailable("Ollama 返回内容过大，已停止处理；请减少输出长度。")
            envelope = json.loads(data)
            if not isinstance(envelope, dict) or not isinstance(envelope.get("response"), str):
                raise ValueError("Unexpected Ollama response envelope.")
            if envelope.get("error"):
                raise LLMUnavailable("Ollama 未完成生成，请检查服务和模型。")
            if envelope.get("done") is not True:
                raise LLMUnavailable("Ollama 未完成生成，请检查服务后重试。")
            if envelope.get("done_reason") == "length":
                raise LLMUnavailable("模型输出达到长度上限；请减少 --max-claims 或增加 --num-predict 后重试。")
            payload = json.loads(envelope["response"])
            duration = (envelope.get("eval_duration") or 0) / 1e9
            self.last_generation = {"model": envelope.get("model", self.model),
                                    "done_reason": envelope.get("done_reason"),
                                    "prompt_tokens": envelope.get("prompt_eval_count"),
                                    "generated_tokens": envelope.get("eval_count"),
                                    "total_seconds": (envelope.get("total_duration") or 0) / 1e9,
                                    "load_seconds": (envelope.get("load_duration") or 0) / 1e9,
                                    "generation_seconds": duration,
                                    "tokens_per_second": envelope.get("eval_count", 0) / duration if duration else None,
                                    "proposed_claims": payload.get("claims") if isinstance(payload, dict) else None,
                                    "proposed_selections": payload.get("selections") if isinstance(payload, dict) else None}
            return payload
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise LLMUnavailable("无法连接 Ollama 或生成超时，请启动服务并确认模型已下载。") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailable("Ollama 返回格式不符合 JSON 协议，请更换模型或重试。") from exc


def add_ollama_arguments(parser, include_model=True):
    if include_model:
        parser.add_argument("--model", default="qwen2.5:7b",
                            help="Local model name, e.g. qwen2.5:7b or llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=1536)
    parser.add_argument("--keep-alive", default="0", help="0 unloads model after each response; 2m retains briefly")


def backend_from_args(args, model=None):
    keep_alive = 0 if args.keep_alive == "0" else args.keep_alive
    return OllamaBackend(model or args.model, args.ollama_url, args.llm_timeout,
                         args.num_ctx, args.num_predict, keep_alive)
