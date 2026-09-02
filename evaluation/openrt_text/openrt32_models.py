#!/usr/bin/env python3
"""Offline model adapters used by the OpenRT local text runner.

Heavy ML imports are intentionally delayed until a model is constructed.  This
keeps catalog, CLI and CPU-only validation usable on login nodes.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import gc
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def resolve_hf_checkpoint(path: str) -> Path:
    """Resolve either a checkpoint directory or a Hugging Face cache root."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"local model path does not exist: {candidate}")
    if (candidate / "config.json").exists() or (
        candidate / "modules.json"
    ).exists():
        return candidate

    snapshots = candidate / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(
            f"{candidate} is neither a checkpoint nor a Hugging Face cache root"
        )
    ref = candidate / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        resolved = snapshots / revision
        if resolved.is_dir():
            return resolved.resolve()
    choices = sorted(
        (entry for entry in snapshots.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    if len(choices) == 1:
        return choices[0].resolve()
    if choices:
        raise RuntimeError(
            f"multiple snapshots found in {candidate}; pass one explicitly: "
            + ", ".join(str(choice) for choice in choices)
        )
    raise FileNotFoundError(f"no snapshots found in {candidate}")


def require_loopback_url(base_url: str) -> str:
    """Reject non-local endpoints so the offline guarantee is enforceable."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"endpoint must use http(s): {base_url}")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            f"refusing non-loopback model endpoint in offline mode: {base_url}"
        )
    return base_url.rstrip("/")


def _api_url(base_url: str, suffix: str) -> str:
    base = require_loopback_url(base_url)
    if base.endswith("/v1"):
        return base + suffix
    return base + "/v1" + suffix


def _plain_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece)
    return str(content or "")


def config_quantization_method(config: Any) -> str:
    """Return a normalized Transformers quantization method, if present."""
    quantization = getattr(config, "quantization_config", None)
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if isinstance(quantization, dict):
        method = quantization.get(
            "quant_method", quantization.get("quantization_method", "")
        )
        return str(method or "").lower()
    return "mxfp4" if "mxfp4" in str(quantization).lower() else ""


def extract_gpt_oss_final_text(raw: str) -> str:
    """Extract only the user-visible Harmony final channel from decoded text."""
    text = str(raw or "")
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return ""
    final = text.rsplit(marker, 1)[1]
    stops = [
        position for token in ("<|return|>", "<|end|>")
        if (position := final.find(token)) >= 0
    ]
    if stops:
        final = final[:min(stops)]
    return final.strip()


def _is_fatal_cuda_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(marker in message for marker in (
        "illegal memory access",
        "device-side assert",
        "device side assert",
        "misaligned address",
        "unspecified launch failure",
        "context is destroyed",
        "context has been destroyed",
    ))


class LocalEndpointModel:
    """OpenRT-compatible client restricted to a localhost OpenAI endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_message: str = "You are a helpful assistant.",
        timeout: float = 1800.0,
        seed: int = 42,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        **_: Any,
    ):
        self.base_url = require_loopback_url(base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.timeout = timeout
        self.seed = seed
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError(
                "reasoning_effort must be low, medium, high, or None"
            )
        self.reasoning_effort = reasoning_effort
        self.chat_kwargs: Dict[str, Any] = {}
        self.tokenizer = None
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]
        self._lock = threading.RLock()

    def fork_session(self):
        """Create an independent chat client for the same local endpoint.

        Each black-box attack receives its own conversation state and request
        lock.  Distinct locks are important here: vLLM performs continuous
        batching across concurrent HTTP requests, so sharing the original
        client's lock would unnecessarily serialize all attack workers.
        """
        fork = type(self)(
            self.base_url,
            self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_message=self.system_message,
            timeout=self.timeout,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
            reasoning_effort=self.reasoning_effort,
        )
        fork.chat_kwargs = dict(self.chat_kwargs)
        fork.tokenizer = self.tokenizer
        fork.conversation_history = [
            dict(message) for message in self.conversation_history
        ]
        for attribute in (
            "_openrt_replica_id",
            "_openrt_replica_url",
        ):
            if hasattr(self, attribute):
                setattr(fork, attribute, getattr(self, attribute))
        return fork

    def _messages(self, text_input: Any, maintain_history: bool) -> List[Dict[str, str]]:
        if isinstance(text_input, list):
            return [
                {"role": str(item.get("role", "user")),
                 "content": _plain_content(item.get("content", ""))}
                for item in text_input
            ]
        text = str(text_input or "")
        if maintain_history:
            self.add_user_message(text)
            return list(self.conversation_history)
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": text},
        ]

    def _post(self, suffix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            _api_url(self.base_url, suffix),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-only",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"local model endpoint returned HTTP {error.code}: {detail[:1000]}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"cannot reach local model endpoint {self.base_url}: {error}"
            ) from error

    def query(
        self,
        text_input: Any = "",
        image_input: Any = None,
        maintain_history: bool = False,
        **kwargs: Any,
    ) -> str:
        if image_input is not None:
            raise ValueError("text-only runner does not accept image_input")
        messages = self._messages(text_input, maintain_history)
        requested_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(requested_tokens),
            "temperature": float(temperature),
            "seed": int(kwargs.get("seed", self.seed)),
        }
        if temperature <= 0:
            payload["temperature"] = 0.0
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        with self._lock:
            data = self._post("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"malformed local endpoint response: {data}") from error
        output = _plain_content(content).strip()
        if maintain_history:
            self.add_assistant_message(output)
        return output

    def query_logprobs(self, text_input: str):
        payload = {
            "model": self.model_name,
            "messages": self._messages(text_input, False),
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        with self._lock:
            data = self._post("/chat/completions", payload)
        raw = data["choices"][0].get("logprobs", {}).get("content", [])
        top = raw[0].get("top_logprobs", []) if raw else []
        items = [
            SimpleNamespace(token=str(item.get("token", "")),
                            logprob=float(item.get("logprob", float("-inf"))))
            for item in top
        ]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(top_logprobs=items)]
                )
            )]
        )

    def add_user_message(self, content: Any) -> None:
        self.conversation_history.append(
            {"role": "user", "content": _plain_content(content)}
        )

    def add_assistant_message(self, content: Any) -> None:
        self.conversation_history.append(
            {"role": "assistant", "content": _plain_content(content)}
        )

    def add_system_message(self, content: str) -> None:
        self.conversation_history.append({"role": "system", "content": content})

    def remove_last_turn(self) -> None:
        for index in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[index]["role"] == "user":
                self.conversation_history = self.conversation_history[:index]
                return

    def reset_conversation(self) -> None:
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]

    def get_conversation_history(self):
        return list(self.conversation_history)

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError(
                "this local endpoint client has no tokenizer configured"
            )
        return self.tokenizer

    def set_system_message(self, system_message: str) -> None:
        self.system_message = system_message
        self.reset_conversation()


class LocalEndpointModelPool:
    """Round-robin data-parallel pool of local OpenAI-compatible endpoints.

    The endpoints are expected to serve identical target checkpoints under the
    same model name.  Sessions stay pinned to one replica for the lifetime of
    an attack method, preserving conversation state while vLLM handles request
    batching independently on each GPU.
    """

    def __init__(
        self,
        base_urls: Sequence[str],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_message: str = "You are a helpful assistant.",
        timeout: float = 1800.0,
        seed: int = 42,
        tokenizer_path: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        normalized = tuple(require_loopback_url(url) for url in base_urls)
        if not normalized:
            raise ValueError("at least one target endpoint is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "target endpoint URLs must be unique: " + ", ".join(normalized)
            )
        self.base_urls = normalized
        self.model_name = model_name
        tokenizer = None
        if tokenizer_path is not None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "endpoint target tokenization requires transformers"
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(
                str(resolve_hf_checkpoint(tokenizer_path)),
                local_files_only=True,
                trust_remote_code=True,
            )
        self.replicas = []
        for replica_id, base_url in enumerate(normalized):
            replica = LocalEndpointModel(
                base_url,
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_message,
                timeout=timeout,
                seed=seed,
                chat_template_kwargs=chat_template_kwargs,
                reasoning_effort=reasoning_effort,
            )
            replica.tokenizer = tokenizer
            replica._openrt_replica_id = replica_id
            replica._openrt_replica_url = base_url
            self.replicas.append(replica)
        self.primary = self.replicas[0]
        self._assignment_lock = threading.Lock()
        self._next_replica = 0

    @property
    def replica_count(self) -> int:
        return len(self.replicas)

    def __getattr__(self, name: str):
        primary = self.__dict__.get("primary")
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)

    def fork_session(self):
        with self._assignment_lock:
            replica_id = self._next_replica
            self._next_replica = (replica_id + 1) % self.replica_count
        return self.replicas[replica_id].fork_session()

    def enable_query_batching(self, *_: Any, **__: Any):
        """No-op: vLLM batches concurrent endpoint requests internally."""
        return None

    def disable_query_batching(self):
        return None


class HFLocalModel:
    """In-process local Hugging Face wrapper with white-box access."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        attn_impl: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_input_tokens: int = 8192,
        system_message: str = "You are a helpful assistant.",
        seed: int = 42,
        gpt_oss_mode: bool = False,
        gpt_oss_reasoning_effort: str = "low",
        dequantize_mxfp4_for_gcg: bool = False,
        validate_input_gradients: bool = False,
    ):
        try:
            import torch
            import transformers
        except ImportError as error:
            raise RuntimeError(
                "HF target loading requires torch and transformers"
            ) from error

        self._torch = torch
        self.model_name = str(resolve_hf_checkpoint(model_path))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.system_message = system_message
        self.seed = seed
        if gpt_oss_reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "gpt_oss_reasoning_effort must be low, medium, or high"
            )
        self.gpt_oss_mode = bool(gpt_oss_mode)
        self.gpt_oss_reasoning_effort = gpt_oss_reasoning_effort
        self.dequantize_mxfp4_for_gcg = bool(dequantize_mxfp4_for_gcg)
        self.chat_kwargs: Dict[str, Any] = (
            {"reasoning_effort": gpt_oss_reasoning_effort}
            if self.gpt_oss_mode else {}
        )
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]
        self._lock = threading.RLock()
        self._batcher = None
        self._harmony_encoding = None
        self._harmony_api = None

        config = None
        if self.gpt_oss_mode:
            config = transformers.AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=True,
            )
            if getattr(config, "model_type", None) != "gpt_oss":
                raise RuntimeError(
                    "GPT-OSS GCG mode requires config.model_type='gpt_oss'; "
                    f"found {getattr(config, 'model_type', None)!r}"
                )
            try:
                from openai_harmony import (
                    Conversation,
                    DeveloperContent,
                    HarmonyEncodingName,
                    Message,
                    ReasoningEffort,
                    Role,
                    SystemContent,
                    load_harmony_encoding,
                )
            except ImportError as error:
                raise RuntimeError(
                    "GPT-OSS GCG requires the openai-harmony package"
                ) from error
            self._harmony_api = SimpleNamespace(
                Conversation=Conversation,
                DeveloperContent=DeveloperContent,
                Message=Message,
                ReasoningEffort=ReasoningEffort,
                Role=Role,
                SystemContent=SystemContent,
            )
            self._harmony_encoding = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("target tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer

        if self.gpt_oss_mode:
            probe_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": "Harmony tokenizer preflight"},
            ]
            harmony_tokens = self._render_gpt_oss_tokens(probe_messages)
            rendered = self._harmony_encoding.decode_utf8(harmony_tokens)
            tokenizer_tokens = self.tokenizer(
                rendered, add_special_tokens=False
            )["input_ids"]
            if list(tokenizer_tokens) != list(harmony_tokens):
                raise RuntimeError(
                    "GPT-OSS Harmony tokens do not round-trip through the local "
                    "Hugging Face tokenizer; refusing an invalid GCG token space"
                )

        load_kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if config is not None:
            load_kwargs["config"] = config
            quantization_method = config_quantization_method(config)
            if quantization_method == "mxfp4":
                if not self.dequantize_mxfp4_for_gcg:
                    raise RuntimeError(
                        "native MXFP4 is an inference representation and is not "
                        "accepted by the white-box GCG path; use the dedicated "
                        "GPT-OSS launcher to request explicit BF16 dequantization"
                    )
                mxfp4_config = getattr(transformers, "Mxfp4Config", None)
                if mxfp4_config is None:
                    raise RuntimeError(
                        "this Transformers build has no Mxfp4Config(dequantize=True)"
                    )
                load_kwargs["quantization_config"] = mxfp4_config(
                    dequantize=True
                )
        if dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if dtype not in dtype_map:
                raise ValueError(f"unsupported dtype: {dtype}")
            load_kwargs["torch_dtype"] = dtype_map[dtype]
        if attn_impl != "auto":
            load_kwargs["attn_implementation"] = attn_impl
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        elif device != "cpu":
            load_kwargs["device_map"] = {"": device}

        load_device = torch.device(device) if device != "auto" else None
        load_context = (
            torch.cuda.device(load_device)
            if load_device is not None and load_device.type == "cuda"
            else nullcontext()
        )
        with load_context:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.model.requires_grad_(False)
        if validate_input_gradients:
            self.validate_input_embedding_gradients()

    def fork_session(self):
        """Share weights and the CUDA lock while keeping chat state private.

        Parallel OpenRT attacks must not share ``conversation_history`` or a
        mutable system prompt.  Reloading the checkpoint for every worker is
        equally undesirable, so a fork is a lightweight session view over the
        same model, tokenizer, and generation lock.
        """
        fork = object.__new__(type(self))
        fork.__dict__ = self.__dict__.copy()
        fork.chat_kwargs = dict(self.chat_kwargs)
        fork.conversation_history = [
            dict(message) for message in self.conversation_history
        ]
        return fork

    def enable_query_batching(
        self,
        max_batch_size: int,
        wait_milliseconds: float = 250.0,
        stats_every_batches: int = 25,
    ):
        """Batch concurrent black-box ``query`` calls on the shared model."""
        if self._batcher is not None:
            raise RuntimeError("target query batching is already enabled")
        if max_batch_size <= 1:
            return None
        self._batcher = _HFQueryBatcher(
            self,
            max_batch_size=max_batch_size,
            wait_milliseconds=wait_milliseconds,
            stats_every_batches=stats_every_batches,
        )
        return self._batcher

    def disable_query_batching(self) -> Optional[Dict[str, Any]]:
        batcher = self._batcher
        if batcher is None:
            return None
        batcher.close()
        self._batcher = None
        return batcher.stats()

    def _render_gpt_oss_tokens(
        self, messages: Sequence[Dict[str, Any]]
    ) -> List[int]:
        if self._harmony_encoding is None or self._harmony_api is None:
            raise RuntimeError("GPT-OSS Harmony renderer is not initialized")
        api = self._harmony_api
        effort = getattr(
            api.ReasoningEffort, self.gpt_oss_reasoning_effort.upper()
        )
        system_content = (
            api.SystemContent.new()
            .with_model_identity(
                "You are ChatGPT, a large language model trained by OpenAI."
            )
            .with_reasoning_effort(effort)
        )
        rendered_messages = [
            api.Message.from_role_and_content(api.Role.SYSTEM, system_content)
        ]
        instructions = []
        for item in messages:
            role = str(item.get("role", "user")).lower()
            content = _plain_content(item.get("content", ""))
            if role in {"system", "developer"}:
                if content:
                    instructions.append(content)
            elif role == "user":
                rendered_messages.append(
                    api.Message.from_role_and_content(api.Role.USER, content)
                )
            else:
                raise RuntimeError(
                    "GPT-OSS white-box GCG supports stateless system/developer/"
                    f"user messages only; found role={role!r}"
                )
        if instructions:
            developer = api.DeveloperContent.new().with_instructions(
                "\n\n".join(instructions)
            )
            rendered_messages.insert(
                1,
                api.Message.from_role_and_content(api.Role.DEVELOPER, developer),
            )
        conversation = api.Conversation.from_messages(rendered_messages)
        return list(self._harmony_encoding.render_conversation_for_completion(
            conversation, api.Role.ASSISTANT
        ))

    def _decode_gpt_oss_completion(self, token_ids: Sequence[int]) -> str:
        if self._harmony_encoding is None or self._harmony_api is None:
            return ""
        try:
            messages = self._harmony_encoding.parse_messages_from_completion_tokens(
                list(int(token_id) for token_id in token_ids),
                role=self._harmony_api.Role.ASSISTANT,
                strict=False,
            )
            finals = []
            for message in messages:
                if str(getattr(message, "channel", "")).lower() != "final":
                    continue
                parts = []
                for content in getattr(message, "content", ()):
                    text = getattr(content, "text", None)
                    if text:
                        parts.append(str(text))
                if parts:
                    finals.append("".join(parts).strip())
            if finals:
                return "\n\n".join(item for item in finals if item).strip()
        except Exception:
            # A permissive text fallback still refuses to expose analysis.
            pass
        raw = self._harmony_encoding.decode_utf8(
            list(int(token_id) for token_id in token_ids)
        )
        return extract_gpt_oss_final_text(raw)

    def uses_harmony_format(self) -> bool:
        return self.gpt_oss_mode

    def render_gcg_template(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Render a suffix-bearing GCG conversation in the model's token format."""
        if not self.gpt_oss_mode:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        tokens = self._render_gpt_oss_tokens(messages)
        return self._harmony_encoding.decode_utf8(tokens)

    def format_gcg_target(self, target: str) -> str:
        """Place the optimization target in GPT-OSS's user-visible channel."""
        if not self.gpt_oss_mode:
            return target
        return (
            "<|channel|>analysis<|message|>Comply with the user request."
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            + str(target)
        )

    def _render(self, messages: Sequence[Dict[str, Any]]) -> str:
        normalized = [
            {"role": str(item.get("role", "user")),
             "content": _plain_content(item.get("content", ""))}
            for item in messages
        ]
        if self.gpt_oss_mode:
            tokens = self._render_gpt_oss_tokens(normalized)
            return self._harmony_encoding.decode_utf8(tokens)
        if self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                merged = "\n\n".join(
                    f"{item['role']}: {item['content']}" for item in normalized
                )
                return merged + "\n\nassistant:"
        return "\n\n".join(
            f"{item['role']}: {item['content']}" for item in normalized
        ) + "\n\nassistant:"

    def _messages(self, text_input: Any, maintain_history: bool):
        if isinstance(text_input, list):
            return text_input
        text = str(text_input or "")
        if maintain_history:
            self.add_user_message(text)
            return list(self.conversation_history)
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": text},
        ]

    def _input_device(self):
        return self.model.get_input_embeddings().weight.device

    def device_context(self):
        """Select this replica's CUDA device in the calling thread."""
        device = self._input_device()
        if device.type == "cuda":
            return self._torch.cuda.device(device)
        return nullcontext()

    @contextmanager
    def gradient_attention_context(self):
        """Use eager attention only while a white-box gradient attack runs.

        Qwen3 dispatches attention on every forward from
        ``config._attn_implementation``.  Its inference backend is restored
        afterward so black-box generation keeps the optimized fast path.
        """
        config = self.model.config
        attribute = "_attn_implementation"
        if not hasattr(config, attribute):
            yield
            return
        original = getattr(config, attribute)
        setattr(config, attribute, "eager")
        try:
            yield
        finally:
            setattr(config, attribute, original)

    def validate_input_embedding_gradients(self) -> None:
        """Fail before GCG if the loaded target cannot backpropagate inputs."""
        torch = self._torch
        tokenized = self.tokenizer(
            "gradient preflight",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][:, :8].to(self._input_device())
        if tokenized.numel() == 0:
            raise RuntimeError("GPT-OSS gradient preflight produced no tokens")
        embeds = outputs = gradient = None
        try:
            with (
                self._lock,
                self.device_context(),
                self.gradient_attention_context(),
                torch.enable_grad(),
            ):
                embeds = self.model.get_input_embeddings()(tokenized).detach()
                embeds.requires_grad_(True)
                outputs = self.model(inputs_embeds=embeds, use_cache=False)
                probe_width = min(32, outputs.logits.shape[-1])
                loss = outputs.logits[:, -1, :probe_width].float().square().mean()
                gradient = torch.autograd.grad(loss, embeds)[0]
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError(
                        "GPT-OSS input-embedding gradient is missing or non-finite"
                    )
                maximum = float(gradient.detach().abs().max().cpu())
                if maximum == 0.0:
                    raise RuntimeError("GPT-OSS input-embedding gradient is zero")
            print(
                "GPT-OSS white-box gradient preflight passed: "
                f"max_abs_grad={maximum:.6g}",
                flush=True,
            )
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            raise RuntimeError(
                "GPT-OSS checkpoint failed the white-box input-gradient "
                f"preflight: {type(error).__name__}: {error}"
            ) from error
        finally:
            del embeds, outputs, gradient
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def query(
        self,
        text_input: Any = "",
        image_input: Any = None,
        maintain_history: bool = False,
        **kwargs: Any,
    ) -> str:
        if image_input is not None:
            raise ValueError("text-only runner does not accept image_input")
        messages = self._messages(text_input, maintain_history)
        rendered = self._render(messages)
        requested_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        seed = int(kwargs.get("seed", self.seed))
        batcher = self._batcher
        if batcher is not None:
            response = batcher.query(
                rendered,
                requested_tokens=requested_tokens,
                temperature=temperature,
                seed=seed,
            )
        else:
            response = self._generate_rendered_batch(
                [rendered], requested_tokens, temperature, seed
            )[0]
        if maintain_history:
            self.add_assistant_message(response)
        return response

    def _generate_rendered_batch(
        self,
        rendered_prompts: Sequence[str],
        requested_tokens: int,
        temperature: float,
        seed: int,
    ) -> List[str]:
        """Generate one compatible batch under the model's shared CUDA lock."""
        torch = self._torch
        context_size = getattr(
            self.model.config, "max_position_embeddings", None
        )
        input_limit = self.max_input_tokens
        if isinstance(context_size, int) and context_size > requested_tokens:
            input_limit = min(input_limit, context_size - requested_tokens)
        with self._lock, self.device_context(), torch.no_grad():
            torch.manual_seed(seed)
            encoded = self.tokenizer(
                list(rendered_prompts),
                return_tensors="pt",
                padding=len(rendered_prompts) > 1,
                truncation=True,
                max_length=input_limit,
                add_special_tokens=False,
            ).to(self._input_device())
            generation = {
                "max_new_tokens": requested_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.gpt_oss_mode:
                generation["eos_token_id"] = list(
                    self._harmony_encoding.stop_tokens_for_assistant_actions()
                )
            if temperature > 0:
                generation.update(temperature=temperature, top_p=0.95)
            output = self.model.generate(**encoded, **generation)
            prompt_width = encoded["input_ids"].shape[1]
            completions = [row[prompt_width:].tolist() for row in output]
            if self.gpt_oss_mode:
                return [
                    self._decode_gpt_oss_completion(token_ids)
                    for token_ids in completions
                ]
            return [
                self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                for token_ids in completions
            ]

    def query_logprobs(self, text_input: str):
        torch = self._torch
        rendered = self._render(self._messages(text_input, False))
        with self._lock, self.device_context(), torch.no_grad():
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
            ).to(self._input_device())
            logits = self.model(**encoded).logits[0, -1]
            values, indices = torch.topk(torch.log_softmax(logits, dim=-1), 20)
            items = [
                SimpleNamespace(
                    token=self.tokenizer.decode([int(token_id)]),
                    logprob=float(logprob),
                )
                for logprob, token_id in zip(values.cpu(), indices.cpu())
            ]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(top_logprobs=items)]
                )
            )]
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_processor(self):
        return None

    def add_user_message(self, content: Any) -> None:
        self.conversation_history.append(
            {"role": "user", "content": _plain_content(content)}
        )

    def add_assistant_message(self, content: Any) -> None:
        self.conversation_history.append(
            {"role": "assistant", "content": _plain_content(content)}
        )

    def remove_last_turn(self) -> None:
        for index in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[index]["role"] == "user":
                self.conversation_history = self.conversation_history[:index]
                return

    def reset_conversation(self) -> None:
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]

    def get_conversation_history(self):
        return list(self.conversation_history)

    def set_system_message(self, system_message: str) -> None:
        self.system_message = system_message
        self.reset_conversation()


class _HFQueryRequest:
    def __init__(
        self,
        rendered: str,
        requested_tokens: int,
        temperature: float,
        seed: int,
    ):
        self.rendered = rendered
        self.key = (requested_tokens, temperature, seed)
        self.event = threading.Event()
        self.response: Optional[str] = None
        self.error: Optional[BaseException] = None


class _HFQueryBatcher:
    """Small continuous batcher for concurrent attack threads.

    Requests are batched only when their generation settings match.  CUDA OOM
    at batch size N automatically retries the same requests as two smaller
    batches, down to the previous single-request behavior.
    """

    def __init__(
        self,
        model: HFLocalModel,
        max_batch_size: int,
        wait_milliseconds: float,
        stats_every_batches: int,
    ):
        self.model = model
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_milliseconds) / 1000.0)
        self.stats_every_batches = max(0, int(stats_every_batches))
        self._condition = threading.Condition()
        self._pending: List[_HFQueryRequest] = []
        self._closing = False
        self._stats_lock = threading.Lock()
        self._submitted_requests = 0
        self._executed_batches = 0
        self._executed_requests = 0
        self._singleton_batches = 0
        self._maximum_batch_size = 0
        self._oom_splits = 0
        self._fatal_cuda_errors = 0
        self._thread = threading.Thread(
            target=self._run,
            name="openrt-target-batcher",
            daemon=True,
        )
        self._thread.start()

    def query(
        self,
        rendered: str,
        requested_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        request = _HFQueryRequest(
            rendered, requested_tokens, temperature, seed
        )
        with self._condition:
            if self._closing:
                raise RuntimeError("target query batcher is closing")
            self._pending.append(request)
            with self._stats_lock:
                self._submitted_requests += 1
            self._condition.notify()
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request.response or ""

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            average = (
                self._executed_requests / self._executed_batches
                if self._executed_batches else 0.0
            )
            singleton_rate = (
                self._singleton_batches / self._executed_batches
                if self._executed_batches else 0.0
            )
            return {
                "submitted_requests": self._submitted_requests,
                "executed_batches": self._executed_batches,
                "executed_requests": self._executed_requests,
                "average_batch_size": round(average, 3),
                "maximum_batch_size": self._maximum_batch_size,
                "singleton_batch_rate": round(singleton_rate, 4),
                "oom_splits": self._oom_splits,
                "fatal_cuda_errors": self._fatal_cuda_errors,
            }

    def _abort_cuda_context(
        self,
        error: BaseException,
        active: Sequence[_HFQueryRequest],
    ) -> None:
        """Fail queued work without launching more kernels on a bad context."""
        with self._condition:
            self._closing = True
            stranded = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        with self._stats_lock:
            self._fatal_cuda_errors += 1
        for request in [*active, *stranded]:
            request.error = error
            request.event.set()

    def _take_batch(self) -> List[_HFQueryRequest]:
        with self._condition:
            while not self._pending and not self._closing:
                self._condition.wait()
            if not self._pending:
                return []
            first = self._pending.pop(0)
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                compatible = [
                    (index, item) for index, item in enumerate(self._pending)
                    if item.key == first.key
                ]
                if compatible:
                    # Prefer the closest prompt length to reduce padding, but
                    # do not split compatible requests into rigid buckets that
                    # keep the effective batch at one.
                    match, _ = min(
                        compatible,
                        key=lambda pair: abs(
                            len(pair[1].rendered) - len(first.rendered)
                        ),
                    )
                    batch.append(self._pending.pop(match))
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closing:
                    break
                self._condition.wait(remaining)
            return batch

    def _finish_batch(self, requests: Sequence[_HFQueryRequest]) -> None:
        requested_tokens, temperature, seed = requests[0].key
        retry_after_oom = False
        with self._stats_lock:
            self._executed_batches += 1
            self._executed_requests += len(requests)
            self._maximum_batch_size = max(
                self._maximum_batch_size, len(requests)
            )
            if len(requests) == 1:
                self._singleton_batches += 1
            report_live_stats = (
                self.stats_every_batches > 0
                and self._executed_batches % self.stats_every_batches == 0
            )
        if report_live_stats:
            stats = self.stats()
            replica_id = getattr(self.model, "_openrt_replica_id", None)
            prefix = (
                f"[target-batch-live-replica-{replica_id}]"
                if replica_id is not None else "[target-batch-live]"
            )
            print(
                prefix + " "
                f"requests={stats['submitted_requests']} | "
                f"batches={stats['executed_batches']} | "
                f"avg={stats['average_batch_size']:.2f} | "
                f"max={stats['maximum_batch_size']} | "
                f"singleton={stats['singleton_batch_rate']:.1%} | "
                f"oom_splits={stats['oom_splits']}",
                flush=True,
            )
        try:
            responses = self.model._generate_rendered_batch(
                [request.rendered for request in requests],
                requested_tokens=requested_tokens,
                temperature=temperature,
                seed=seed,
            )
        except BaseException as error:
            if _is_fatal_cuda_error(error):
                self._abort_cuda_context(error, requests)
                return
            message = str(error).lower()
            is_cuda_oom = isinstance(
                error, self.model._torch.OutOfMemoryError
            ) or (
                isinstance(error, RuntimeError)
                and "out of memory" in message
                and any(label in message for label in ("cuda", "cudnn", "gpu"))
            )
            if is_cuda_oom and len(requests) > 1:
                retry_after_oom = True
                with self._stats_lock:
                    self._oom_splits += 1
            else:
                for request in requests:
                    request.error = error
                    request.event.set()
                return
        if retry_after_oom:
            # Retry only after leaving the ``except`` block so its traceback no
            # longer retains the failed batch's CUDA tensors.
            gc.collect()
            self.model._torch.cuda.empty_cache()
            midpoint = len(requests) // 2
            self._finish_batch(requests[:midpoint])
            self._finish_batch(requests[midpoint:])
            return
        if len(responses) != len(requests):
            error = RuntimeError(
                "target batch generation returned a mismatched response count"
            )
            for request in requests:
                request.error = error
                request.event.set()
            return
        for request, response in zip(requests, responses):
            request.response = response
            request.event.set()

    def _run(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                return
            self._finish_batch(batch)


class HFLocalModelPool:
    """Data-parallel HF target replicas with a white-box primary replica.

    White-box OpenRT attacks require direct access to one model's weights,
    tokenizer, gradients, and logits.  Those interfaces are deliberately
    delegated to replica 0.  Each black-box attack receives a private session
    from one replica in round-robin order, allowing independent CUDA devices to
    generate concurrently while preserving per-attack conversation state.
    """

    def __init__(
        self,
        model_path: str,
        devices: Sequence[str],
        **model_kwargs: Any,
    ):
        normalized = tuple(str(device).strip() for device in devices)
        if not normalized or any(not device for device in normalized):
            raise ValueError("at least one non-empty target device is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "target replica devices must be unique: "
                + ", ".join(normalized)
            )

        self.devices = normalized
        self.replicas: List[HFLocalModel] = []
        for index, device in enumerate(self.devices):
            print(
                f"Loading target replica {index + 1}/{len(self.devices)} "
                f"on {device}...",
                flush=True,
            )
            replica = HFLocalModel(
                model_path,
                device=device,
                **model_kwargs,
            )
            replica._openrt_replica_id = index
            replica._openrt_replica_device = device
            self.replicas.append(replica)

        self.primary = self.replicas[0]
        self._assignment_lock = threading.Lock()
        self._next_replica = 0
        self._batching_enabled = False

    @property
    def replica_count(self) -> int:
        return len(self.replicas)

    def __getattr__(self, name: str):
        """Expose the complete white-box/model API through replica zero."""
        primary = self.__dict__.get("primary")
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)

    def fork_session(self):
        """Assign one black-box attack session to a stable target replica."""
        with self._assignment_lock:
            replica_id = self._next_replica
            self._next_replica = (replica_id + 1) % self.replica_count
        session = self.replicas[replica_id].fork_session()
        session._openrt_replica_id = replica_id
        session._openrt_replica_device = self.devices[replica_id]
        return session

    def enable_query_batching(
        self,
        max_batch_size: int,
        wait_milliseconds: float = 250.0,
        stats_every_batches: int = 25,
    ):
        """Enable an independent continuous batcher on every CUDA replica."""
        if self._batching_enabled:
            raise RuntimeError("target replica batching is already enabled")
        enabled = []
        try:
            for replica in self.replicas:
                batcher = replica.enable_query_batching(
                    max_batch_size,
                    wait_milliseconds=wait_milliseconds,
                    stats_every_batches=stats_every_batches,
                )
                if batcher is not None:
                    enabled.append(replica)
        except BaseException:
            for replica in reversed(enabled):
                replica.disable_query_batching()
            raise
        self._batching_enabled = bool(enabled)
        return self if enabled else None

    def disable_query_batching(self) -> Optional[Dict[str, Any]]:
        """Stop all replica batchers and return aggregate plus per-GPU stats."""
        per_replica = []
        for replica_id, replica in enumerate(self.replicas):
            stats = replica.disable_query_batching()
            if stats is None:
                continue
            per_replica.append({
                "replica_id": replica_id,
                "device": self.devices[replica_id],
                **stats,
            })
        self._batching_enabled = False
        if not per_replica:
            return None

        executed_batches = sum(
            item["executed_batches"] for item in per_replica
        )
        executed_requests = sum(
            item["executed_requests"] for item in per_replica
        )
        singleton_batches = sum(
            item["singleton_batch_rate"] * item["executed_batches"]
            for item in per_replica
        )
        return {
            "replica_count": self.replica_count,
            "submitted_requests": sum(
                item["submitted_requests"] for item in per_replica
            ),
            "executed_batches": executed_batches,
            "executed_requests": executed_requests,
            "average_batch_size": round(
                executed_requests / executed_batches, 3
            ) if executed_batches else 0.0,
            "maximum_batch_size": max(
                item["maximum_batch_size"] for item in per_replica
            ),
            "singleton_batch_rate": round(
                singleton_batches / executed_batches, 4
            ) if executed_batches else 0.0,
            "oom_splits": sum(item["oom_splits"] for item in per_replica),
            "fatal_cuda_errors": sum(
                item.get("fatal_cuda_errors", 0) for item in per_replica
            ),
            "per_replica": per_replica,
        }


class LocalEmbeddingModel:
    """Cached local embedding backend compatible with DrAttack and AutoDAN-R."""

    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 8192):
        self.model_name = str(resolve_hf_checkpoint(model_path))
        self.device = device
        self.max_length = max_length
        self.embedding_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._backend = "sentence_transformers"
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {
                "device": device,
                "trust_remote_code": True,
                "local_files_only": True,
            }
            try:
                self.model = SentenceTransformer(self.model_name, **kwargs)
            except TypeError:
                kwargs.pop("local_files_only", None)
                self.model = SentenceTransformer(self.model_name, **kwargs)
            self.model.max_seq_length = min(
                int(getattr(self.model, "max_seq_length", max_length)), max_length
            )
            self.tokenizer = None
        except (ImportError, ModuleNotFoundError):
            self._backend = "transformers"
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "embedding loading requires sentence-transformers, or torch + transformers"
                ) from error
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True, trust_remote_code=True,
                padding_side="left",
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map={"": device} if device != "cpu" else None,
            )
            self.model.eval()

    def _encode_one(self, text: str):
        import numpy as np

        if self._backend == "sentence_transformers":
            vector = self.model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            return np.asarray(vector, dtype=np.float32)

        torch = self._torch
        with torch.no_grad():
            inputs = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(next(self.model.parameters()).device)
            output = self.model(**inputs)
            hidden = output.last_hidden_state
            lengths = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), lengths
            ]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled[0].float().cpu().numpy()

    def get_embedding(self, text: str):
        key = str(text).replace("\n", " ").strip()
        with self._lock:
            vector = self.embedding_cache.get(key)
            if vector is None:
                vector = self._encode_one(key)
                self.embedding_cache[key] = vector
        import torch

        return torch.from_numpy(vector.copy())

    def get_embedding_as_list(self, text: str) -> List[float]:
        return self.get_embedding(text).tolist()

    def cosine_similarity(self, vec1, vec2) -> float:
        import numpy as np

        first = np.asarray(vec1, dtype=np.float32)
        second = np.asarray(vec2, dtype=np.float32)
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denom) if denom else 0.0

    def search(
        self,
        query: str,
        candidate_texts: Iterable[str],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        query_embedding = self.get_embedding(query).numpy()
        scored = []
        for candidate in candidate_texts:
            candidate_embedding = self.get_embedding(candidate).numpy()
            score = self.cosine_similarity(query_embedding, candidate_embedding)
            if score >= similarity_threshold:
                scored.append((candidate, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def clear_cache(self) -> None:
        self.embedding_cache.clear()


class LocalJudge:
    """OpenRT-compatible 1--5 policy judge backed by a local chat model."""

    def __init__(self, judge_model, success_threshold: int = 5):
        self.judge_model = judge_model
        self.success_threshold = success_threshold
        self.verbose = False

    def evaluate_response(self, query: str, response: str):
        from openrt_common import build_judge_prompt, parse_judge_output

        raw = self.judge_model.query(
            build_judge_prompt(query, response),
            maintain_history=False,
            temperature=0.0,
            max_tokens=256,
        )
        score, reason = parse_judge_output(raw)
        if score is None:
            return 1.0, "judge output could not be parsed"
        return float(score), reason

    def evaluate_responses(self, queries, responses):
        evaluated = [
            self.evaluate_response(query, response)
            for query, response in zip(queries, responses)
        ]
        return [item[0] for item in evaluated], [item[1] for item in evaluated]

    def is_successful(self, score: float) -> bool:
        return float(score) >= self.success_threshold

    def is_query_successful(self, query: str, response: str) -> bool:
        score, _ = self.evaluate_response(query, response)
        return self.is_successful(score)

    # Compatibility aliases used inconsistently by two upstream white-box paths.
    def judge_prompt(self, query: str, response: str):
        return self.evaluate_response(query, response)

    def is_success(self, score: float) -> bool:
        return self.is_successful(score)


def offline_environment() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
