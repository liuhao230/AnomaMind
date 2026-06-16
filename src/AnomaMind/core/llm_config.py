from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: str = "http://localhost:8000/v1"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")


def create_llm(config: LLMConfig) -> BaseChatModel:
    llm_kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "base_url": config.base_url,
    }

    if config.api_key:
        llm_kwargs["api_key"] = config.api_key
    if config.max_tokens:
        llm_kwargs["max_tokens"] = config.max_tokens
    if config.timeout:
        llm_kwargs["timeout"] = config.timeout

    model_kwargs: Dict[str, Any] = {}
    default_extra_body: Dict[str, Any] = {}

    if config.extra_params:
        generation_params = {
            "repetition_penalty",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
        }

        for key, value in config.extra_params.items():
            if key in generation_params:
                default_extra_body[key] = value
            else:
                model_kwargs[key] = value

        if model_kwargs:
            llm_kwargs["model_kwargs"] = model_kwargs

    try:
        import httpx  # type: ignore

        llm_kwargs.setdefault("http_client", httpx.Client(trust_env=False))
        llm_kwargs.setdefault("http_async_client", httpx.AsyncClient(trust_env=False))
    except Exception:
        pass

    try:
        llm = ChatOpenAI(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("http_client", None)
        llm_kwargs.pop("http_async_client", None)
        llm = ChatOpenAI(**llm_kwargs)

    if default_extra_body:
        original_create = llm.client.create

        def create_with_extra_body(*args, **kwargs):
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict) and extra_body:
                kwargs["extra_body"] = {**default_extra_body, **extra_body}
            elif extra_body:
                kwargs["extra_body"] = {**default_extra_body, **extra_body}
            else:
                kwargs["extra_body"] = default_extra_body.copy()
            return original_create(*args, **kwargs)

        llm.client.create = create_with_extra_body

        if getattr(llm, "async_client", None):
            original_async_create = llm.async_client.create

            async def async_create_with_extra_body(*args, **kwargs):
                extra_body = kwargs.get("extra_body")
                if isinstance(extra_body, dict) and extra_body:
                    kwargs["extra_body"] = {**default_extra_body, **extra_body}
                elif extra_body:
                    kwargs["extra_body"] = {**default_extra_body, **extra_body}
                else:
                    kwargs["extra_body"] = default_extra_body.copy()
                return await original_async_create(*args, **kwargs)

            llm.async_client.create = async_create_with_extra_body

    return llm


def default_locator_llm_config() -> LLMConfig:
    max_tokens_env = os.getenv("LOCATOR_MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env and max_tokens_env.strip().isdigit() else None
    return LLMConfig(
        model_name=os.getenv("LOCATOR_VISION_MODEL", os.getenv("AUX_LLM_MODEL", "gpt-4o-mini")),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        temperature=float(os.getenv("LOCATOR_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
    )
