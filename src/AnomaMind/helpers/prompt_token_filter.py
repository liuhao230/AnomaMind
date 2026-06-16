from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

TextTokenCounter = Callable[[str], int]
MessageTokenCounter = Callable[[Sequence[Mapping[str, Any]]], int]


def build_text_token_counter(tokenizer_path: str) -> TextTokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)

    def _count(text: str) -> int:
        return len(tokenizer(text or "", add_special_tokens=False).input_ids)

    return _count


def build_message_token_counter(tokenizer_path: str) -> MessageTokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)

    def _fallback_text(messages: Sequence[Mapping[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            content = message.get("content", "")
            parts.append(content if isinstance(content, str) else str(content or ""))
        return "\n\n".join(parts)

    def _count(messages: Sequence[Mapping[str, Any]]) -> int:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                token_ids = tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(token_ids)
            except Exception:
                pass
        return len(tokenizer(_fallback_text(messages), add_special_tokens=False).input_ids)

    return _count


def keep_text_prompt(
    text: str,
    *,
    max_prompt_tokens: int,
    token_counter: TextTokenCounter,
) -> bool:
    if int(max_prompt_tokens) <= 0:
        return True
    return int(token_counter(text or "")) <= int(max_prompt_tokens)


def keep_message_prompt(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_prompt_tokens: int,
    token_counter: MessageTokenCounter,
) -> bool:
    if int(max_prompt_tokens) <= 0:
        return True
    return int(token_counter(messages)) <= int(max_prompt_tokens)
