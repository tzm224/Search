from __future__ import annotations

from typing import Any, Sequence
import time


def normalize_base_url(base_url: str) -> str:
    value = base_url.strip()
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    if value.endswith("/v1"):
        return value
    return value.rstrip("/") + "/v1"


def request_model(
    base_url: str,
    model_name: str,
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 8192,
    retries: int = 5,
    retry_delay: float = 1.0,
    api_key: str = "None",
    extra_body: dict[str, Any] | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is required. Install project dependencies first."
        ) from exc

    client = OpenAI(base_url=normalize_base_url(base_url), api_key=api_key)
    last_error: Exception | None = None

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=list(messages),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or {},
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            time.sleep(retry_delay)

    raise RuntimeError(
        f"Failed to get a response from model `{model_name}` after {retries} attempts: {last_error}"
    ) from last_error
