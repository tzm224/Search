from __future__ import annotations

import logging
import time
import uuid


LOGGER = logging.getLogger(__name__)


def generate_snippet_id() -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = uuid.uuid4().int
    chars: list[str] = []
    for _ in range(7):
        value, remainder = divmod(value, 62)
        chars.append(alphabet[remainder])
    return "S_" + "".join(chars)


def generate_search_snippets(results: list[dict[str, str]]) -> str:
    if not results:
        return (
            "<tool_response>\n"
            "Google search encountered an error and was unable to extract valid information.\n"
            "</tool_response>"
        )

    blocks: list[str] = []
    for item in results:
        snippet_id = generate_snippet_id()
        blocks.append(
            "\n".join(
                [
                    f'<snippet id="{snippet_id}">',
                    f"Title: {item.get('title', '')}",
                    f"URL: {item.get('href', '')}",
                    f"Text: {item.get('body', '')}",
                    "</snippet>",
                ]
            )
        )

    return "<tool_response>\n" + "\n".join(blocks) + "\n</tool_response>"


def ddgs_search(
    query: str,
    *,
    top_k: int = 5,
    ddgs_backend: str = "auto",
    max_results: int = 10,
    retries: int = 5,
    retry_delay: float = 1.0,
) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "The `ddgs` package is required. Install project dependencies first."
        ) from exc

    for _ in range(retries):
        with DDGS(timeout=180) as client:
            try:
                results = client.text(
                    query,
                    region="us-en",
                    safesearch="off",
                    max_results=max_results,
                    timelimit=None,
                    ddgs_backend=ddgs_backend,
                )
                return [{"query": query, **item} for item in results][:top_k]
            except Exception:  # pragma: no cover - network/runtime dependent
                time.sleep(retry_delay)

    return []


def search(queries: list[str], top_k: int = 5) -> str:
    if not queries:
        return (
            "<tool_response>\n"
            "Google search encountered an error and was unable to extract valid information.\n"
            "</tool_response>"
        )

    query = queries[0].strip()
    start_time = time.time()
    results = ddgs_search(query, top_k=top_k, ddgs_backend="auto")
    elapsed = time.time() - start_time
    LOGGER.info("search query=%r elapsed=%.2fs results=%d", query, elapsed, len(results))
    return generate_search_snippets(results)
