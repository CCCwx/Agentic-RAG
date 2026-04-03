"""
Helpers for robust structured-output calls.

Goal: simple + efficient.
- Try structured output once (fast path).
- If it fails, retry once with stricter instruction.
- If still fails, fall back to a lightweight parser/heuristic.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Type, TypeVar

from langchain_core.prompts import PromptTemplate

T = TypeVar("T")


def invoke_structured_with_retry(
    *,
    prompt: PromptTemplate,
    model: Any,
    schema: Type[T],
    inputs: dict,
    retries: int = 1,
    extra_strict_suffix: str = (
        "\n\nSTRICT OUTPUT:\n"
        "- Output must be valid JSON that matches the required schema.\n"
        "- Output JSON only. No markdown, no explanations, no extra keys.\n"
    ),
    fallback: Optional[Callable[[], T]] = None,
) -> T:
    """
    Robust wrapper around `model.with_structured_output(schema)`.
    """
    last_err: Exception | None = None

    def _run(p: PromptTemplate) -> T:
        chain = p | model.with_structured_output(schema)
        return chain.invoke(inputs)

    try:
        return _run(prompt)
    except Exception as e:
        last_err = e

    # Retry with stricter instruction appended
    for _ in range(max(0, retries)):
        try:
            strict_prompt = PromptTemplate(
                input_variables=prompt.input_variables,
                template=(prompt.template + extra_strict_suffix),
            )
            return _run(strict_prompt)
        except Exception as e:
            last_err = e

    if fallback is not None:
        return fallback()
    # Bubble up for callers that want to fail loud
    assert last_err is not None
    raise last_err
