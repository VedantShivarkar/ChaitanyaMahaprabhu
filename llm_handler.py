import logging
import os
from typing import Dict, Any

from config import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def call_llm_openai(context: str, question: str) -> str:
    """Minimal OpenAI ChatCompletion wrapper (gpt-4o-mini/gpt-3.5 style).

    If OpenAI is unavailable, this should not be called.
    """
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message.content.strip()


def call_llm_mock(context: str, question: str) -> str:
    """Mock response for hackathon demo without external API.

    Follows the grounding rules: if question words appear in context, echo
    a short answer-like snippet; otherwise return the exact refusal phrase.
    """
    ctx_lower = context.lower()
    q_lower = question.lower()

    keywords = [w for w in q_lower.split() if len(w) > 3]
    if any(k in ctx_lower for k in keywords):
        return (
            "Based on the provided documents, here is a relevant snippet:\n\n" """"""
            + context[:400]
        )
    return "Answer not found in provided documents."


def answer_question(context: str, question: str) -> Dict[str, Any]:
    """Call LLM (real or mock) with strict system prompt.

    Returns dict with at least: {"answer"}.
    """
    if not context.strip():
        answer = "Answer not found in provided documents."
        return {"answer": answer}

    if _has_openai_key():
        try:
            answer = call_llm_openai(context, question)
        except Exception:  # noqa: BLE001
            logger.exception("OpenAI call failed, falling back to mock")
            answer = call_llm_mock(context, question)
    else:
        answer = call_llm_mock(context, question)

    return {"answer": answer}
