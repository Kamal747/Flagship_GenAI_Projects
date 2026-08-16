"""
Chatbot layer: the LLM (via Groq) ONLY narrates/explains numbers that were
already computed deterministically by calculator.py / comparison.py. It
never performs its own arithmetic — build_chat_context() packages the
already-calculated results as the model's sole source of truth.

Includes automatic model fallback: if the active model is out of capacity
(429 rate limit, or a 413 "tokens per minute" quota error), the engine
transparently switches to the next model in the fallback chain and retries
the same request — conversation context is preserved either way.
"""
from __future__ import annotations

import json
import time

from groq import Groq, APIError, APIConnectionError, RateLimitError

from app.core.config import settings

SYSTEM_PROMPT = """You are an Electricity Bill Assistant. You help the user understand how their
electricity bill would differ under the OLD tariff policy (previous government) versus the
NEW tariff policy (current government), for the SAME units of consumption they entered.

CRITICAL RULE: You must NEVER calculate or invent any number yourself — no units, no rates,
no amounts, no percentages. ALL numeric facts you state MUST come directly from the
structured JSON data provided below, which was computed by deterministic Python code.
If the data needed to answer isn't in the JSON, say so — do not estimate or guess.

Note: "previous_bill"/"previous_units" in the JSON refer to the OLD POLICY bill, and
"current_bill"/"current_units" refer to the NEW POLICY bill — both computed for the exact
same consumption (units_consumed). This is a tariff POLICY comparison, not a previous-month
vs current-month comparison.

You may explain, summarize, compare, and give qualitative insight (e.g. why the new policy
is cheaper/costlier at this usage level, which slab matters most) as long as every NUMBER
you cite is copied from the JSON data.

If the user asks something unrelated to their electricity bill (e.g. general chit-chat,
unrelated topics), politely explain that you're designed specifically for electricity bill
analysis and steer them back.

Structured bill data (ground truth — the ONLY source for any number you state):
{context_json}
"""

_RATE_LIMIT_COOLDOWN_SECONDS = 60
_rate_limited_until: dict[str, float] = {}
_QUOTA_MARKERS = ("rate_limit_exceeded", "tokens per minute", "requests per minute", "rate limit")
_QUOTA_STATUS_CODES = {429, 413}


class ChatbotError(Exception):
    """Friendly, user-facing error for chatbot/LLM failures."""


def build_chat_context(comparison_summary: dict) -> str:
    """Serializes the deterministic comparison summary into the JSON block
    injected into the system prompt as the model's only source of truth."""
    return json.dumps(comparison_summary, indent=2, default=str)


def _client() -> Groq:
    if not settings.groq_api_key:
        raise ChatbotError("Groq API key is missing. Please set GROQ_API_KEY in your .env file.")
    return Groq(api_key=settings.groq_api_key)


def _is_quota_exhausted(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in _QUOTA_STATUS_CODES:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _candidate_models() -> list[str]:
    now = time.monotonic()
    chain = list(settings.groq_model_fallback_chain) or [settings.groq_model]
    fresh = [m for m in chain if _rate_limited_until.get(m, 0) <= now]
    cooling = [m for m in chain if _rate_limited_until.get(m, 0) > now]
    return fresh + cooling


def _create_completion_with_fallback(client: Groq, **kwargs):
    models_to_try = _candidate_models()
    last_error: Exception | None = None

    for model in models_to_try:
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except RateLimitError as e:
            _rate_limited_until[model] = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
            last_error = e
            continue
        except APIConnectionError:
            raise ChatbotError("Could not connect to Groq. Please check your internet connection.")
        except APIError as e:
            if _is_quota_exhausted(e):
                _rate_limited_until[model] = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
                last_error = e
                continue
            raise ChatbotError(f"Groq API error: {e}")

    raise ChatbotError(
        f"All configured Groq models are currently out of capacity/quota. Please wait and try again."
    ) from last_error


def ask_chatbot(user_message: str, comparison_summary: dict, conversation_history: list[dict] | None = None) -> str:
    """
    Sends the user's question to Groq along with the deterministic bill data
    as context. Returns the model's natural-language reply (grounded in that
    data by the system prompt's strict instructions).
    """
    client = _client()
    context_json = build_chat_context(comparison_summary)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context_json=context_json)}]
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    response = _create_completion_with_fallback(
        client, messages=messages, temperature=settings.temperature,
    )
    return response.choices[0].message.content or "I don't have a response for that."
