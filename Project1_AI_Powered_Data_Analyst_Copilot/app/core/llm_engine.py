"""
Groq LLM orchestration: a tool-calling loop that lets the model call
deterministic tools (pandas/SQL/charts/anomaly detection) against the real
dataset, then narrate only the real results it gets back.

Includes automatic model fallback: if the active Groq model reports that
it's out of capacity — a 429 rate-limit error, OR a 413/"tokens per minute"
quota error (Groq returns both for different kinds of exhausted limits) —
the engine transparently switches to the next model in
`settings.groq_model_fallback_chain` and retries the SAME request — the
conversation/messages list is untouched, so no context is lost. Only these
capacity/quota signals trigger a switch; genuine errors (bad request,
connection failure, malformed tool call) are surfaced immediately instead.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from groq import Groq, APIError, APIConnectionError, RateLimitError

from app.core import prompts, tools
from app.core.config import settings

MAX_TOOL_ITERATIONS = 40

logger = logging.getLogger("data_analyst_copilot.llm_engine")

# Models that have hit a capacity/quota limit in this process are skipped on
# subsequent calls too, so we don't keep re-hitting a known-exhausted model
# every turn (avoids unnecessary duplicate requests). Cleared automatically
# once the model's quota window has likely reset — see the cooldown constant.
_RATE_LIMIT_COOLDOWN_SECONDS = 60
_rate_limited_until: dict[str, float] = {}

# Signals that indicate the model is out of capacity/quota (should trigger a
# fallback to the next model) rather than a genuine request error. Groq
# returns these under a few different HTTP statuses and error codes
# depending on which limit was hit (requests/min vs tokens/min etc).
_QUOTA_EXHAUSTED_MARKERS = (
    "rate_limit_exceeded",
    "tokens per minute",
    "requests per minute",
    "rate limit",
)
_QUOTA_EXHAUSTED_STATUS_CODES = {429, 413}


def _is_quota_exhausted_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in _QUOTA_EXHAUSTED_STATUS_CODES:
        return True
    err_text = str(exc).lower()
    return any(marker in err_text for marker in _QUOTA_EXHAUSTED_MARKERS)


class LLMEngineError(Exception):
    """Friendly, user-facing error for LLM/tool failures."""


class ToolCallParseError(LLMEngineError):
    """
    Raised when Groq returns a tool call whose arguments aren't valid JSON
    (a known failure mode when a single response tries to pack in too many
    tool calls at once, especially for reasoning models). This is distinct
    from other API errors so callers can preserve partial progress instead
    of discarding everything already completed in this turn.
    """


@dataclass
class ChatTurnResult:
    reply_text: str
    charts: list[Any] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)


def _client() -> Groq:
    if not settings.groq_api_key:
        raise LLMEngineError(
            "Groq API key is missing. Please set GROQ_API_KEY in your .env file."
        )
    return Groq(api_key=settings.groq_api_key)


def _build_system_message(profile_summary: str) -> dict:
    return {"role": "system", "content": prompts.SYSTEM_PROMPT.format(profile=profile_summary)}


def _candidate_models() -> list[str]:
    """Fallback chain with currently-cooling-down models moved to the end
    (not dropped entirely, in case the whole chain is exhausted)."""
    now = time.monotonic()
    chain = list(settings.groq_model_fallback_chain) or [settings.groq_model]
    fresh = [m for m in chain if _rate_limited_until.get(m, 0) <= now]
    cooling = [m for m in chain if _rate_limited_until.get(m, 0) > now]
    return fresh + cooling


def _create_completion_with_fallback(client: Groq, **kwargs) -> Any:
    """
    Calls client.chat.completions.create, automatically falling back to the
    next model in settings.groq_model_fallback_chain whenever the CURRENT
    model reports it's out of capacity/quota (429 rate limit, or 413 "tokens
    per minute" exceeded — Groq uses both depending on which limit is hit).
    Any other exception is raised immediately without switching models. The
    same request payload (messages, tools, etc.) is reused as-is for each
    attempt, so no conversation context is lost or duplicated.
    """
    models_to_try = _candidate_models()
    last_quota_error: Exception | None = None

    for model in models_to_try:
        try:
            response = client.chat.completions.create(model=model, **kwargs)
            logger.info("Groq request served by model: %s", model)
            return response
        except RateLimitError as e:
            logger.warning(
                "Model '%s' hit a 429 rate limit. Switching to next available model.", model
            )
            _rate_limited_until[model] = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
            last_quota_error = e
            continue
        except APIConnectionError:
            raise LLMEngineError("Could not connect to Groq. Please check your internet connection.")
        except APIError as e:
            err_text = str(e)
            if "tool_use_failed" in err_text or "Failed to parse tool call arguments" in err_text:
                raise ToolCallParseError(
                    "The model tried to pack too much into one response and produced an "
                    "invalid tool call. This can happen when asking for a large batch of "
                    "charts/analyses all at once."
                )
            if _is_quota_exhausted_error(e):
                logger.warning(
                    "Model '%s' is out of capacity/quota (%s). Switching to next available model.",
                    model, err_text[:120],
                )
                _rate_limited_until[model] = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
                last_quota_error = e
                continue
            raise LLMEngineError(f"Groq API error: {e}")

    tried = ", ".join(models_to_try)
    raise LLMEngineError(
        f"All configured Groq models are currently out of capacity/quota: {tried}. "
        f"Please wait a moment and try again."
    ) from last_quota_error


def run_chat_turn(
    conversation_history: list[dict],
    user_message: str,
    df: pd.DataFrame,
    profile_summary: str,
) -> ChatTurnResult:
    """
    Runs one user turn through the Groq tool-calling loop.
    `conversation_history` should be a list of {"role", "content"} dicts
    WITHOUT the system message (it's injected fresh each call so the profile
    stays current if the data was cleaned).
    """
    client = _client()

    messages: list[dict] = [_build_system_message(profile_summary)]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    collected_charts: list[Any] = []
    collected_tables: list[pd.DataFrame] = []
    tool_trace: list[dict] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = _create_completion_with_fallback(
                client,
                messages=messages,
                tools=tools.TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=settings.temperature,
            )
        except LLMEngineError as e:
            # Never discard progress already made this turn — whatever the
            # failure (malformed tool call, all models out of quota, a
            # connection hiccup mid-batch), if we already built some
            # charts/tables, hand those back instead of losing them.
            if collected_charts or collected_tables:
                progress_note = (
                    f"\n\nI completed {len(collected_charts)} chart(s) before running into "
                    f"an issue: {e} Ask me to \"continue with the rest\" and I'll pick up "
                    f"where I left off."
                )
                return ChatTurnResult(
                    reply_text=f"Here's what I generated so far.{progress_note}",
                    charts=collected_charts,
                    tables=collected_tables,
                    tool_trace=tool_trace,
                )
            hint = (
                " Try asking for a smaller batch at a time (e.g. \"show me 5 charts\" "
                "instead of all 30 at once)."
                if isinstance(e, ToolCallParseError)
                else ""
            )
            return ChatTurnResult(
                reply_text=f"❌ {e}{hint}",
                charts=[], tables=[], tool_trace=tool_trace,
            )

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            final_text = msg.content or "I don't have a response for that."
            return ChatTurnResult(
                reply_text=final_text,
                charts=collected_charts,
                tables=collected_tables,
                tool_trace=tool_trace,
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            text_result, raw_payload = tools.execute_tool(
                fn_name, fn_args, df, sandbox_timeout=settings.sandbox_timeout_seconds
            )

            tool_trace.append({"tool": fn_name, "arguments": fn_args, "result_preview": text_result[:500]})

            if fn_name == "build_chart" and raw_payload is not None:
                collected_charts.append(raw_payload)
                # Give the model an explicit, hard-to-miss running count in
                # its own tool results — this is what it actually "sees" for
                # tracking progress across multiple response turns, and is
                # more reliable than expecting it to recount its own history.
                chart_type_made = fn_args.get("chart_type", "chart")
                text_result += (
                    f"\n[PROGRESS] {len(collected_charts)} chart(s) created so far this turn "
                    f"(latest: {chart_type_made}). Continue with more build_chart calls "
                    f"if the requested total isn't reached yet."
                )
            elif isinstance(raw_payload, pd.DataFrame) and not raw_payload.empty:
                collected_tables.append(raw_payload)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text_result,
                }
            )

    # Safety net if the loop exceeds max iterations
    return ChatTurnResult(
        reply_text="I made several tool calls but couldn't finalize an answer. "
                    "Could you rephrase or narrow your question?",
        charts=collected_charts,
        tables=collected_tables,
        tool_trace=tool_trace,
    )


def generate_report_narrative(structured_findings: dict) -> str:
    """Generates a Markdown report narrative grounded strictly in structured
    findings already computed deterministically (profile, cleaning log, Q&A)."""
    client = _client()
    try:
        response = _create_completion_with_fallback(
            client,
            messages=[
                {"role": "system", "content": prompts.REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(structured_findings, default=str)},
            ],
            temperature=settings.temperature,
        )
    except LLMEngineError as e:
        raise LLMEngineError(f"Could not generate report narrative: {e}")

    return response.choices[0].message.content or "Report generation returned no content."
