from unittest.mock import MagicMock, patch

import pytest
from groq import APIConnectionError, APIError, RateLimitError

from app.core import llm_engine


def _fake_response(text="ok"):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None
    resp.choices = [choice]
    return resp


def _rate_limit_error():
    request = MagicMock()
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {}
    return RateLimitError("rate limited", response=resp, body=None)


def _token_quota_error():
    """Reproduces Groq's 413 'tokens per minute (TPM)' quota error — a
    different exception shape than the 429 RateLimitError, but semantically
    the same thing: this model is out of capacity right now."""
    request = MagicMock()
    return APIError(
        message=(
            "Error code: 413 - Request too large for model `openai/gpt-oss-20b` "
            "on tokens per minute (TPM): Limit 8000, Requested 8182"
        ),
        request=request,
        body={"error": {"code": "rate_limit_exceeded", "type": "tokens"}},
    )


@pytest.fixture(autouse=True)
def reset_cooldowns():
    llm_engine._rate_limited_until.clear()
    yield
    llm_engine._rate_limited_until.clear()


@pytest.fixture
def fallback_chain(monkeypatch):
    chain = ("model-a", "model-b", "model-c")
    monkeypatch.setattr(llm_engine, "settings", MagicMock(groq_model_fallback_chain=chain, groq_model="model-a"))
    return chain


def test_switches_to_next_model_on_429(fallback_chain):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _fake_response("from model-b"),
    ]

    response = llm_engine._create_completion_with_fallback(client, messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "from model-b"
    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] == "model-a"
    assert calls[1].kwargs["model"] == "model-b"


def test_non_429_error_does_not_switch_models(fallback_chain):
    client = MagicMock()
    client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    with pytest.raises(llm_engine.LLMEngineError, match="Could not connect"):
        llm_engine._create_completion_with_fallback(client, messages=[])

    # Only ONE model attempted — non-429 errors must not trigger fallback.
    assert client.chat.completions.create.call_count == 1


def test_all_models_exhausted_raises_clean_error(fallback_chain):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _rate_limit_error(), _rate_limit_error(), _rate_limit_error()
    ]

    with pytest.raises(llm_engine.LLMEngineError, match="out of capacity"):
        llm_engine._create_completion_with_fallback(client, messages=[])

    assert client.chat.completions.create.call_count == 3


def test_conversation_messages_unchanged_across_fallback(fallback_chain):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _fake_response("ok"),
    ]
    original_messages = [{"role": "user", "content": "what is total revenue?"}]

    llm_engine._create_completion_with_fallback(client, messages=original_messages)

    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["messages"] == original_messages


def test_rate_limited_model_deprioritized_on_next_call(fallback_chain):
    client = MagicMock()
    # First call: model-a rate limited, model-b succeeds.
    client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _fake_response("first"),
    ]
    llm_engine._create_completion_with_fallback(client, messages=[])

    # Second call: model-a should be skipped (still cooling down) and
    # model-b tried first directly — avoiding a duplicate/wasted request to model-a.
    client.chat.completions.create.side_effect = [_fake_response("second")]
    llm_engine._create_completion_with_fallback(client, messages=[])

    second_call_model = client.chat.completions.create.call_args_list[-1].kwargs["model"]
    assert second_call_model == "model-b"


def test_switches_to_next_model_on_413_token_quota_error(fallback_chain):
    """
    Regression: Groq returns a 413 'tokens per minute (TPM)' quota error
    (not the 429 RateLimitError class) when the per-minute token budget is
    exhausted. This must ALSO trigger a model switch — it's a capacity
    limit just like a 429, just reported differently.
    """
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _token_quota_error(),
        _fake_response("from model-b"),
    ]

    response = llm_engine._create_completion_with_fallback(client, messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "from model-b"
    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["model"] == "model-a"
    assert calls[1].kwargs["model"] == "model-b"


def test_genuine_bad_request_does_not_trigger_fallback(fallback_chain):
    """A real client-error (not a capacity/quota issue) must still fail
    immediately without burning through the whole fallback chain."""
    request = MagicMock()
    genuine_error = APIError(
        message="Error code: 400 - invalid request: missing required field",
        request=request,
        body={"error": {"code": "invalid_request_error"}},
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = genuine_error

    with pytest.raises(llm_engine.LLMEngineError, match="Groq API error"):
        llm_engine._create_completion_with_fallback(client, messages=[])

    assert client.chat.completions.create.call_count == 1
