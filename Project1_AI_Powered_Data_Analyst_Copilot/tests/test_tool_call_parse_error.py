from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from groq import APIError

from app.core import llm_engine


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _response_with_tool_calls(tool_calls):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = tool_calls
    resp.choices = [choice]
    return resp


def _tool_use_failed_error():
    request = MagicMock()
    return APIError(
        message="Failed to parse tool call arguments as JSON",
        request=request,
        body={"error": {"code": "tool_use_failed"}},
    )


@pytest.fixture(autouse=True)
def reset_cooldowns():
    llm_engine._rate_limited_until.clear()
    yield
    llm_engine._rate_limited_until.clear()


@pytest.fixture
def single_model(monkeypatch):
    monkeypatch.setattr(
        llm_engine, "settings",
        MagicMock(groq_model_fallback_chain=("model-a",), groq_model="model-a",
                   temperature=0.2, sandbox_timeout_seconds=5, groq_api_key="fake"),
    )


def test_tool_use_failed_raises_tool_call_parse_error(single_model):
    client = MagicMock()
    client.chat.completions.create.side_effect = _tool_use_failed_error()

    with pytest.raises(llm_engine.ToolCallParseError):
        llm_engine._create_completion_with_fallback(client, messages=[])


def test_partial_chart_progress_preserved_on_parse_failure(single_model, monkeypatch):
    """
    Reproduces the reported bug: several build_chart calls succeed, then a
    later response fails to parse. The already-built charts must NOT be
    discarded — they should come back in the ChatTurnResult.
    """
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 200]})

    valid_chart_call = _tool_call(
        "build_chart",
        '{"chart_type": "bar", "x": "region", "y": "revenue"}',
    )
    first_response = _response_with_tool_calls([valid_chart_call])

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        first_response,          # iteration 1: succeeds, builds one chart
        _tool_use_failed_error(),  # iteration 2: model chokes
    ]

    with patch.object(llm_engine, "_client", return_value=client):
        result = llm_engine.run_chat_turn(
            conversation_history=[],
            user_message="generate all 30 charts",
            df=df,
            profile_summary="region, revenue",
        )

    assert len(result.charts) == 1, "the chart built before the failure must be preserved"
    assert "1 chart" in result.reply_text
    assert "continue" in result.reply_text.lower()


def test_parse_failure_with_zero_progress_gives_actionable_message(single_model):
    df = pd.DataFrame({"region": ["North"], "revenue": [100]})
    client = MagicMock()
    client.chat.completions.create.side_effect = _tool_use_failed_error()

    with patch.object(llm_engine, "_client", return_value=client):
        result = llm_engine.run_chat_turn(
            conversation_history=[],
            user_message="generate all 30 charts",
            df=df,
            profile_summary="region, revenue",
        )

    assert result.charts == []
    assert "smaller batch" in result.reply_text.lower()


def test_partial_progress_preserved_for_any_llm_engine_error_not_just_parse(single_model):
    """
    Generalization check: the earlier fix only preserved progress for
    ToolCallParseError specifically. It must also apply to OTHER mid-batch
    failures — e.g. every model in the fallback chain running out of quota
    partway through a large batch — since losing 20 already-built charts to
    a token-limit error is just as bad as losing them to a parse error.
    """
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 200]})

    valid_chart_call = _tool_call(
        "build_chart",
        '{"chart_type": "bar", "x": "region", "y": "revenue"}',
    )
    first_response = _response_with_tool_calls([valid_chart_call])

    request = MagicMock()
    quota_error = APIError(
        message="Error code: 413 - tokens per minute (TPM) exceeded",
        request=request,
        body={"error": {"code": "rate_limit_exceeded", "type": "tokens"}},
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [first_response, quota_error]

    with patch.object(llm_engine, "_client", return_value=client):
        result = llm_engine.run_chat_turn(
            conversation_history=[],
            user_message="generate all 30 charts",
            df=df,
            profile_summary="region, revenue",
        )

    assert len(result.charts) == 1, "chart built before the quota error must be preserved"
    assert "continue" in result.reply_text.lower()
    """
    Regression for "only 9/30 charts generated": the model needs an explicit,
    reliable progress count in its own tool results to know whether to keep
    going or stop. Verify each build_chart tool message carries the running
    total, not just a generic success string.
    """
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 200]})

    call1 = _tool_call("build_chart", '{"chart_type": "bar", "x": "region", "y": "revenue"}', "c1")
    call2 = _tool_call("build_chart", '{"chart_type": "pie", "x": "region", "y": "revenue"}', "c2")
    first_response = _response_with_tool_calls([call1, call2])
    final_response = MagicMock()
    final_choice = MagicMock()
    final_choice.message.content = "Done."
    final_choice.message.tool_calls = None
    final_response.choices = [final_choice]

    client = MagicMock()
    client.chat.completions.create.side_effect = [first_response, final_response]

    with patch.object(llm_engine, "_client", return_value=client):
        llm_engine.run_chat_turn(
            conversation_history=[], user_message="give me 2 charts",
            df=df, profile_summary="region, revenue",
        )

    # Inspect the messages sent on the SECOND completions.create call —
    # it must contain both tool results with progressive counts (1, then 2).
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "1 chart(s) created so far" in tool_messages[0]["content"]
    assert "2 chart(s) created so far" in tool_messages[1]["content"]
