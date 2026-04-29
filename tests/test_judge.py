"""Tests for llm/judge.py — mock LLM calls."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch, MagicMock
from llm.judge import evaluate_session
from db.models import SessionEvaluation, AnswerEvaluation


EVAL_JSON = {
    "answer_evaluations": [
        {
            "question": "Tell me about your Python experience.",
            "answer_quality": 7,
            "strengths": "Mentioned real projects",
            "weaknesses": "No async experience discussed",
            "suggested_improvement": "Discuss asyncio and FastAPI",
        }
    ],
    "overall_score": 65,
    "overall_feedback": "Solid foundation, needs async depth.",
    "areas_to_study": ["asyncio", "FastAPI", "Kubernetes basics"],
}

HISTORY = [
    {"role": "assistant", "content": "Tell me about your Python experience."},
    {"role": "user", "content": "I've used Python for 4 years, mostly data pipelines."},
]


def _make_chat_result(data: dict):
    return {"content": json.dumps(data), "model": "openai/gpt-5", "usage": {}, "finish_reason": "stop"}


@patch("llm.judge.chat")
def test_evaluate_session_basic(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    result = evaluate_session(HISTORY, pipeline_result=None)

    assert isinstance(result, SessionEvaluation)
    assert result.overall_score == 65
    assert len(result.answer_evaluations) == 1
    assert result.answer_evaluations[0].answer_quality == 7
    assert "asyncio" in result.areas_to_study


@patch("llm.judge.chat")
def test_evaluate_session_uses_judge_model(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    evaluate_session(HISTORY, pipeline_result=None)

    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["model"] == "openai/gpt-5"
    assert call_kwargs.kwargs["max_tokens"] == 4096
    assert call_kwargs.kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


@patch("llm.judge.chat")
def test_evaluate_session_includes_context(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)

    pipeline_result = MagicMock()
    pipeline_result.jd.role_title = "Data Engineer"
    pipeline_result.jd.role_level = "mid"
    pipeline_result.gaps.readiness_score = 58
    pipeline_result.gaps.gaps = []

    evaluate_session(HISTORY, pipeline_result=pipeline_result)

    user_content = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "Data Engineer" in user_content
    assert "58/100" in user_content


@patch("llm.judge.chat")
def test_evaluate_session_transcript_format(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    evaluate_session(HISTORY, pipeline_result=None)

    user_content = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "Q: Tell me about your Python experience." in user_content
    assert "A: I've used Python for 4 years" in user_content


@patch("llm.judge.chat")
def test_evaluate_session_empty_history(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    result = evaluate_session([], pipeline_result=None)
    assert isinstance(result, SessionEvaluation)


@patch("llm.judge.chat")
def test_evaluate_session_invalid_json_raises(mock_chat):
    mock_chat.return_value = {"content": "not json at all", "model": "openai/gpt-5", "usage": {}, "finish_reason": "stop"}
    try:
        evaluate_session(HISTORY, pipeline_result=None)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "invalid JSON" in str(e)


@patch("llm.judge.chat")
def test_evaluate_session_handles_list_content(mock_chat):
    """Gradio 6.x passes content as list[{type, text}] — must not crash."""
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    history_list_content = [
        {"role": "assistant", "content": [{"type": "text", "text": "Tell me about Python."}]},
        {"role": "user", "content": [{"type": "text", "text": "I have 4 years experience."}]},
    ]
    result = evaluate_session(history_list_content, pipeline_result=None)
    assert isinstance(result, SessionEvaluation)
    user_content = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "Tell me about Python." in user_content
    assert "I have 4 years experience." in user_content


@patch("llm.judge.chat")
def test_evaluate_session_skips_system_role(mock_chat):
    mock_chat.return_value = _make_chat_result(EVAL_JSON)
    history_with_system = [
        {"role": "system", "content": "You are an interviewer."},
        *HISTORY,
    ]
    evaluate_session(history_with_system, pipeline_result=None)

    user_content = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "You are an interviewer." not in user_content
