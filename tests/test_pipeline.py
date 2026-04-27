"""Tests for parsers/pipeline — mock LLM calls."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch, MagicMock
from parsers.pipeline import (
    parse_cv, parse_jd,
    generate_embeddings, run_gap_analysis, generate_questions,
)
from db.models import (
    CVProfile, JobDescription, GapAnalysis,
    SkillGap, SkillMatch, PartialMatch, RequirementSeverity,
)


CV_JSON = {
    "technical_skills": ["Python", "SQL", "Docker"],
    "soft_skills": ["communication", "teamwork"],
    "years_of_experience": 4,
    "education": ["BSc Computer Science"],
    "projects": ["data pipeline", "api server"],
    "certifications": [],
    "summary": "Backend engineer with 4 years experience",
}

JD_JSON = {
    "role_title": "Data Engineer",
    "role_level": "mid",
    "requirements": [
        {"skill": "Python", "severity": "critical"},
        {"skill": "Kubernetes", "severity": "important"},
    ],
    "nice_to_haves": ["Spark"],
    "company_type": "startup",
    "summary": "Build scalable data pipelines",
}

GAPS = GapAnalysis(
    matching_skills=[SkillMatch(skill="Python", evidence="used in all projects")],
    gaps=[
        SkillGap(requirement="Kubernetes", severity=RequirementSeverity.CRITICAL),
        SkillGap(requirement="Spark", severity=RequirementSeverity.IMPORTANT),
    ],
    partial_matches=[PartialMatch(skill="SQL", has="basic queries", needs="window functions")],
    readiness_score=58,
)

GAP_JSON = {
    "matching_skills": [{"skill": "Python", "evidence": "used in all projects"}],
    "gaps": [
        {"requirement": "Kubernetes", "severity": "critical"},
        {"requirement": "Spark", "severity": "important"},
    ],
    "partial_matches": [{"skill": "SQL", "has": "basic queries", "needs": "window functions"}],
    "readiness_score": 58,
}

CV_PROFILE = CVProfile(**CV_JSON)
JD_PROFILE = JobDescription(**JD_JSON)


def _mock_chat_response(content: dict) -> dict:
    return {"content": json.dumps(content), "model": "gpt-5-mini", "usage": {}}


# --- parse_cv ---

def test_parse_cv_returns_cv_profile():
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(CV_JSON)):
        result = parse_cv("some cv text")
    assert isinstance(result, CVProfile)
    assert result.technical_skills == ["Python", "SQL", "Docker"]
    assert result.years_of_experience == 4


def test_parse_cv_uses_parse_model():
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(CV_JSON)) as mock_c:
        parse_cv("cv text")
    call_kwargs = mock_c.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-5-mini"
    assert call_kwargs["max_tokens"] == 2048


def test_parse_cv_raises_on_invalid_json():
    with patch("parsers.pipeline.chat", return_value={"content": "not json", "model": "", "usage": {}}):
        try:
            parse_cv("cv text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "CV parse failed" in str(e)


# --- parse_jd ---

def test_parse_jd_returns_job_description():
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(JD_JSON)):
        result = parse_jd("some jd text")
    assert isinstance(result, JobDescription)
    assert result.role_title == "Data Engineer"
    assert result.role_level == "mid"
    assert len(result.requirements) == 2


def test_parse_jd_raises_on_invalid_json():
    with patch("parsers.pipeline.chat", return_value={"content": "{broken", "model": "", "usage": {}}):
        try:
            parse_jd("jd text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "JD parse failed" in str(e)


# --- generate_embeddings ---

def test_generate_embeddings_returns_list():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 256)]
    )
    with patch("parsers.pipeline.get_ollama_client", return_value=mock_client):
        result = generate_embeddings("some text")
    assert isinstance(result, list)
    assert len(result) == 256


def test_generate_embeddings_returns_none_on_error():
    with patch("parsers.pipeline.get_ollama_client", side_effect=Exception("Ollama down")):
        result = generate_embeddings("some text")
    assert result is None


# --- run_gap_analysis ---

def test_run_gap_analysis_returns_gap_analysis():
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(GAP_JSON)):
        result = run_gap_analysis(CV_PROFILE, JD_PROFILE)
    assert isinstance(result, GapAnalysis)
    assert result.readiness_score == 58
    assert len(result.gaps) == 2
    assert result.gaps[0].requirement == "Kubernetes"


def test_run_gap_analysis_raises_on_bad_json():
    with patch("parsers.pipeline.chat", return_value={"content": "bad", "model": "", "usage": {}}):
        try:
            run_gap_analysis(CV_PROFILE, JD_PROFILE)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Gap analysis failed" in str(e)


# --- generate_questions ---

def test_generate_questions_returns_list_of_strings():
    q_json = {"questions": [
        "How would you deploy to Kubernetes?",
        "Describe your Spark experience.",
        "How do you write SQL window functions?",
        "Tell me about a data pipeline you built.",
        "How do you handle pipeline failures?",
    ]}
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(q_json)):
        result = generate_questions(GAPS, "medium", n=5)
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(q, str) for q in result)


def test_generate_questions_uses_parse_model():
    q_json = {"questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]}
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(q_json)) as mock_c:
        generate_questions(GAPS, "medium")
    call_kwargs = mock_c.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-5-mini"
    assert "response_format" not in call_kwargs


def test_generate_questions_gap_list_uses_severity_value():
    """Verify enum .value is used so prompt gets 'critical' not 'RequirementSeverity.CRITICAL'."""
    q_json = {"questions": ["Q1"]}
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(q_json)) as mock_c:
        generate_questions(GAPS, "medium", n=1)
    user_content = mock_c.call_args[1]["messages"][0]["content"]
    assert "RequirementSeverity" not in user_content
    assert "critical" in user_content


def test_generate_questions_raises_on_bad_response():
    with patch("parsers.pipeline.chat", return_value={"content": "not json", "model": "", "usage": {}}):
        try:
            generate_questions(GAPS, "medium")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Question generation failed" in str(e)


def test_generate_questions_raises_on_missing_key():
    """LLM returns valid JSON but without 'questions' key."""
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response({"answers": []})):
        try:
            generate_questions(GAPS, "medium")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Question generation failed" in str(e)
            assert "questions" in str(e).lower()


# --- run_pipeline (orchestrator) ---

from parsers.pipeline import run_pipeline, PipelineResult


def test_run_pipeline_returns_pipeline_result():
    with patch("parsers.pipeline.parse_cv", return_value=CV_PROFILE), \
         patch("parsers.pipeline.parse_jd", return_value=JD_PROFILE), \
         patch("parsers.pipeline.generate_embeddings", return_value=None), \
         patch("parsers.pipeline.run_gap_analysis", return_value=GAPS), \
         patch("parsers.pipeline.generate_questions", return_value=["Q1", "Q2", "Q3", "Q4", "Q5"]), \
         patch("parsers.pipeline.create_session", return_value="sess-uuid"), \
         patch("parsers.pipeline.store_cv_profile", return_value="cv-uuid"), \
         patch("parsers.pipeline.store_jd", return_value="jd-uuid"), \
         patch("parsers.pipeline.store_gap_analysis", return_value="gap-uuid"):
        result = run_pipeline("cv text", "jd text", "medium")
    assert isinstance(result, PipelineResult)
    assert result.session_id == "sess-uuid"
    assert result.cv == CV_PROFILE
    assert result.jd == JD_PROFILE
    assert result.gaps == GAPS
    assert result.questions == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_run_pipeline_propagates_cv_parse_error():
    with patch("parsers.pipeline.parse_cv", side_effect=ValueError("CV parse failed: bad json")):
        try:
            run_pipeline("bad cv", "jd text", "medium")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "CV parse failed" in str(e)


def test_run_pipeline_calls_db_in_order():
    call_order = []
    with patch("parsers.pipeline.parse_cv", return_value=CV_PROFILE), \
         patch("parsers.pipeline.parse_jd", return_value=JD_PROFILE), \
         patch("parsers.pipeline.generate_embeddings", return_value=None) as mock_embed, \
         patch("parsers.pipeline.run_gap_analysis", return_value=GAPS), \
         patch("parsers.pipeline.generate_questions", return_value=["Q1"]), \
         patch("parsers.pipeline.create_session", side_effect=lambda: call_order.append("session") or "s"), \
         patch("parsers.pipeline.store_cv_profile", side_effect=lambda *a: call_order.append("cv") or "c"), \
         patch("parsers.pipeline.store_jd", side_effect=lambda *a: call_order.append("jd") or "j"), \
         patch("parsers.pipeline.store_gap_analysis", side_effect=lambda *a: call_order.append("gap") or "g"):
        run_pipeline("cv", "jd", "easy")
    assert call_order == ["session", "cv", "jd", "gap"]
    assert mock_embed.call_count == 2
