"""Tests for parsers/pipeline — mock LLM calls."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch, MagicMock
from parsers.pipeline import parse_cv, parse_jd
from db.models import CVProfile, JobDescription


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


def _mock_chat_response(content: dict) -> dict:
    return {"content": json.dumps(content), "model": "gpt-5-mini", "usage": {}}


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
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_parse_cv_raises_on_invalid_json():
    bad_response = {"content": "not json", "model": "gpt-5-mini", "usage": {}}
    with patch("parsers.pipeline.chat", return_value=bad_response):
        try:
            parse_cv("cv text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "CV parse failed" in str(e)


def test_parse_jd_returns_job_description():
    with patch("parsers.pipeline.chat", return_value=_mock_chat_response(JD_JSON)):
        result = parse_jd("some jd text")
    assert isinstance(result, JobDescription)
    assert result.role_title == "Data Engineer"
    assert result.role_level == "mid"
    assert len(result.requirements) == 2


def test_parse_jd_raises_on_invalid_json():
    bad_response = {"content": "{broken", "model": "gpt-5-mini", "usage": {}}
    with patch("parsers.pipeline.chat", return_value=bad_response):
        try:
            parse_jd("jd text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "JD parse failed" in str(e)
