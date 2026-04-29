"""Tests for db/queries — mock execute_query to avoid real DB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch
from db.queries import create_session, store_cv_profile, store_jd, store_gap_analysis, find_similar_cv, find_similar_jd
from db.models import CVProfile, JobDescription, GapAnalysis, SkillGap, SkillMatch, RequirementSeverity

CV = CVProfile(
    technical_skills=["Python", "SQL"],
    soft_skills=["communication"],
    years_of_experience=3,
    education=["BSc Computer Science"],
    projects=["interview-app"],
    certifications=[],
    summary="Software engineer with 3 years experience",
)

JD = JobDescription(
    role_title="Data Engineer",
    role_level="mid",
    requirements=[],
    nice_to_haves=[],
    company_type="startup",
    summary="Build data pipelines",
)

GAPS = GapAnalysis(
    matching_skills=[SkillMatch(skill="Python", evidence="used in projects")],
    gaps=[SkillGap(requirement="Kubernetes", severity=RequirementSeverity.CRITICAL)],
    partial_matches=[],
    readiness_score=62,
)


def test_create_session_returns_id():
    with patch("db.queries.execute_query", return_value=[{"id": "uuid-abc"}]) as mock_q:
        result = create_session()
    assert result == "uuid-abc"
    mock_q.assert_called_once_with(
        "INSERT INTO sessions DEFAULT VALUES RETURNING id",
        fetch=True,
    )


def test_store_cv_profile_no_embedding():
    with patch("db.queries.execute_query", return_value=[{"id": "cv-uuid"}]) as mock_q:
        result = store_cv_profile("sess-1", "raw cv text", CV, None)
    assert result == "cv-uuid"
    call_args = mock_q.call_args
    sql, params = call_args[0]
    assert "INSERT INTO cv_profiles" in sql
    assert params[0] == "sess-1"
    assert params[1] == "raw cv text"
    assert json.loads(params[2])["technical_skills"] == ["Python", "SQL"]
    assert params[3] is None  # no embedding


def test_store_cv_profile_with_embedding():
    embedding = [0.1, 0.2, 0.3]
    with patch("db.queries.execute_query", return_value=[{"id": "cv-uuid"}]) as mock_q:
        store_cv_profile("sess-1", "raw cv text", CV, embedding)
    call_args = mock_q.call_args
    _, params = call_args[0]
    assert params[3] == "[0.1,0.2,0.3]"


def test_store_jd():
    with patch("db.queries.execute_query", return_value=[{"id": "jd-uuid"}]) as mock_q:
        result = store_jd("sess-1", "raw jd text", JD, None)
    assert result == "jd-uuid"
    _, params = mock_q.call_args[0]
    assert params[0] == "sess-1"
    assert params[1] == "raw jd text"
    assert json.loads(params[2])["role_title"] == "Data Engineer"


def test_store_gap_analysis():
    with patch("db.queries.execute_query", return_value=[{"id": "gap-uuid"}]) as mock_q:
        result = store_gap_analysis("sess-1", "cv-1", "jd-1", GAPS)
    assert result == "gap-uuid"
    _, params = mock_q.call_args[0]
    assert params[0] == "sess-1"
    assert params[1] == "cv-1"
    assert params[2] == "jd-1"
    assert json.loads(params[3])["readiness_score"] == 62
    assert params[4] == 62


def test_find_similar_cv_returns_results():
    mock_rows = [
        {"id": "cv-uuid-1", "session_id": "sess-1", "similarity_score": 0.92},
        {"id": "cv-uuid-2", "session_id": "sess-2", "similarity_score": 0.88},
    ]
    embedding = [0.1, 0.2, 0.3]
    with patch("db.queries.execute_query", return_value=mock_rows) as mock_q:
        result = find_similar_cv(embedding, threshold=0.85, limit=5)
    assert len(result) == 2
    assert result[0] == {"cv_id": "cv-uuid-1", "session_id": "sess-1", "similarity_score": 0.92}
    assert result[1] == {"cv_id": "cv-uuid-2", "session_id": "sess-2", "similarity_score": 0.88}
    # Verify vec literal format was used
    _, params = mock_q.call_args[0]
    assert params[0] == "[0.1,0.2,0.3]"
    assert params[1] == "[0.1,0.2,0.3]"
    assert params[2] == 0.85
    assert params[3] == 5


def test_find_similar_cv_empty_results():
    embedding = [0.5, 0.5, 0.5]
    with patch("db.queries.execute_query", return_value=[]):
        result = find_similar_cv(embedding, threshold=0.99)
    assert result == []


def test_find_similar_jd_returns_results():
    mock_rows = [
        {"id": "jd-uuid-1", "session_id": "sess-3", "similarity_score": 0.95},
    ]
    embedding = [0.4, 0.5, 0.6]
    with patch("db.queries.execute_query", return_value=mock_rows) as mock_q:
        result = find_similar_jd(embedding, threshold=0.85, limit=3)
    assert len(result) == 1
    assert result[0] == {"jd_id": "jd-uuid-1", "session_id": "sess-3", "similarity_score": 0.95}
    # Verify vec literal format was used
    _, params = mock_q.call_args[0]
    assert params[0] == "[0.4,0.5,0.6]"
    assert params[1] == "[0.4,0.5,0.6]"
    assert params[2] == 0.85
    assert params[3] == 3


def test_find_similar_jd_empty_results():
    embedding = [0.9, 0.1, 0.0]
    with patch("db.queries.execute_query", return_value=[]):
        result = find_similar_jd(embedding, threshold=0.99)
    assert result == []
