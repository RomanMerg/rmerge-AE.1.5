# Session 2: Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the parsing pipeline so clicking "Analyze CV & JD" calls four LLM functions, persists results to PostgreSQL, and populates dynamic interview questions in the UI.

**Architecture:** `parsers/pipeline.py` runs four sequential LLM calls (parse CV → parse JD → gap analysis → generate questions) plus silent Ollama embedding generation, then persists results via `db/queries.py`. The Confirm button in `app.py` triggers the pipeline, stores the result in `gr.State`, updates the Generated Questions accordion, and upgrades `respond()` to use structured data instead of raw text.

**Tech Stack:** Python, Gradio 4.x, openai SDK (OpenRouter gpt-5-mini for parsing), psycopg2, pgvector string literals, Pydantic v2, pytest + unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `db/queries.py` | **Create** | CRUD: create_session, store_cv_profile, store_jd, store_gap_analysis |
| `parsers/pipeline.py` | **Create** | parse_cv, parse_jd, generate_embeddings, run_gap_analysis, generate_questions, run_pipeline, PipelineResult |
| `tests/test_queries.py` | **Create** | Unit tests for db/queries (mock execute_query) |
| `tests/test_pipeline.py` | **Create** | Unit tests for pipeline functions (mock chat + get_ollama_client) |
| `app.py` | **Modify** | Confirm button, gap summary panel, dynamic questions accordion, pipeline_state, updated respond() |

---

## Task 1: db/queries.py — Session and profile CRUD

**Files:**
- Create: `db/queries.py`
- Create: `tests/test_queries.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_queries.py`:

```python
"""Tests for db/queries — mock execute_query to avoid real DB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch
from db.queries import create_session, store_cv_profile, store_jd, store_gap_analysis
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd C:\Users\markm\Documents\Claude\Projects\Turing\rmerge-AE.1.5-main
python -m pytest tests/test_queries.py -v
```

Expected: `ModuleNotFoundError: No module named 'db.queries'`

- [ ] **Step 3: Create db/queries.py**

```python
"""CRUD operations for session data."""

import json
import logging
from db.connection import execute_query
from db.models import CVProfile, JobDescription, GapAnalysis

logger = logging.getLogger(__name__)


def _vec_literal(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    return "[" + ",".join(str(v) for v in embedding) + "]"


def create_session() -> str:
    rows = execute_query(
        "INSERT INTO sessions DEFAULT VALUES RETURNING id",
        fetch=True,
    )
    return str(rows[0]["id"])


def store_cv_profile(
    session_id: str,
    raw_text: str,
    cv: CVProfile,
    embedding: list[float] | None,
) -> str:
    rows = execute_query(
        """INSERT INTO cv_profiles (session_id, raw_text, parsed_data, embedding)
           VALUES (%s, %s, %s, %s::vector)
           RETURNING id""",
        (session_id, raw_text, json.dumps(cv.model_dump()), _vec_literal(embedding)),
        fetch=True,
    )
    return str(rows[0]["id"])


def store_jd(
    session_id: str,
    raw_text: str,
    jd: JobDescription,
    embedding: list[float] | None,
) -> str:
    rows = execute_query(
        """INSERT INTO job_descriptions (session_id, raw_text, parsed_data, embedding)
           VALUES (%s, %s, %s, %s::vector)
           RETURNING id""",
        (session_id, raw_text, json.dumps(jd.model_dump()), _vec_literal(embedding)),
        fetch=True,
    )
    return str(rows[0]["id"])


def store_gap_analysis(
    session_id: str,
    cv_id: str,
    jd_id: str,
    gaps: GapAnalysis,
) -> str:
    rows = execute_query(
        """INSERT INTO gap_analyses (session_id, cv_id, jd_id, analysis, readiness_score)
           VALUES (%s, %s, %s, %s, %s)
           RETURNING id""",
        (session_id, cv_id, jd_id, json.dumps(gaps.model_dump()), gaps.readiness_score),
        fetch=True,
    )
    return str(rows[0]["id"])
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_queries.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add db/queries.py tests/test_queries.py
git commit -m "feat: add db/queries CRUD for session, cv_profile, jd, gap_analysis"
```

---

## Task 2: Pipeline — CV and JD parsing functions

**Files:**
- Create: `parsers/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for parse_cv and parse_jd**

Create `tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_pipeline.py::test_parse_cv_returns_cv_profile tests/test_pipeline.py::test_parse_jd_returns_job_description -v
```

Expected: `ModuleNotFoundError: No module named 'parsers.pipeline'`

- [ ] **Step 3: Create parsers/pipeline.py with parse_cv and parse_jd**

```python
"""Parsing pipeline: CV → JD → embeddings → gap analysis → questions → DB."""

import json
import logging
from dataclasses import dataclass

from llm.router import chat, get_ollama_client
from llm.prompts import CV_EXTRACTION, JD_EXTRACTION, GAP_ANALYSIS, QUESTION_GENERATION
from db.models import CVProfile, JobDescription, GapAnalysis
from db.queries import create_session, store_cv_profile, store_jd, store_gap_analysis
from config import MODELS, PROVIDER_OPENROUTER

logger = logging.getLogger(__name__)


def parse_cv(cv_text: str) -> CVProfile:
    result = chat(
        messages=[
            {"role": "system", "content": CV_EXTRACTION},
            {"role": "user", "content": cv_text},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        data = json.loads(result["content"])
        cv = CVProfile(**data)
    except Exception as e:
        raise ValueError(f"CV parse failed: {e}") from e
    logger.info(
        "[pipeline] CV parsed: %d skills, %sy exp, %d projects",
        len(cv.technical_skills),
        cv.years_of_experience or 0,
        len(cv.projects),
    )
    return cv


def parse_jd(jd_text: str) -> JobDescription:
    result = chat(
        messages=[
            {"role": "system", "content": JD_EXTRACTION},
            {"role": "user", "content": jd_text},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        data = json.loads(result["content"])
        jd = JobDescription(**data)
    except Exception as e:
        raise ValueError(f"JD parse failed: {e}") from e
    logger.info(
        "[pipeline] JD parsed: %s @ %s, %d requirements",
        jd.role_title,
        jd.company_type,
        len(jd.requirements),
    )
    return jd
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_pipeline.py::test_parse_cv_returns_cv_profile tests/test_pipeline.py::test_parse_cv_uses_parse_model tests/test_pipeline.py::test_parse_cv_raises_on_invalid_json tests/test_pipeline.py::test_parse_jd_returns_job_description tests/test_pipeline.py::test_parse_jd_raises_on_invalid_json -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add parsers/pipeline.py tests/test_pipeline.py
git commit -m "feat: add parse_cv and parse_jd to pipeline"
```

---

## Task 3: Pipeline — embeddings, gap analysis, question generation

**Files:**
- Modify: `parsers/pipeline.py` (add 3 functions)
- Modify: `tests/test_pipeline.py` (add tests for 3 functions)

- [ ] **Step 1: Write failing tests — append to tests/test_pipeline.py**

Add these tests at the bottom of `tests/test_pipeline.py`:

```python
from parsers.pipeline import generate_embeddings, run_gap_analysis, generate_questions
from db.models import SkillGap, SkillMatch, PartialMatch, RequirementSeverity

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


def test_generate_questions_raises_on_bad_response():
    with patch("parsers.pipeline.chat", return_value={"content": "not json", "model": "", "usage": {}}):
        try:
            generate_questions(GAPS, "medium")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Question generation failed" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_pipeline.py -k "embeddings or gap_analysis or questions" -v
```

Expected: `ImportError` — functions not yet defined in pipeline.py

- [ ] **Step 3: Add generate_embeddings, run_gap_analysis, generate_questions to parsers/pipeline.py**

Append to `parsers/pipeline.py` (after `parse_jd`):

```python
def generate_embeddings(text: str) -> list[float] | None:
    try:
        client = get_ollama_client()
        response = client.embeddings.create(
            model=MODELS["embed"],
            input=text[:8000],
        )
        embedding = response.data[0].embedding
        logger.info("[pipeline] Embedding: %d dims", len(embedding))
        return embedding
    except Exception as e:
        logger.warning("[pipeline] Embedding unavailable: %s", e)
        return None


def run_gap_analysis(cv: CVProfile, jd: JobDescription) -> GapAnalysis:
    context = json.dumps({"cv": cv.model_dump(), "job_description": jd.model_dump()})
    result = chat(
        messages=[
            {"role": "system", "content": GAP_ANALYSIS},
            {"role": "user", "content": f"Analyze this CV and job description:\n{context}"},
        ],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        response_format={"type": "json_object"},
        max_tokens=2048,
    )
    try:
        data = json.loads(result["content"])
        gaps = GapAnalysis(**data)
    except Exception as e:
        raise ValueError(f"Gap analysis failed: {e}") from e
    logger.info(
        "[pipeline] Gap analysis: score=%d, %d gaps (%d critical), %d matches",
        gaps.readiness_score,
        len(gaps.gaps),
        sum(1 for g in gaps.gaps if g.severity == RequirementSeverity.CRITICAL),
        len(gaps.matching_skills),
    )
    return gaps


def generate_questions(gaps: GapAnalysis, difficulty: str, n: int = 5) -> list[str]:
    gap_list = "\n".join(
        f"- {g.requirement} (severity: {g.severity})" for g in gaps.gaps[:10]
    ) or "General technical and behavioural skills"
    user_content = (
        QUESTION_GENERATION.format(num_questions=n, difficulty=difficulty, gaps=gap_list)
        + '\n\nReturn ONLY a JSON object with key "questions" containing an array of strings.'
    )
    result = chat(
        messages=[{"role": "user", "content": user_content}],
        model=MODELS["parse"],
        provider=PROVIDER_OPENROUTER,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        data = json.loads(result["content"])
        questions = [str(q) for q in data["questions"][:n]]
        if not questions:
            raise ValueError("Empty questions list")
    except Exception as e:
        raise ValueError(f"Question generation failed: {e}") from e
    logger.info("[pipeline] Questions: %d generated for difficulty=%s", len(questions), difficulty)
    return questions
```

Also add this import at the top of `parsers/pipeline.py` (it's needed for the CRITICAL severity check):

```python
from db.models import CVProfile, JobDescription, GapAnalysis, RequirementSeverity
```

Replace the existing models import line with this one.

- [ ] **Step 4: Run all pipeline tests**

```
python -m pytest tests/test_pipeline.py -v
```

Expected: all 11 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add parsers/pipeline.py tests/test_pipeline.py
git commit -m "feat: add generate_embeddings, run_gap_analysis, generate_questions to pipeline"
```

---

## Task 4: Pipeline — run_pipeline orchestrator

**Files:**
- Modify: `parsers/pipeline.py` (add PipelineResult + run_pipeline)
- Modify: `tests/test_pipeline.py` (add orchestrator tests)

- [ ] **Step 1: Write failing test — append to tests/test_pipeline.py**

```python
from parsers.pipeline import run_pipeline, PipelineResult


def test_run_pipeline_returns_pipeline_result():
    q_json = {"questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]}
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
         patch("parsers.pipeline.generate_embeddings", return_value=None), \
         patch("parsers.pipeline.run_gap_analysis", return_value=GAPS), \
         patch("parsers.pipeline.generate_questions", return_value=["Q1"]), \
         patch("parsers.pipeline.create_session", side_effect=lambda: call_order.append("session") or "s"), \
         patch("parsers.pipeline.store_cv_profile", side_effect=lambda *a: call_order.append("cv") or "c"), \
         patch("parsers.pipeline.store_jd", side_effect=lambda *a: call_order.append("jd") or "j"), \
         patch("parsers.pipeline.store_gap_analysis", side_effect=lambda *a: call_order.append("gap") or "g"):
        run_pipeline("cv", "jd", "easy")
    assert call_order == ["session", "cv", "jd", "gap"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_pipeline.py -k "run_pipeline" -v
```

Expected: `ImportError` — PipelineResult and run_pipeline not defined

- [ ] **Step 3: Add PipelineResult dataclass and run_pipeline to parsers/pipeline.py**

Append to `parsers/pipeline.py` (after `generate_questions`):

```python
@dataclass
class PipelineResult:
    cv: CVProfile
    jd: JobDescription
    gaps: GapAnalysis
    questions: list[str]
    session_id: str


def run_pipeline(cv_text: str, jd_text: str, difficulty: str = "medium") -> PipelineResult:
    cv = parse_cv(cv_text)
    jd = parse_jd(jd_text)

    cv_embedding = generate_embeddings(cv_text)
    jd_embedding = generate_embeddings(jd_text)

    session_id = create_session()
    cv_id = store_cv_profile(session_id, cv_text, cv, cv_embedding)
    jd_id = store_jd(session_id, jd_text, jd, jd_embedding)

    gaps = run_gap_analysis(cv, jd)
    store_gap_analysis(session_id, cv_id, jd_id, gaps)

    questions = generate_questions(gaps, difficulty)
    return PipelineResult(cv=cv, jd=jd, gaps=gaps, questions=questions, session_id=session_id)
```

- [ ] **Step 4: Run all pipeline + queries tests**

```
python -m pytest tests/test_pipeline.py tests/test_queries.py -v
```

Expected: all 16 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add parsers/pipeline.py tests/test_pipeline.py
git commit -m "feat: add PipelineResult dataclass and run_pipeline orchestrator"
```

---

## Task 5: app.py — Confirm button, gap summary, pipeline_state

**Files:**
- Modify: `app.py`
- Modify: `tests/test_settings_handlers.py` (add handle_confirm tests)

- [ ] **Step 1: Write failing tests — append to tests/test_settings_handlers.py**

```python
# --- Task 5: handle_confirm logic ---

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from db.models import CVProfile, JobDescription, GapAnalysis, SkillGap, RequirementSeverity, SkillMatch
from parsers.pipeline import PipelineResult

_CV = CVProfile(technical_skills=["Python"], soft_skills=[], years_of_experience=2,
                education=[], projects=[], certifications=[], summary="Dev")
_JD = JobDescription(role_title="Engineer", role_level="mid", requirements=[],
                     nice_to_haves=[], company_type="startup", summary="Build things")
_GAPS = GapAnalysis(
    matching_skills=[SkillMatch(skill="Python", evidence="projects")],
    gaps=[SkillGap(requirement="Kubernetes", severity=RequirementSeverity.CRITICAL)],
    partial_matches=[],
    readiness_score=70,
)
_RESULT = PipelineResult(cv=_CV, jd=_JD, gaps=_GAPS, questions=["Q1", "Q2"], session_id="s1")


def _run_handle_confirm(cv_text, jd_text, difficulty):
    """Import and drive the generator to completion, return list of yielded values."""
    from app import handle_confirm
    return list(handle_confirm(cv_text, jd_text, difficulty))


def test_handle_confirm_empty_cv_yields_error():
    yields = _run_handle_confirm("", "some jd", "medium")
    first_status = yields[0][0]
    assert "Error" in first_status and "CV" in first_status


def test_handle_confirm_empty_jd_yields_error():
    yields = _run_handle_confirm("some cv", "", "medium")
    first_status = yields[0][0]
    assert "Error" in first_status and "description" in first_status.lower()


def test_handle_confirm_success_yields_status_and_result():
    with patch("app.run_pipeline", return_value=_RESULT):
        yields = _run_handle_confirm("cv text", "jd text", "medium")
    final = yields[-1]
    status = final[0]
    result = final[1]
    assert "✓" in status
    assert "70" in status
    assert isinstance(result, PipelineResult)


def test_handle_confirm_pipeline_error_yields_error_status():
    with patch("app.run_pipeline", side_effect=ValueError("CV parse failed: bad json")):
        yields = _run_handle_confirm("cv text", "jd text", "medium")
    final_status = yields[-1][0]
    assert "Error" in final_status
    assert "CV parse failed" in final_status
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_settings_handlers.py -k "handle_confirm" -v
```

Expected: `ImportError` — `handle_confirm` not in app

- [ ] **Step 3: Add imports, handle_confirm, and new UI components to app.py**

**3a. Add import at top of app.py** (after the existing imports block):

```python
from parsers.pipeline import run_pipeline, PipelineResult
```

**3b. Add handle_confirm function** (after the existing `check_status` function, before `_model_choices_for_provider`):

```python
def handle_confirm(cv_text, jd_text, difficulty):
    """Run the parsing pipeline when user clicks Analyze CV & JD."""
    if not cv_text or not cv_text.strip():
        yield "Error: CV cannot be empty", None, *[""] * 5, ""
        return
    if not jd_text or not jd_text.strip():
        yield "Error: Job description cannot be empty", None, *[""] * 5, ""
        return
    yield "Analyzing CV and JD...", None, *[""] * 5, ""
    try:
        result = run_pipeline(cv_text, jd_text, difficulty)
        n_gaps = len(result.gaps.gaps)
        n_crit = sum(1 for g in result.gaps.gaps if g.severity.value == "critical")
        status = f"✓ Ready — readiness score {result.gaps.readiness_score}/100, {n_gaps} gaps ({n_crit} critical)"
        gap_summary = (
            f"**Role:** {result.jd.role_title} ({result.jd.role_level})  \n"
            f"**Readiness:** {result.gaps.readiness_score}/100  \n"
            f"**Key gaps:** {', '.join(g.requirement for g in result.gaps.gaps[:3]) or 'none'}"
        )
        questions_padded = (result.questions + [""] * 5)[:5]
        yield status, result, *questions_padded, gap_summary
    except ValueError as e:
        yield f"Error: {e}", None, *[""] * 5, ""
    except Exception as e:
        yield f"Error ({type(e).__name__}): {e}", None, *[""] * 5, ""
```

**3c. In create_app(), add components and wire them up.**

In the CV/JD accordion area, after the `jd_text` accordion block and before the `Settings` accordion, insert:

```python
                confirm_btn = gr.Button("Analyze CV & JD", variant="primary")
                confirm_status = gr.Markdown("")
                gap_summary_md = gr.Markdown("", visible=False)
```

Below the chat column's `gr.ChatInterface` block (after the closing of Input Templates accordion), add:

```python
                with gr.Accordion("Generated Questions", open=False) as gen_q_accordion:
                    gr.Markdown("*Run 'Analyze CV & JD' to generate personalized questions.*")
                    gen_q_tbs = []
                    for i in range(5):
                        with gr.Row():
                            tb = gr.Textbox(
                                value="",
                                label=f"Question {i + 1}",
                                interactive=False,
                                lines=2,
                                placeholder="Will appear after analysis...",
                            )
                            use_btn = gr.Button("↑ Use", size="sm", min_width=60)
                            use_btn.click(fn=lambda v: v, inputs=[tb], outputs=[chat_input])
                            gen_q_tbs.append(tb)
```

Then wire the confirm button (after declaring all components, before `return app`):

```python
                pipeline_state = gr.State(None)
                confirm_btn.click(
                    fn=handle_confirm,
                    inputs=[cv_text, jd_text, difficulty],
                    outputs=[confirm_status, pipeline_state] + gen_q_tbs + [gap_summary_md],
                )
```

- [ ] **Step 4: Run all tests**

```
python -m pytest tests/ -v
```

Expected: all tests PASSED (the new handle_confirm tests + all existing tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_settings_handlers.py
git commit -m "feat: add Confirm button, handle_confirm pipeline trigger, gap summary, generated questions accordion"
```

---

## Task 6: app.py — respond() uses structured pipeline_result

**Files:**
- Modify: `app.py` (respond signature + system prompt logic + pipeline_state in additional_inputs)
- Modify: `tests/test_settings_handlers.py` (add system prompt tests)

- [ ] **Step 1: Write failing tests — append to tests/test_settings_handlers.py**

```python
# --- Task 6: respond() system prompt with pipeline_result ---

def _build_system_parts(cv_text, jd_text, persona, difficulty, pipeline_result):
    """Extracted from respond() for unit testing."""
    parts = [
        "You are a " + persona + " technical interviewer.",
        "Difficulty level: " + difficulty + ".",
    ]
    if pipeline_result is not None:
        cv = pipeline_result.cv
        jd = pipeline_result.jd
        gaps = pipeline_result.gaps
        parts.append("\nCandidate skills: " + ", ".join(cv.technical_skills[:15]))
        if cv.years_of_experience:
            parts.append(f"Years of experience: {cv.years_of_experience}")
        parts.append(f"\nTarget role: {jd.role_title} ({jd.role_level}) at {jd.company_type} company")
        parts.append(f"\nRole summary: {jd.summary}")
        if gaps.gaps:
            parts.append("Key skill gaps to probe: " + ", ".join(g.requirement for g in gaps.gaps[:5]))
        parts.append(f"Readiness score: {gaps.readiness_score}/100")
    else:
        if cv_text and cv_text.strip():
            parts.append("\nCandidate CV:\n" + cv_text[:2000])
        if jd_text and jd_text.strip():
            parts.append("\nJob description:\n" + jd_text[:2000])
    parts.append(
        "\nAsk interview questions one at a time. "
        "After the candidate answers, provide brief feedback "
        "and ask the next question. Stay in character."
    )
    return parts


def test_system_prompt_uses_structured_data_when_pipeline_ran():
    parts = _build_system_parts("", "", "neutral", "medium", _RESULT)
    combined = " ".join(parts)
    assert "Python" in combined
    assert "Engineer" in combined
    assert "Kubernetes" in combined
    assert "70/100" in combined
    assert "Candidate CV:" not in combined


def test_system_prompt_falls_back_to_raw_text_without_pipeline():
    parts = _build_system_parts("My CV text here", "JD text here", "neutral", "medium", None)
    combined = " ".join(parts)
    assert "Candidate CV:" in combined
    assert "My CV text here" in combined


def test_system_prompt_raw_fallback_skips_empty_cv():
    parts = _build_system_parts("", "JD text here", "neutral", "medium", None)
    combined = " ".join(parts)
    assert "Candidate CV:" not in combined
    assert "Job description:" in combined
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_settings_handlers.py -k "system_prompt" -v
```

Expected: FAILED — `_build_system_parts` function does not exist yet

- [ ] **Step 3: Update respond() in app.py**

Replace the existing `respond()` function signature and system prompt block:

```python
def respond(message, history, cv_text, jd_text, model, provider,
            temperature, top_p, max_tokens, freq_pen, pres_pen,
            persona, difficulty, pipeline_result):
    """Handle chat messages with streaming."""
    safe, reason = check_prompt_injection(message)
    if not safe:
        yield "Input blocked: " + reason
        return

    system_parts = [
        "You are a " + persona + " technical interviewer.",
        "Difficulty level: " + difficulty + ".",
    ]
    if pipeline_result is not None:
        cv = pipeline_result.cv
        jd = pipeline_result.jd
        gaps = pipeline_result.gaps
        system_parts.append("\nCandidate skills: " + ", ".join(cv.technical_skills[:15]))
        if cv.years_of_experience:
            system_parts.append(f"Years of experience: {cv.years_of_experience}")
        system_parts.append(
            f"\nTarget role: {jd.role_title} ({jd.role_level}) at {jd.company_type} company"
        )
        system_parts.append(f"\nRole summary: {jd.summary}")
        if gaps.gaps:
            system_parts.append(
                "Key skill gaps to probe: " + ", ".join(g.requirement for g in gaps.gaps[:5])
            )
        system_parts.append(f"Readiness score: {gaps.readiness_score}/100")
    else:
        if cv_text and cv_text.strip():
            system_parts.append("\nCandidate CV:\n" + cv_text[:2000])
        if jd_text and jd_text.strip():
            system_parts.append("\nJob description:\n" + jd_text[:2000])
    system_parts.append(
        "\nAsk interview questions one at a time. "
        "After the candidate answers, provide brief feedback "
        "and ask the next question. Stay in character."
    )
    # ... rest of respond() unchanged from here
```

Also update `gr.ChatInterface` `additional_inputs` to include `pipeline_state`:

```python
                gr.ChatInterface(
                    fn=respond,
                    textbox=chat_input,
                    additional_inputs=[
                        cv_text, jd_text, model, provider,
                        temperature, top_p, max_tokens,
                        freq_pen, pres_pen, persona, difficulty,
                        pipeline_state,
                    ],
                    fill_height=True,
                )
```

Note: `pipeline_state` must be declared before `gr.ChatInterface` in `create_app()`. Move the `pipeline_state = gr.State(None)` line to just before the `gr.ChatInterface` call.

- [ ] **Step 4: Add _build_system_parts as a testable function in app.py**

To make the system prompt logic unit-testable without Gradio, extract it as a standalone function above `respond()`:

```python
def _build_system_parts(cv_text, jd_text, persona, difficulty, pipeline_result):
    parts = [
        "You are a " + persona + " technical interviewer.",
        "Difficulty level: " + difficulty + ".",
    ]
    if pipeline_result is not None:
        cv = pipeline_result.cv
        jd = pipeline_result.jd
        gaps = pipeline_result.gaps
        parts.append("\nCandidate skills: " + ", ".join(cv.technical_skills[:15]))
        if cv.years_of_experience:
            parts.append(f"Years of experience: {cv.years_of_experience}")
        parts.append(
            f"\nTarget role: {jd.role_title} ({jd.role_level}) at {jd.company_type} company"
        )
        parts.append(f"\nRole summary: {jd.summary}")
        if gaps.gaps:
            parts.append(
                "Key skill gaps to probe: " + ", ".join(g.requirement for g in gaps.gaps[:5])
            )
        parts.append(f"Readiness score: {gaps.readiness_score}/100")
    else:
        if cv_text and cv_text.strip():
            parts.append("\nCandidate CV:\n" + cv_text[:2000])
        if jd_text and jd_text.strip():
            parts.append("\nJob description:\n" + jd_text[:2000])
    parts.append(
        "\nAsk interview questions one at a time. "
        "After the candidate answers, provide brief feedback "
        "and ask the next question. Stay in character."
    )
    return parts
```

Then in `respond()`, replace the system_parts block with:

```python
    system_parts = _build_system_parts(cv_text, jd_text, persona, difficulty, pipeline_result)
```

- [ ] **Step 5: Run all tests**

```
python -m pytest tests/ -v
```

Expected: all tests PASSED

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_settings_handlers.py
git commit -m "feat: respond() uses structured pipeline data; add _build_system_parts for testability"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| parse_cv() with CV_EXTRACTION prompt | Task 2 |
| parse_jd() with JD_EXTRACTION prompt | Task 2 |
| generate_embeddings() with Ollama, NULL fallback | Task 3 |
| run_gap_analysis() with GAP_ANALYSIS prompt | Task 3 |
| generate_questions() with QUESTION_GENERATION prompt | Task 3 |
| run_pipeline() orchestrator + PipelineResult | Task 4 |
| Each function logs one-liner | Tasks 2, 3 |
| db/queries.py CRUD | Task 1 |
| DB writes inside run_pipeline | Task 4 |
| Confirm button + status label | Task 5 |
| pipeline_state (gr.State) | Task 5 |
| Dynamic Generated Questions accordion | Task 5 |
| Gap summary panel | Task 5 |
| respond() uses structured data with raw fallback | Task 6 |
| pipeline_state as additional_input to ChatInterface | Task 6 |

**Type consistency check:**
- `PipelineResult` defined in Task 4, used in Task 5 (`handle_confirm`) and Task 6 (`respond`) — consistent
- `_build_system_parts` defined and used in Task 6 — consistent
- `gen_q_tbs` is a `list[gr.Textbox]` of length 5; `handle_confirm` yields `*questions_padded` (5 items) — length matches
- `confirm_btn.click` outputs: `[confirm_status, pipeline_state] + gen_q_tbs + [gap_summary_md]` = 2 + 5 + 1 = 8; `handle_confirm` yields 8-tuples — consistent

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.
