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
    session_id = str(rows[0]["id"])
    logger.debug("[db] session created: %s", session_id)
    return session_id


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
    cv_id = str(rows[0]["id"])
    logger.debug("[db] cv_profile stored: %s", cv_id)
    return cv_id


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
    jd_id = str(rows[0]["id"])
    logger.debug("[db] job_description stored: %s", jd_id)
    return jd_id


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
    gap_id = str(rows[0]["id"])
    logger.debug("[db] gap_analysis stored: %s score=%d", gap_id, gaps.readiness_score)
    return gap_id
