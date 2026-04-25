"""Pydantic models for structured data throughout the app."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# --- CV Extraction ---

class CVProfile(BaseModel):
    """Structured extraction from a CV."""
    technical_skills: list[str] = Field(default_factory=list, description="Technical skills found in the CV")
    soft_skills: list[str] = Field(default_factory=list, description="Soft skills found in the CV")
    years_of_experience: Optional[int] = Field(None, description="Total years of professional experience")
    education: list[str] = Field(default_factory=list, description="Education entries")
    projects: list[str] = Field(default_factory=list, description="Notable projects")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    summary: str = Field("", description="Brief professional summary")


# --- Job Description Extraction ---

class RequirementSeverity(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    NICE_TO_HAVE = "nice_to_have"


class JobRequirement(BaseModel):
    skill: str
    severity: RequirementSeverity


class JobDescription(BaseModel):
    """Structured extraction from a job description."""
    role_title: str = Field("", description="Job title")
    role_level: str = Field("", description="Seniority level (junior, mid, senior, lead)")
    requirements: list[JobRequirement] = Field(default_factory=list, description="Required skills with severity")
    nice_to_haves: list[str] = Field(default_factory=list, description="Nice-to-have skills")
    company_type: str = Field("", description="Type of company (startup, enterprise, agency)")
    summary: str = Field("", description="Brief role summary")


# --- Gap Analysis ---

class SkillMatch(BaseModel):
    skill: str
    evidence: str


class SkillGap(BaseModel):
    requirement: str
    severity: RequirementSeverity


class PartialMatch(BaseModel):
    skill: str
    has: str
    needs: str


class GapAnalysis(BaseModel):
    """Result of comparing CV against job description."""
    matching_skills: list[SkillMatch] = Field(default_factory=list)
    gaps: list[SkillGap] = Field(default_factory=list)
    partial_matches: list[PartialMatch] = Field(default_factory=list)
    readiness_score: int = Field(0, ge=0, le=100, description="Overall readiness 0-100")


# --- Evaluation ---

class AnswerEvaluation(BaseModel):
    question: str
    answer_quality: int = Field(ge=1, le=10)
    strengths: str
    weaknesses: str
    suggested_improvement: str


class SessionEvaluation(BaseModel):
    """LLM-as-Judge evaluation of an interview session."""
    answer_evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    overall_score: int = Field(0, ge=0, le=100)
    overall_feedback: str = ""
    areas_to_study: list[str] = Field(default_factory=list)
