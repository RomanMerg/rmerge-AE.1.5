# Session 2: Core Pipeline — Design Spec

**Date:** 2026-04-27  
**Status:** Approved  
**Scope:** CV/JD parsing, gap analysis, question generation, DB persistence, embeddings

---

## Goal

When the user fills in CV + JD and clicks Confirm, a pipeline runs that:
1. Parses both inputs into structured JSON via LLM
2. Generates vector embeddings for both (Ollama, fallback to NULL)
3. Runs gap analysis comparing CV skills against JD requirements
4. Generates 5 targeted interview questions from the gaps
5. Persists all results to PostgreSQL
6. Updates the UI: status label, dynamic Input Templates, gap summary in chat

---

## New Files

### `parsers/pipeline.py`

Single pipeline module with four extraction functions + one orchestrator.

**Functions:**

```
parse_cv(cv_text: str) -> CVProfile
    model: MODELS["parse"] (gpt-5-mini, OpenRouter)
    prompt: CV_EXTRACTION (zero-shot)
    format: response_format={"type": "json_object"}
    parse: json.loads() → CVProfile(**data)
    log: "[pipeline] CV parsed: {n} skills, {exp}y exp, {p} projects"

parse_jd(jd_text: str) -> JobDescription
    model: MODELS["parse"]
    prompt: JD_EXTRACTION (structured output)
    log: "[pipeline] JD parsed: {title} @ {company_type}, {n} requirements"

run_gap_analysis(cv: CVProfile, jd: JobDescription) -> GapAnalysis
    model: MODELS["parse"]
    prompt: GAP_ANALYSIS (chain-of-thought)
    context: cv and jd serialized as JSON in user message
    log: "[pipeline] Gap analysis: score={score}, {gaps} gaps ({crit} critical), {matches} matches"

generate_questions(gaps: GapAnalysis, difficulty: str, n: int = 5) -> list[str]
    model: MODELS["parse"]
    prompt: QUESTION_GENERATION (few-shot)
    log: "[pipeline] Questions: {n} generated for difficulty={difficulty}"

generate_embeddings(text: str) -> list[float] | None
    client: Ollama /v1/embeddings
    model: MODELS["embed"] (nomic-embed-text-v2-moe)
    on failure: log warning, return None (never raises)
    log: "[pipeline] Embedding: {dims} dims" or "[pipeline] Embedding: unavailable (Ollama down)"
```

**Orchestrator:**

```python
@dataclass
class PipelineResult:
    cv: CVProfile
    jd: JobDescription
    gaps: GapAnalysis
    questions: list[str]
    session_id: str          # UUID from DB

def run_pipeline(cv_text: str, jd_text: str, difficulty: str) -> PipelineResult:
    # 1. parse_cv → raises ValueError("CV parse failed: ...") on bad JSON/schema
    # 2. parse_jd → raises ValueError("JD parse failed: ...")
    # 3. generate_embeddings for both (silent fallback)
    # 4. create_session() in DB → get session_id
    # 5. store_cv_profile(session_id, cv_text, cv, cv_embedding)
    # 6. store_jd(session_id, jd_text, jd, jd_embedding)
    # 7. run_gap_analysis → raises ValueError("Gap analysis failed: ...")
    # 8. store_gap_analysis(session_id, cv_id, jd_id, gaps)
    # 9. generate_questions → raises ValueError("Question generation failed: ...")
    # 10. return PipelineResult
```

Each `ValueError` is caught in `app.py` and shown as the status label text.

---

### `db/queries.py` (new)

CRUD functions using the existing `execute_query` from `db/connection.py`.

```
create_session() -> str                              # returns UUID
store_cv_profile(session_id, raw_text, cv: CVProfile, embedding) -> str   # returns cv_id UUID
store_jd(session_id, raw_text, jd: JobDescription, embedding) -> str      # returns jd_id UUID
store_gap_analysis(session_id, cv_id, jd_id, gaps: GapAnalysis) -> str    # returns gap_id UUID
```

`parsed_data` stored as `json.dumps(model.model_dump())`. Embedding stored as list or NULL.

---

## Changes to `app.py`

### New Gradio components (in CV/JD accordion area)

```
confirm_btn   = gr.Button("Analyze CV & JD")
confirm_status = gr.Markdown("")          # "Analyzing..." / "✓ Ready" / "Error: ..."
pipeline_state = gr.State(None)           # holds PipelineResult
```

### `handle_confirm(cv_text, jd_text, difficulty)` handler

```python
def handle_confirm(cv_text, jd_text, difficulty):
    yield "Analyzing CV and JD..."        # immediate feedback
    try:
        result = run_pipeline(cv_text, jd_text, difficulty)
        yield f"✓ Ready — score {result.gaps.readiness_score}/100, {len(result.gaps.gaps)} gaps found"
        return result
    except ValueError as e:
        yield f"Error: {e}"
        return None
```

Outputs: `[confirm_status, pipeline_state]`

### Dynamic Input Templates

After confirm succeeds, the Input Templates accordion is rebuilt with the generated questions replacing the static defaults. Each question gets a "↑ Use" button wired to `chat_input` exactly as the existing static templates.

### Gap summary as first chat message

`respond()` gains a check: if `pipeline_state` is set and `history` is empty, prepend a system-authored assistant message summarising the gap analysis before the interviewer's first question:

```
"I've reviewed your CV against the {role_title} role.
Readiness score: {score}/100.
Key gaps to address: {gap1}, {gap2}, {gap3}.
Let's start the interview."
```

This is injected into the Gradio history, not sent to the LLM — it's display-only context for the user.

### `respond()` uses structured pipeline data

When `pipeline_state` is set, the system prompt in `respond()` uses the structured `CVProfile` and `JobDescription` fields instead of raw text slices:

```python
# Before (raw):
system_parts.append("\nCandidate CV:\n" + cv_text[:2000])

# After (structured, when pipeline ran):
system_parts.append("\nCandidate skills: " + ", ".join(cv.technical_skills[:15]))
system_parts.append("\nRole: " + jd.role_title + " — " + jd.summary)
system_parts.append("\nKey gaps: " + ", ".join(g.requirement for g in gaps.gaps[:5]))
```

Raw text fallback remains if user skips Confirm and goes straight to chat.

---

## Embeddings

- Generated immediately after `parse_cv` and `parse_jd`
- Ollama client reuses `get_ollama_client()` from `llm/router.py`
- Endpoint: `client.embeddings.create(model=MODELS["embed"], input=text)`
- Dimensions: 256 (Matryoshka truncation via `extra_body={"truncate": True}` if needed)
- Failure: log `[pipeline] Embedding unavailable`, store NULL — pipeline continues
- Dedup query logic deferred to Session 4

---

## Error Handling

| Stage | Error | User sees |
|-------|-------|-----------|
| CV too short / empty | `validate_cv()` pre-check | "Error: CV cannot be empty" |
| LLM returns invalid JSON | `json.JSONDecodeError` | "Error: CV parse failed: invalid JSON from LLM" |
| Pydantic validation fails | `ValidationError` | "Error: CV parse failed: missing required fields" |
| DB write fails | `psycopg2.Error` | "Error: Could not save session — check DB connection" |
| All errors | any `Exception` | "Error: {type}: {message}" |

Pipeline is all-or-nothing: if any step fails, nothing is stored and `pipeline_state` stays None.

---

## What's NOT in Session 2

- Dedup query against pgvector (Session 4)
- LLM-as-Judge evaluation (Session 4)
- Cost tracking per prompt (Session 4)
- `llm/judge.py` (Session 4)
