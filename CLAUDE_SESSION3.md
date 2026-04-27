# CLAUDE_SESSION3.md — Interview Practice App: Session 3

## Project Overview
Turing College Sprint 1: CV-aware interview practice chatbot.
**GitHub:** https://github.com/TuringCollegeSubmissions/rmerge-AE.1.5
**Branch:** `personal/dev` (merge → `main` before submission)
**Full ultraplan:** `05-project-ultraplan.md` in Obsidian vault (sprint-1-llm-fundamentals/)

Sessions 1 & 2 are **complete and working in Docker**. Session 3 adds LLM-as-judge + enhanced guards.

---

## Current Architecture (post-Session 2)

```
app.py                        # Gradio UI — ChatInterface + CV upload + JD + pipeline trigger
config.py                     # env vars, MODELS dict, MODEL_CONFIGS, provider constants
llm/
  router.py                   # chat(), chat_stream(), chat_collect(), smoke_test()
  prompts.py                  # 6 system prompts (CV_EXTRACTION, JD_EXTRACTION, GAP_ANALYSIS,
                              #   QUESTION_GENERATION, INTERVIEWER_PERSONA, SESSION_EVALUATION)
  guards.py                   # validate_cv/jd (length), check_prompt_injection (pattern-only),
                              #   classify_input() (async stub — NOT yet wired)
parsers/
  pipeline.py                 # run_pipeline() → PipelineResult; parse_cv, parse_jd,
                              #   generate_embeddings, run_gap_analysis, generate_questions
db/
  connection.py               # PostgreSQL + execute_query(); init.sql auto-run on startup
  models.py                   # CVProfile, JobDescription, GapAnalysis, SessionEvaluation,
                              #   AnswerEvaluation (Pydantic v2)
  queries.py                  # create_session, store_cv_profile, store_jd, store_gap_analysis
utils/
  cost_tracker.py             # SessionCostTracker — estimates cost per session
tests/
  test_pipeline.py            # 17 unit tests (mock parsers.pipeline.chat)
  test_queries.py             # 5 unit tests (mock execute_query)
  test_settings_handlers.py   # settings handler tests
```

### DB Tables (all exist, schema in init.sql)
- `sessions` — UUID, created_at
- `cv_profiles` — session_id, raw_text, parsed_data (JSONB), embedding (vector(256))
- `job_descriptions` — session_id, raw_text, parsed_data (JSONB), embedding (vector(256))
- `gap_analyses` — session_id, cv_id, jd_id, analysis (JSONB), readiness_score
- `chat_messages` — session_id, role, content, model_used, cost_usd *(exists in schema, not yet written to)*
- `evaluations` — session_id, evaluation (JSONB), overall_score *(exists in schema, not yet written to)*

---

## Model Routing (config.py)

```python
MODELS = {
    "guard":      "openai/gpt-5-nano",       # cheap, fast classification
    "parse":      "openai/gpt-5-mini",       # structured extraction
    "chat":       "openai/gpt-5-mini",       # interactive default
    "chat_local": "qwen3.5:9b",             # Ollama optional
    "judge":      "openai/gpt-5",           # highest capability, 1x per session
    "embed":      "nomic-embed-text-v2-moe", # Ollama, Matryoshka @ 256 dims
}
```

**Important:** `openai/gpt-5-mini` and `openai/gpt-5-nano` are **reasoning models** (o4-based).
- They ignore the `temperature` parameter
- Non-streaming calls sometimes return empty `content` when reasoning budget is exhausted
- **Fix used in pipeline:** `extra_body={"reasoning": {"effort": "low"}}` caps reasoning tokens (~128 vs ~1408 default)
- Always pass this `extra_body` kwarg to `chat()` for any structured parse call using these models

---

## Session 3 Tasks

### Task 1 — LLM-as-Judge (`llm/judge.py` + `db/queries.py` + `app.py`)

**What:** An "Evaluate Session" button that sends the full chat history to `openai/gpt-5` and gets back a structured `SessionEvaluation` with per-answer scores, overall feedback, and study areas. Result displayed in a new accordion and persisted to `evaluations` table.

**Implementation steps:**

1. **Create `llm/judge.py`** — single public function:
   ```python
   def evaluate_session(history: list[dict], pipeline_result) -> SessionEvaluation:
       """
       history: list of {"role": "user"|"assistant", "content": str}
                (the Gradio chat history list — same format as ChatInterface state)
       pipeline_result: PipelineResult | None — for context (role, readiness score)
       Returns: SessionEvaluation (already defined in db/models.py)
       """
   ```
   - Build a transcript string from history (skip system messages; include Q: / A: labels)
   - Include role context and readiness score in the prompt if pipeline_result is not None
   - Call `chat()` from `llm/router.py` with:
     - `model=MODELS["judge"]` (`openai/gpt-5`)
     - `provider=PROVIDER_OPENROUTER`
     - `max_tokens=4096`
     - `extra_body={"reasoning": {"effort": "low"}}` — judge is also a reasoning model
   - Parse response with `_parse_llm_json()` — **do NOT copy this helper; import from `parsers.pipeline`** or duplicate it in judge.py (same logic)
   - Construct `SessionEvaluation(**data)`, return it

2. **Add `store_evaluation()` to `db/queries.py`**:
   ```python
   def store_evaluation(session_id: str, evaluation: SessionEvaluation) -> str:
       # INSERT INTO evaluations (session_id, evaluation, overall_score) VALUES (...)
       # evaluation.model_dump() → json.dumps() → JSONB
   ```

3. **Wire into `app.py`**:
   - Add an "Evaluate Session" button below the chat, visible once `pipeline_state` is set
   - Add a `gr.Accordion("Session Evaluation", open=False)` with a `gr.Markdown` output
   - Button handler:
     ```python
     def handle_evaluate(history, pipeline_result):
         if not history:
             return "No conversation to evaluate yet."
         evaluation = evaluate_session(history, pipeline_result)
         # Format evaluation as readable markdown
         # store_evaluation if session_id available (pipeline_result.session_id)
         return format_evaluation_md(evaluation)
     ```
   - The `chatbot` state in Gradio's `ChatInterface` is NOT directly accessible as a component — you need to get history via `gr.State`. See "Gradio ChatInterface history access" below.

**Gradio ChatInterface history access:**
The `gr.ChatInterface` manages its own internal chatbot state. To access history from outside:
- Declare `chat_history = gr.State([])` before the ChatInterface
- OR: use `chatbot = gr.Chatbot()` explicitly and pass it to ChatInterface via `chatbot=chatbot` param
- The simplest approach: add `chat_history` as a `gr.State` that mirrors history, updated in `respond()` by yielding a side-channel update — but this is complex.
- **Easiest approach:** In the evaluate button handler, the chatbot component IS accessible if you keep a reference: `chatbot_component = gr.Chatbot()` and pass it to `gr.ChatInterface(chatbot=chatbot_component, ...)`. Then `evaluate_btn.click(fn=handle_evaluate, inputs=[chatbot_component, pipeline_state], outputs=[eval_output])`.

**`SESSION_EVALUATION` prompt** is already in `llm/prompts.py`. Use it as the system prompt.
The prompt asks for JSON with:
```json
{
  "answer_evaluations": [{"question": str, "answer_quality": int, "strengths": str, "weaknesses": str, "suggested_improvement": str}],
  "overall_score": int,
  "overall_feedback": str,
  "areas_to_study": [str]
}
```
`SessionEvaluation` and `AnswerEvaluation` Pydantic models already exist in `db/models.py`.

---

### Task 2 — Enhanced Guards (wire `classify_input()` to live chat)

**What:** Replace the current pattern-only `check_prompt_injection()` with an LLM-based classifier that runs on every user message. The stub `classify_input()` already exists in `guards.py` but is async and not wired.

**Current state in `guards.py`:**
```python
async def classify_input(text: str, chat_fn) -> tuple[bool, str]:
    """Stub — not yet wired. Uses MODELS['guard'] (gpt-5-nano)."""
```

**Implementation steps:**

1. **Make `classify_input()` synchronous** (remove `async`) — Gradio generators don't support `await` well:
   ```python
   def classify_input(text: str) -> tuple[bool, str]:
       from llm.router import chat
       from config import MODELS, PROVIDER_OPENROUTER
       result = chat(
           messages=[
               {"role": "system", "content": GUARD_PROMPT},
               {"role": "user", "content": text[:500]},
           ],
           model=MODELS["guard"],
           provider=PROVIDER_OPENROUTER,
           max_tokens=20,
           temperature=0.0,
           extra_body={"reasoning": {"effort": "low"}},
       )
       category = result["content"].strip().upper()
       # Normalise: some models return "LEGITIMATE." or "LEGITIMATE\n"
       for valid in ("LEGITIMATE", "JAILBREAK", "OFF_TOPIC", "MALICIOUS"):
           if valid in category:
               return valid == "LEGITIMATE", valid
       return True, "UNKNOWN"   # fail-open: guard error → allow through
   ```

2. **Wire into `respond()` in `app.py`** — replace the pattern-only call:
   ```python
   # BEFORE:
   safe, reason = check_prompt_injection(message)

   # AFTER:
   safe, reason = check_prompt_injection(message)   # keep pattern check (fast)
   if safe:
       safe, reason = classify_input(message)       # LLM classifier (slower)
   ```
   Keep both checks: pattern check is instant and catches obvious injections before spending a guard LLM call.

3. **Fail-open on guard error** — already in the stub; if classify_input throws, log and allow through. Do NOT block the user if the guard LLM is down.

4. **Add guard result to logs**: `logger.info("Guard: %s", reason)` in respond() after classification.

---

### Task 3 — UX Polish (optional, low priority)

- Show evaluation button only when pipeline has run (use `gr.Button(interactive=False)` by default, update on pipeline completion)
- Format evaluation markdown nicely: overall score as headline, per-answer table or collapsible sections
- Any remaining rough edges spotted during testing

---

## Key Technical Constraints

### Do NOT change these (working correctly):
- `parsers/pipeline.py` — fully functional, all parse calls use `extra_body={"reasoning": {"effort": "low"}}`
- `llm/router.py` — `chat()` signature with `extra_body` param works
- `app.py` streaming loop — `in_think_block` tracking for Ollama think tags is correct
- `db/queries.py` — all CRUD working

### JSON parsing pattern (use `_parse_llm_json` from `parsers/pipeline.py` or replicate):
```python
def _parse_llm_json(raw: str, label: str) -> dict:
    content = raw.strip()
    if not content:
        raise ValueError(f"{label} failed: LLM returned empty response")
    if content.startswith("```"):
        content = content.split("```", 2)[-1] if content.count("```") >= 2 else content
        content = content.lstrip("json").strip().rstrip("`").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} failed: invalid JSON — {e}") from e
```

### Prompt syntax: do NOT use `{{` / `}}` in prompts that are NOT passed through `.format()`
- `SESSION_EVALUATION` in `llm/prompts.py` currently has single braces `{` `}` — correct, do not change
- Only `QUESTION_GENERATION` and `INTERVIEWER_PERSONA` use `.format()` — they correctly use `{{`/`}}`

### Pydantic v2: use `model.model_dump()` not `model.dict()`

### pgvector storage: `json.dumps(obj.model_dump())` then `%s` param for JSONB columns

### Gradio version: 4.x (not 5.x) — use `gr.ChatInterface`, `gr.State`, generator-based respond()

---

## Infrastructure
- **Docker:** `docker compose up --build` from project root
- **Ports:** App=7860, PostgreSQL=5432 (host), Ollama=11434 (host)
- **Env vars:** `.env` file (OPENROUTER_API_KEY, DATABASE_URL, etc.)
- **Logs:** `docker compose logs -f app`
- **Branch:** `personal/dev` → commit + push after session

---

## Files to Create/Modify in Session 3

| File | Action | Description |
|------|--------|-------------|
| `llm/judge.py` | **CREATE** | `evaluate_session(history, pipeline_result) → SessionEvaluation` |
| `db/queries.py` | **MODIFY** | Add `store_evaluation(session_id, evaluation)` |
| `llm/guards.py` | **MODIFY** | Make `classify_input()` sync; wire pattern |
| `app.py` | **MODIFY** | Evaluate button + output accordion; wire classify_input |
| `tests/test_judge.py` | **CREATE** | Unit tests for judge (mock `llm.judge.chat`) |
| `tests/test_guards.py` | **CREATE** | Unit tests for classify_input (mock `llm.guards.chat`) |

---

## Session 3 Done Criteria
1. "Evaluate Session" button runs LLM-as-Judge (gpt-5) on the chat history
2. Returns `SessionEvaluation` with per-answer scores + overall score
3. Persisted to `evaluations` table
4. `classify_input()` is synchronous and wired into `respond()` — every user message passes LLM guard
5. Guard fails open (error → allow through)
6. All new code has unit tests
7. Confirmed working in Docker (manually test via UI)
