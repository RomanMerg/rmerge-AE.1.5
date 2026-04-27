"""Diagnostic: test JD parse call variants to find what works."""
import os, json, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

JD_SYSTEM = """Analyze the following job description and extract structured information.

Return ONLY valid JSON matching this exact schema:
{
  "role_title": "string",
  "role_level": "junior|mid|senior|lead|principal",
  "requirements": [
    {"skill": "string", "severity": "critical|important|nice_to_have"}
  ],
  "nice_to_haves": ["string"],
  "company_type": "startup|scaleup|enterprise|agency|consultancy|other",
  "summary": "one-sentence role summary"
}

Be thorough — extract every mentioned skill, tool, technology, and soft skill requirement."""

SHORT_JD = "Senior Python backend engineer at a startup. Requires FastAPI, PostgreSQL, Docker. Nice to have: Kubernetes, Redis."

def test(label, **extra_kwargs):
    print(f"\n=== {label} ===")
    resp = client.chat.completions.create(
        model="openai/gpt-5-mini",
        messages=[
            {"role": "system", "content": JD_SYSTEM},
            {"role": "user", "content": SHORT_JD},
        ],
        **extra_kwargs,
    )
    content = resp.choices[0].message.content
    usage = resp.usage
    print(f"content ({len(content or '')} chars): {repr((content or '')[:200])}")
    print(f"finish_reason: {resp.choices[0].finish_reason}")
    print(f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
        print(f"  reasoning_tokens: {getattr(usage.completion_tokens_details, 'reasoning_tokens', '?')}")

# Test 1: baseline — same as current code
test("1. max_tokens=2048 (current)", max_tokens=2048)

# Test 2: big budget — does more room fix it?
test("2. max_tokens=8192", max_tokens=8192)

# Test 3: reasoning_effort low — cap reasoning budget
test("3. max_tokens=2048 + reasoning_effort=low", max_tokens=2048,
     extra_body={"reasoning": {"effort": "low"}})

# Test 4: no max_tokens at all
test("4. no max_tokens")
