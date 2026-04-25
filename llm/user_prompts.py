"""User-facing input template prompts.

These appear in the 'Input Templates' accordion in the UI.
Clicking a template replaces the chat input with the template text,
ready to edit before sending.
"""

INPUT_TEMPLATES = [
    {
        "label": "Start the interview",
        "text": "Please begin the interview. Introduce yourself briefly and ask your first question.",
    },
    {
        "label": "Weakest skills focus",
        "text": "Based on my CV and the job description, list 5 interview questions targeting my weakest or most underrepresented skills.",
    },
    {
        "label": "Behavioral round",
        "text": "Ask me 3 behavioral questions relevant to this role. Use the STAR method format.",
    },
    {
        "label": "Harder follow-up",
        "text": "That was too straightforward. Ask a more challenging follow-up on the same topic.",
    },
    {
        "label": "Session evaluation",
        "text": "Please evaluate my performance in this session so far. Give me an overall readiness score and the top 3 things I should improve.",
    },
]
