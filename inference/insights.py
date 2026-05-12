from __future__ import annotations

from typing import Any

from openai import OpenAI


def get_ai_insights(
    summary: dict[str, Any],
    *,
    api_key: str | None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Optional narrative layer. Keep the model small/cheap by default (gpt-4o-mini).

    Do not hardcode secrets: pass api_key from Streamlit secrets or environment in the app layer.
    """
    if not api_key:
        return (
            "LLM insights are disabled until you add an API key. "
            "For Streamlit Community Cloud: Settings → Secrets → add OPENAI_API_KEY (and optionally OPENAI_MODEL)."
        )

    client = OpenAI(api_key=api_key)
    unit = summary.get("unit", "units")
    prompt = f"""You are a senior statistician talking to a data analyst.

Analyze the inference summary below and respond in Markdown with exactly these sections:

### 1. What the data suggests
### 2. CLT vs bootstrap (if both exist)
### 3. What to trust and why
### 4. Plain-language interpretation (use unit: {unit})
### 5. Next steps

Rules:
- Be precise; avoid claiming 'random sample' unless stated.
- If only bootstrap exists, explain why classical shortcuts may be weak.
- Keep the entire response under ~250 words.

Summary (Python dict, values may be missing):
{summary}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
