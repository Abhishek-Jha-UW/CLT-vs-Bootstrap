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
    prompt = f"""You are a senior statistician writing for a data analyst who may not remember theory details.

You will receive a Python dictionary called the **summary**. Use **only** facts that appear in the summary. Do not invent sample sizes, intervals, or widths. If a field is missing, say you do not have it.

Write Markdown with **exactly** these five sections (same headings):

### 1. What the sample is telling you
### 2. How classical and bootstrap compare (only if both intervals exist in the summary)
### 3. What I would emphasize in a readout
### 4. Plain-language interpretation (use the unit: {unit})
### 5. Sensible next steps

**Width comparison rule (important):**
- If `width_difference_pct_vs_classical` is present, you may describe relative width using that percentage (bootstrap vs classical).
- If `width_difference_absolute` is present, you may mention the absolute difference in the **same units as the estimand** (not as a substitute for the percent unless the summary has no percent field).
- Never describe width change using a bare number without stating whether it is **percent** or **absolute**.

Style:
- Short paragraphs, no filler, under ~280 words total.
- Do not claim the data are a "random sample" unless the summary explicitly says so.

Summary:
{summary}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
