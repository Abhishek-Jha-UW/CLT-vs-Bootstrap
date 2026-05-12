# Inference Arena (CLT vs bootstrap)

Interactive Streamlit app for analysts who want a **clean, defensible** comparison between:

- **Classical intervals** where standard pivots exist (Student *t* for a mean, Wilson for a binomial proportion)
- **Percentile bootstrap** for the same estimand (plus bootstrap-first paths for median / SD / percentiles)

The first implementation is preserved under `initial_data/` for portfolio traceability.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional local secrets file (never commit real keys):

1. Create `.streamlit/secrets.toml`
2. Add:

```toml
OPENAI_API_KEY = "sk-..."
# optional override (defaults to a small/cheap model)
OPENAI_MODEL = "gpt-4o-mini"
```

## Streamlit Community Cloud

In **App settings → Secrets**, add the same keys. **Do not** commit API keys to GitHub.

Recommended default model: **`gpt-4o-mini`** (low cost, strong enough for short explanations).

## Tests

```bash
pytest -q
```

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `inference/` | Pure-Python statistics + optional LLM client wrapper |
| `initial_data/` | Archived first version (`app.py`, `model.py`, …) |
| `.streamlit/secrets.toml.example` | Copy/rename pattern for local secrets |
