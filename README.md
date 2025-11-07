# Emotion in Motion — Professor-ready (visible API key design)
This build shows the API key design clearly **on page**:
- If `OPENAI_API_KEY` exists in **Streamlit Secrets**, the app auto-uses it and shows a green badge.
- Otherwise, the app blocks and asks the user to paste an OpenAI API key (session-only, not stored).
- The expander includes a didactic link to **Alpha Vantage** as a generic example of what an API key is.

Also includes a 10s per-session cooldown for AI calls to protect costs.

## Deploy
1) (Recommended) Put course key in Streamlit Cloud → App → Settings → Secrets:
   `OPENAI_API_KEY = "sk-..."`
2) Deploy; professor opens the link → green badge appears, no action needed.
3) Python 3.11 pinned via `runtime.txt`.

