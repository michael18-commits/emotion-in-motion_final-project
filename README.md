# Emotion in Motion — API key required
This version requires an OpenAI API key *before* the app can be used.

## How to use
1. Get your API key from your OpenAI dashboard (API keys → Create new secret key).
2. Start the app and paste the key into the **Enter your API key** field.
3. Click **Use this key** to unlock the interface.

You may also pre-configure an app-level key via Streamlit Secrets:
- In Streamlit Cloud → Settings → Secrets: `OPENAI_API_KEY = "sk-..."`
If a secret key exists, the app will auto-use it (no paste needed).
