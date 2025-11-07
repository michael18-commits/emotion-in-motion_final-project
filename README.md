# Emotion in Motion 🎨
**Turning health & mood into generative art**

## Overview
An interactive Streamlit app that converts daily health metrics and mood into a personalized abstract artwork. Users can upload a CSV (flexible column names) or use sliders, export PNGs, and optionally generate a short **AI Health Summary** (OpenAI). A 7-day collage mode is included.

## How it works (mapping)
- **Mood → Colors** (palette)
- **Steps → Complexity** (shapes/bubbles)
- **Heart rate → Flow/Jitter** (curvature)
- **Sleep → Softness/Transparency**
- **Fatigue → Density/Line weight**

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Preferred columns:
```
date, steps, heart_rate_avg, sleep_hours, mood, fatigue
```
Common aliases (auto-mapped): `step_count`, `avg_hr`, `sleep`, `tiredness`, etc.  
See `sample_data.csv`.

## Deploy (Streamlit Cloud)
- Push to a **public GitHub repo**
- Streamlit Cloud → **New app** → select `app.py` → Deploy  
- To enable **AI Health Summary**, add this to **Secrets**:
```
OPENAI_API_KEY = "sk-..."
```

## Tech
Python, Streamlit, NumPy, Pandas, Matplotlib, OpenAI.

## License
MIT (or your choice).
