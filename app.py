import io
import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional

# Optional OpenAI import (only used if API key provided)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Emotion in Motion", page_icon="🎨", layout="wide")

# ----------------------------
# Palettes & Mapping
# ----------------------------
MOOD_PALETTES = {
    "calm":      ["#2B6CB0", "#63B3ED", "#BEE3F8", "#90CDF4"],
    "energetic": ["#C53030", "#F56565", "#F6AD55", "#ED8936"],
    "tired":     ["#4A5568", "#A0AEC0", "#CBD5E0", "#718096"],
    "focused":   ["#22543D", "#38A169", "#9AE6B4", "#68D391"],
    "blue":      ["#1A365D", "#2C5282", "#4299E1", "#BEE3F8"],
    "joyful":    ["#B83280", "#ED64A6", "#F6ADCD", "#FBD38D"],
    "anxious":   ["#2D3748", "#805AD5", "#4C51BF", "#718096"]
}
DEFAULT_MOOD = "calm"

# Accept common column aliases from different exports (Google Fit, Apple Health, Samsung Health, etc.)
COLUMN_ALIASES = {
    "date": ["date", "day", "timestamp"],
    "steps": ["steps", "step", "step_count", "steps_count", "count_steps"],
    "heart_rate_avg": ["heart_rate_avg", "hr_avg", "avg_hr", "average_heart_rate", "mean_hr", "heart rate avg"],
    "sleep_hours": ["sleep_hours", "sleep", "sleep_time_hours", "hours_sleep", "sleep_duration"],
    "mood": ["mood", "feeling", "emotion", "status"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "fatigue_level"]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map aliases to canonical names and keep extras."""
    lower = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=lower)

    mapped = {}
    used = set()
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns and a not in used:
                mapped[canon] = a
                used.add(a)
                break

    # Create unified view with canonical names where possible
    for canon, src in mapped.items():
        if canon != src:
            df[canon] = df[src]

    # Coerce types carefully
    for col in ["steps", "heart_rate_avg", "sleep_hours", "fatigue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def pick_palette(mood: str):
    return MOOD_PALETTES.get(mood, MOOD_PALETTES[DEFAULT_MOOD])

def lerp(a, b, t):
    return a + (b - a) * t

def hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

def blend_hex(c1, c2, t):
    r1,g1,b1 = hex_to_rgb01(c1)
    r2,g2,b2 = hex_to_rgb01(c2)
    r = lerp(r1, r2, t)
    g = lerp(g1, g2, t)
    b = lerp(b1, b2, t)
    return (r, g, b)

# ----------------------------
# Generative core
# ----------------------------
def generate_art(img_seed: int,
                 steps_today: int,
                 hr_avg: float,
                 sleep_hours: float,
                 fatigue: float,
                 mood: str,
                 canvas_size=(900, 600)) -> bytes:
    """
    Mapping:
    - steps_today  -> number of shapes / bubbles
    - hr_avg       -> flow jitter / curvature
    - sleep_hours  -> softness / transparency
    - fatigue      -> density / line width
    - mood         -> color palette
    """
    random.seed(img_seed)
    np.random.seed(img_seed)

    W, H = canvas_size
    palette = pick_palette(str(mood).strip().lower())

    # Derived parameters
    n_shapes = int(np.clip((steps_today or 0)/1200 + 6, 6, 40))
    jitter = np.clip(((hr_avg or 60) - 60)/60, 0.05, 0.9)
    softness = np.clip((sleep_hours or 0)/8, 0.3, 1.2)
    density = np.clip((fatigue if fatigue is not None else 0.4), 0.15, 1.0)
    alpha_base = np.clip(0.15 + (sleep_hours or 0)/12, 0.15, 0.85)

    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background gradient
    for i in range(120):
        t = i / 119
        c = blend_hex(palette[0], palette[-1], t)
        ax.add_patch(plt.Rectangle((0, t-0.01), 1, 0.02, linewidth=0, color=c, alpha=0.4*alpha_base))

    # Flow field
    field_scale = 1.0 + jitter * 0.8
    n_paths = int(80 * density)
    n_steps = int(200 * (0.7 + jitter))
    step_len = 0.0025 * (1.1 - softness*0.3)

    def flow_dir(x, y):
        return np.sin(10*x*field_scale) * np.cos(10*y*field_scale + jitter*np.pi)

    for _ in range(n_paths):
        x = random.random()
        y = random.random()
        tcol = random.random()
        c = blend_hex(random.choice(palette), random.choice(palette), tcol)
        a = alpha_base * (0.35 + 0.65*random.random()) * (1.0 - 0.5*density)
        lw = np.clip(0.5 + 2.5*density*(0.5+random.random()*0.5), 0.4, 2.5)

        xs, ys = [x], [y]
        for _ in range(n_steps):
            angle = flow_dir(x, y) * np.pi
            x += np.cos(angle) * step_len
            y += np.sin(angle) * step_len
            if x < 0 or x > 1 or y < 0 or y > 1:
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                x += (random.random()-0.5)*0.02
                y += (random.random()-0.5)*0.02
            xs.append(x); ys.append(y)

        ax.plot(xs, ys, linewidth=lw, alpha=a, color=c)

    # Mood bubbles (steps → richness)
    n_bubbles = int(np.clip((steps_today or 0)/2000 + 5, 5, 60))
    for _ in range(n_bubbles):
        cx, cy = random.random(), random.random()
        r = np.clip(np.random.normal(0.035, 0.02) * (1 + (steps_today or 0)/20000), 0.01, 0.12)
        c = blend_hex(random.choice(palette), random.choice(palette), random.random())
        a = np.clip(alpha_base*(0.25 + 0.6*random.random())*(1-softness*0.2), 0.1, 0.8)
        circle = plt.Circle((cx, cy), r, color=c, alpha=a, linewidth=0)
        ax.add_patch(circle)

    # Render to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Collage (7-day gallery)
# ----------------------------
def generate_collage(df: pd.DataFrame,
                     mood_fallback: str,
                     seed: int = 42) -> Optional[bytes]:
    """Take up to 7 rows and stitch them into one image (2x4 grid layout using 7 slots)."""
    rows = min(len(df), 7)
    if rows == 0:
        return None
    sub = df.tail(rows).copy()

    fig = plt.figure(figsize=(12, 6), dpi=150)
    for i in range(rows):
        r = sub.iloc[i]
        steps = int(r.get("steps", 8000) or 8000)
        hr = float(r.get("heart_rate_avg", 80) or 80.0)
        sleep = float(r.get("sleep_hours", 7.0) or 7.0)
        fatigue = float(r.get("fatigue", 0.4) or 0.4)
        mood = str(r.get("mood", mood_fallback) or mood_fallback).strip().lower()

        img_bytes = generate_art(seed + i, steps, hr, sleep, fatigue, mood, canvas_size=(600, 400))
        img = plt.imread(io.BytesIO(img_bytes), format="png")

        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img)
        ax.set_title(str(r.get("date", f"Day {i+1}")))
        ax.axis("off")

    # Fill the 8th slot with a legend
    if rows < 8:
        ax = fig.add_subplot(2, 4, 8)
        ax.axis("off")
        ax.set_title("Legend", fontsize=10)
        text = ("Mapping:\\n"
                "Mood → Colors\\n"
                "Steps → Complexity\\n"
                "Heart rate → Flow\\n"
                "Sleep → Softness\\n"
                "Fatigue → Density")
        ax.text(0.05, 0.9, text, va="top")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# AI Summary (optional, OpenAI)
# ----------------------------
def get_openai_client(api_key: Optional[str] = None):
    key = api_key or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key or not OPENAI_AVAILABLE:
        return None
    try:
        client = OpenAI(api_key=key)
        return client
    except Exception:
        return None

def build_summary_prompt(df: pd.DataFrame) -> str:
    # Create a compact JSON-like snapshot with basic stats
    desc = {
        "days": int(len(df)),
        "totals": {
            "steps_sum": float(pd.to_numeric(df.get("steps", pd.Series()), errors="coerce").fillna(0).sum()),
        },
        "averages": {
            "steps_avg": float(pd.to_numeric(df.get("steps", pd.Series()), errors="coerce").fillna(0).mean() or 0),
            "hr_avg": float(pd.to_numeric(df.get("heart_rate_avg", pd.Series()), errors="coerce").fillna(0).mean() or 0),
            "sleep_avg": float(pd.to_numeric(df.get("sleep_hours", pd.Series()), errors="coerce").fillna(0).mean() or 0),
            "fatigue_avg": float(pd.to_numeric(df.get("fatigue", pd.Series()), errors="coerce").fillna(0).mean() or 0),
        },
        "mood_counts": df.get("mood", pd.Series(dtype=str)).astype(str).str.lower().value_counts().to_dict(),
    }
    prompt = f"""
You are a concise health coach and art explainer. Based on the following weekly metrics snapshot, write:
1) A short, positive health summary (2-3 sentences).
2) One actionable suggestion (1 bullet).
3) A one-sentence explanation of how these inputs would influence the generative artwork (mapping).

Data snapshot (JSON-like):
{desc}

Keep the tone encouraging, concrete, and specific. Do not include code. Limit to ~120 words total.
"""
    return prompt.strip()

def generate_ai_summary(df: pd.DataFrame, api_key_input: Optional[str]) -> Optional[str]:
    client = get_openai_client(api_key_input)
    if client is None:
        return None
    try:
        prompt = build_summary_prompt(df)
        # Use Responses API (OpenAI Python SDK v1.x)
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        # Extract text safely
        if resp and resp.output_text:
            return resp.output_text.strip()
        # Fallback parse
        try:
            return resp.output[0].content[0].text.strip()
        except Exception:
            return None
    except Exception:
        return None

# ----------------------------
# UI
# ----------------------------
st.title("Emotion in Motion 🎨")
st.caption("Turn your health & mood into generative art")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Data Source", ["Upload CSV", "Manual Entry", "Use Sample"], index=0)

    mood = st.selectbox("Mood", list(MOOD_PALETTES.keys()), index=list(MOOD_PALETTES.keys()).index("calm"))
    img_seed = st.number_input("Image Seed", min_value=0, max_value=1_000_000, value=42, step=1)

    st.markdown("---")
    st.markdown("**Manual Entry (fallback)**")
    steps_today = st.slider("Steps", 0, 30000, 8000, step=500)
    hr_avg = st.slider("Average Heart Rate (bpm)", 40, 160, 80)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.1)
    fatigue = st.slider("Fatigue (0~1)", 0.0, 1.0, 0.4, step=0.05)

    st.markdown("---")
    collage_on = st.checkbox("Create 7-day collage (if data available)", value=False)

    st.markdown("---")
    st.subheader("AI Health Summary (optional)")
    st.caption("Provide an OpenAI API key via **Secrets** or paste here to enable a short health & art summary.")
    api_key_input = st.text_input("OpenAI API Key", type="password", value="")

col1, col2 = st.columns([1.2, 1])

# Load data by mode
df = None
if mode == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload CSV (preferred columns: date, steps, heart_rate_avg, sleep_hours, mood, fatigue)",
        type=["csv"]
    )
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df = normalize_columns(df)
        except Exception as e:
            st.error(f"CSV parse failed: {e}")

elif mode == "Use Sample":
    df = pd.DataFrame({
        "date": ["2025-11-01","2025-11-02","2025-11-03","2025-11-04","2025-11-05","2025-11-06","2025-11-07"],
        "steps": [8421,12100,4100,9600,3000,12850,7200],
        "heart_rate_avg": [79,92,72,88,68,95,76],
        "sleep_hours": [6.2,7.1,5.0,7.6,6.8,7.3,6.4],
        "mood": ["calm","energetic","tired","focused","blue","joyful","anxious"],
        "fatigue": [0.3,0.2,0.7,0.4,0.6,0.25,0.5],
    })

with col1:
    st.subheader("Generated Artwork")

    if collage_on and df is not None and len(df) >= 2:
        collage_bytes = generate_collage(df, mood_fallback=mood, seed=img_seed)
        if collage_bytes:
            st.image(collage_bytes, use_container_width=True)
            fname = f"emotion_in_motion_collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button("Download Collage PNG", data=collage_bytes, file_name=fname, mime="image/png")
    else:
        if df is not None and len(df) > 0:
            idx = st.slider("Pick a row (index)", 0, len(df)-1, len(df)-1)
            row = df.iloc[idx]
            steps_v = int(row.get("steps", steps_today) or steps_today)
            hr_v = float(row.get("heart_rate_avg", hr_avg) or hr_avg)
            sleep_v = float(row.get("sleep_hours", sleep_hours) or sleep_hours)
            mood_v = str(row.get("mood", mood) or mood).strip().lower()
            fatigue_v = float(row.get("fatigue", fatigue) or fatigue)
        else:
            steps_v, hr_v, sleep_v, mood_v, fatigue_v = steps_today, hr_avg, sleep_hours, mood, fatigue

        png_bytes = generate_art(
            img_seed=img_seed,
            steps_today=steps_v,
            hr_avg=hr_v,
            sleep_hours=sleep_v,
            fatigue=fatigue_v,
            mood=mood_v
        )
        st.image(png_bytes, use_container_width=True)
        fname = f"emotion_in_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button("Download PNG", data=png_bytes, file_name=fname, mime="image/png")

with col2:
    st.subheader("How inputs map → visuals")
    st.markdown("""
- **Mood → Colors**: palette drives background gradient and stroke tones  
- **Steps → Complexity**: more steps → more shapes/bubbles  
- **Heart Rate → Flow/Jitter**: higher bpm → more curvature & agitation  
- **Sleep → Softness/Transparency**: more sleep → softer & clearer layers  
- **Fatigue → Density/Line Weight**: higher fatigue → denser/darker strokes  
""")
    st.markdown("---")
    st.subheader("AI Summary")
    if df is not None and len(df) > 0:
        if OPENAI_AVAILABLE and (api_key_input or "OPENAI_API_KEY" in st.secrets or os.getenv("OPENAI_API_KEY")):
            with st.spinner("Generating health & art summary..."):
                summary = generate_ai_summary(df, api_key_input)
            if summary:
                st.success(summary)
            else:
                st.info("Could not generate summary. Please check your API key or try again.")
        else:
            st.caption("Provide an OpenAI API key in **Secrets** or paste above to enable.")
    else:
        st.caption("Upload data or use sample to enable AI summary.")

    st.markdown("---")
    st.subheader("Notes")
    st.markdown("""
- CSV columns are flexible; common aliases are auto-mapped.  
- Use the **collage** option to generate a 7-day gallery.  
- Export PNG with timestamped file names.
""")

st.markdown("---")
with st.expander("About the project"):
    st.write("""
**Emotion in Motion** visualizes how your body feels beyond numbers.  
Upload your health data or use the sliders, pick your mood, and turn it into art.
""")
