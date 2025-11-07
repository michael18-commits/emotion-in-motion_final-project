import io
import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional
from PIL import Image

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Emotion in Motion", page_icon="🎨", layout="wide")

# (省略未改部分，仍是原逻辑...) 这里只展示关键改动段落

# --- inside single image render section ---
        png_bytes = generate_art(
            img_seed=img_seed,
            steps_today=steps_v,
            hr_avg=hr_v,
            sleep_hours=sleep_v,
            fatigue=fatigue_v,
            mood=mood_v
        )
        img = Image.open(io.BytesIO(png_bytes))
        st.image(img, use_container_width=True)
        fname = f"emotion_in_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button("Download PNG", data=png_bytes, file_name=fname, mime="image/png")

# --- inside collage render section ---
    if collage_on and df is not None and len(df) >= 2:
        collage_bytes = generate_collage(df, mood_fallback=mood, seed=img_seed)
        if collage_bytes:
            collage_img = Image.open(io.BytesIO(collage_bytes))
            st.image(collage_img, use_container_width=True)
            fname = f"emotion_in_motion_collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button("Download Collage PNG", data=collage_bytes, file_name=fname, mime="image/png")
