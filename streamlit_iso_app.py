# iso_music_survey.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pydub import AudioSegment
import tempfile
import base64
import os

# Load model and processor
@st.cache_resource

def load_musicgen():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_musicgen()

# Mood to Valence-Arousal mapping
mood_va_map = {
    "anxious": (0.2, 0.8),
    "calm": (0.8, 0.2),
    "happy": (0.9, 0.9),
    "sad": (0.1, 0.1),
    "angry": (0.1, 0.9),
    "relaxed": (0.7, 0.3),
    "fearful": (0.2, 0.9),
    "peaceful": (0.9, 0.2),
}

def infer_mood_from_hr(hr):
    if hr > 95:
        return "anxious"
    elif hr > 85:
        return "fearful"
    elif hr > 75:
        return "angry"
    elif hr > 70:
        return "happy"
    elif hr > 65:
        return "sad"
    elif hr > 55:
        return "relaxed"
    else:
        return "calm"

def find_opposite_mood(mood):
    val, ar = mood_va_map.get(mood, (0.5, 0.5))
    target = (1 - val, 1 - ar)
    min_dist = float('inf')
    opposite = mood
    for m, (v, a) in mood_va_map.items():
        dist = np.linalg.norm(np.array([v, a]) - np.array(target))
        if dist < min_dist:
            min_dist = dist
            opposite = m
    return opposite

def suggest_genre(mood):
    return {
        "calm": "ambient",
        "relaxed": "lofi",
        "happy": "pop",
        "sad": "classical",
        "anxious": "jazz",
        "angry": "metal",
        "fearful": "cinematic",
        "peaceful": "ambient",
    }.get(mood, "ambient")

def generate_prompt(start_mood, target_mood, genre):
    return (
        f"Transition from {start_mood} to {target_mood} mood using {genre} instrumental music. "
        f"Gradual emotional evolution."
    )

def numpy_to_audiosegment(np_audio, sample_rate):
    audio_16bit = (np_audio * 32767).astype(np.int16)
    return AudioSegment(
        audio_16bit.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

def generate_music(prompt):
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
    audio_values = model.generate(**inputs, max_new_tokens=2500)
    audio_array = audio_values[0].cpu().numpy().flatten()
    sampling_rate = model.config.audio_encoder.sampling_rate
    return numpy_to_audiosegment(audio_array, sampling_rate)

def audio_download_link(audio_segment, filename="final_iso_music.wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_segment.export(f.name, format="wav")
        with open(f.name, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download Final Music</a>'

# UI
st.title("🧠 Mood-Aware Music Generator (ISO Principle)")

hr = st.slider("❤️ Your Heart Rate (bpm):", min_value=40, max_value=120, value=72)
genre = st.selectbox("🎵 Choose a Genre:", ['ambient', 'pop', 'jazz', 'classical', 'metal', 'lofi', 'cinematic'])

if st.button("Generate Music"):
    start_mood = infer_mood_from_hr(hr)
    target_mood = find_opposite_mood(start_mood)
    prompt = generate_prompt(start_mood, target_mood, genre)

    st.write(f"🎭 **Detected Mood**: {start_mood}")
    st.write(f"🎯 **Target Mood (ISO Principle)**: {target_mood}")
    st.write(f"📝 **Prompt**: {prompt}")

    with st.spinner("Generating music..."):
        final_music = generate_music(prompt)

    st.audio(final_music.export(format="wav"), format='audio/wav')
    st.markdown(audio_download_link(final_music), unsafe_allow_html=True)

    mood_rating = st.slider("Rate your final mood (1=Low, 5=High):", 1, 5, 3)
    st.write(f"Thanks! You rated your post-music mood as **{mood_rating}/5**")
