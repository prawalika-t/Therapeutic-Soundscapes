# streamlit_iso_music_app.py

import streamlit as st
import numpy as np
from collections import Counter
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
from IPython.display import Audio
import os

# Set up
st.title("🎵 Therapeutic Music Generator (ISO Principle)")

# Genre selection
genre = st.selectbox("Choose your preferred genre:", ['ambient', 'pop', 'jazz', 'classical', 'metal'])

# User mood selection
start_mood = st.selectbox("How are you feeling now?", ['anxious', 'sad', 'angry', 'calm', 'happy'])

# Define valence-arousal map
mood_va_map = {
    "anxious": (0.2, 0.8),
    "sad": (0.1, 0.1),
    "angry": (0.1, 0.9),
    "calm": (0.8, 0.2),
    "happy": (0.9, 0.9),
}

# Define mood prompt styles
mood_style_map = {
    "anxious": "tense, fast, synth",
    "sad": "slow, minor, violin",
    "angry": "intense, distorted, drums",
    "calm": "soft, ambient, piano",
    "happy": "upbeat, acoustic, guitar",
}

def find_opposite_mood(mood):
    val, ar = mood_va_map[mood]
    target = (1 - val, 1 - ar)
    min_dist = float('inf')
    opposite = mood
    for m, (v, a) in mood_va_map.items():
        if m == mood:
            continue
        dist = np.linalg.norm(np.array([v, a]) - np.array(target))
        if dist < min_dist:
            min_dist = dist
            opposite = m
    return opposite

target_mood = find_opposite_mood(start_mood)

st.markdown(f"**Target mood:** {target_mood} (ISO transition)")

# On button click, generate music
if st.button("🎧 Generate Therapy Music"):
    model = MusicGen.get_pretrained('melody')
    model.set_generation_params(duration=15)

    steps = 5
    prompts = []
    for i in range(steps):
        if i < steps // 2:
            mood = start_mood
        elif i == steps // 2 and steps % 2 == 1:
            mood = f"{start_mood}, {target_mood}"
        else:
            mood = target_mood
        prompt = f"{genre} music, {mood_style_map.get(mood.strip(), mood)}"
        prompts.append(prompt)

    output_dir = "./streamlit_segments"
    os.makedirs(output_dir, exist_ok=True)
    segment_paths = []

    for i, prompt in enumerate(prompts):
        st.write(f"Generating segment {i+1}: {prompt}")
        audio = model.generate([prompt])
        audio_path = os.path.join(output_dir, f"segment_{i+1}.wav")
        audio_write(audio_path[:-4], audio[0].cpu(), model.sample_rate, strategy="loudness")
        segment_paths.append(audio_path)

    final = AudioSegment.empty()
    for path in segment_paths:
        seg = AudioSegment.from_wav(path).fade_in(3000).fade_out(3000)
        final += seg

    final_path = os.path.join(output_dir, "final_therapy_music.wav")
    final.export(final_path, format="wav")

    st.audio(final_path)
    st.success("Music generation complete. Relax and enjoy 🎶")
