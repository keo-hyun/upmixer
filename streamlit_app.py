import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
from upmix_logic import upmix_and_normalize

st.title("🎧 Stereo to 7.1.4 Upmixer")

uploaded_file = st.file_uploader("Upload a stereo WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("Loading IR files..."):
        ir_L, _ = librosa.load("Bricasti M7 Room 02 -Studio B Close-L.wav", sr=None)
        ir_R, _ = librosa.load("Bricasti M7 Room 02 -Studio B Close-R.wav", sr=None)
    
    with st.spinner("Processing upmix and normalization..."):
        y, sr = librosa.load(uploaded_file, sr=None, mono=False)
        if y.ndim == 1:
            y = np.vstack((y, y))  # Duplicate mono to stereo

        output = upmix_and_normalize(y, sr, ir_L, ir_R)
        
        buf = io.BytesIO()
        sf.write(buf, output.T, sr, format='WAV', subtype='PCM_24')
        buf.seek(0)

        st.success("✅ Processing complete!")
        st.download_button("📥 Download 7.1.4 Upmixed File", buf, file_name="upmixed_7.1.4.wav", mime="audio/wav")
