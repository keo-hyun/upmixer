import logging
import streamlit as st
import requests
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [streamlit] %(message)s"
)

# FastAPI 서버 주소 (퍼블릭 IP 기반)
API_URL = "http://16.176.222.198:8001/upload-audio/"

# 앱 제목
st.title("🎧 Stereo to Multi-Channel Upmixer")

# 출력 포맷 선택
output_options = ["5.1", "5.1.2", "7.1", "7.1.2", "7.1.4"]
selected_format = st.selectbox("Select output format", output_options, index=4)

# 오디오 업로드
uploaded_file = st.file_uploader("Upload a stereo audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🚀 Start Upmixing"):
        with st.spinner("Sending to server for processing..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            data = {
                "output_format": selected_format
            }

            logging.info(f"[REQ] Uploading {uploaded_file.name} with format {selected_format} to FastAPI")

            try:
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()

                logging.info("[RESP] Received upmixed file from server")

                st.success("✅ Processing complete!")
                st.download_button(
                    label=f"📥 Download {selected_format} Upmixed File",
                    data=response.content,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_{selected_format}.wav",
                    mime="audio/wav"
                )
            except requests.exceptions.RequestException as e:
                logging.error(f"[ERROR] Request failed: {e}")
                st.error(f"❌ Server error: {e}")
