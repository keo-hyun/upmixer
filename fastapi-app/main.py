from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from upmix_logic import upmix
import tempfile
import os
import librosa
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# IR 파일 로드
base_dir = os.path.dirname(os.path.dirname(__file__))  # ~/upmixer
ir_L_path = os.path.join(base_dir, "ir", "ir_left.wav")
ir_R_path = os.path.join(base_dir, "ir", "ir_right.wav")

ir_L, ir_sr = librosa.load(ir_L_path, sr=None, mono=True)
ir_R, _ = librosa.load(ir_R_path, sr=ir_sr, mono=True)

logging.info(f"IR LOADED: L={ir_L_path}, R={ir_R_path}, SR={ir_sr}")

# 업믹스 API
@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    output_format: str = Form("7.1.4")
):
    logging.info(f"[RECEIVED] Upload: {file.filename}, format: {output_format}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    output_path = input_path.replace(".wav", f"_{output_format}.wav")
    logging.info(f"[START] Upmixing to {output_format}...")

    try:
        upmix(input_path, output_path, ir_L, ir_R, ir_sr, output_format=output_format)
        logging.info(f"[DONE] Output saved: {output_path}")
    except Exception as e:
        logging.error(f"[ERROR] Upmix failed: {e}")
        raise

    return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))
