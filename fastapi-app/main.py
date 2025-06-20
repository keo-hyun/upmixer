from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from upmix_logic import upmix
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(__file__))  # ~/upmixer
ir_L_path = os.path.join(base_dir, "ir", "ir_left.wav")
ir_R_path = os.path.join(base_dir, "ir", "ir_right.wav")

print(f"IR LEFT PATH: {ir_L_path}")
print(f"IR RIGHT PATH: {ir_R_path}")

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    output_format: str = Form("7.1.4")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    output_path = input_path.replace(".wav", f"_{output_format}.wav")

    upmix(input_path, output_path, ir_L_path, ir_R_path, output_format=output_format)

    return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")
