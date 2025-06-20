from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from upmix_logic import upmix
import tempfile
import os

app = FastAPI()

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    output_format: str = Form("7.1.4")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    output_path = input_path.replace(".wav", f"_{output_format}.wav")

    # ✅ IR 파일 경로 추가
    ir_L_path = "/home/ubuntu/upmixer/ir/ir_left.wav"
    ir_R_path = "/home/ubuntu/upmixer/ir/ir_right.wav"

    # ✅ IR 경로와 함께 업믹스 호출
    upmix(input_path, output_path, ir_L_path, ir_R_path, output_format=output_format)

    return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")
