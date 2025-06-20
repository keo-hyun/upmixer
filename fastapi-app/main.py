from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from upmix_logic import upmix
import tempfile
import os

app = FastAPI()

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    output_format: str = Form("7.1.4")  # Form을 통해 포맷 입력 받기
):
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    output_path = input_path.replace(".wav", f"_{output_format}.wav")

    # ✅ IR 파일 경로 수정
    ir_L_path = "ir/ir_left.wav"
    ir_R_path = "ir/ir_right.wav"

    try:
        upmix(input_path, output_path, ir_L_path, ir_R_path, output_format=output_format)
        return StreamingResponse(
            open(output_path, "rb"),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=upmixed_{output_format}.wav"}
        )
    finally:
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
