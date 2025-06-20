from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from upmix_logic import upmix  # 로직 가져오기
import tempfile
import os

app = FastAPI()

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    output_format: str = Form("7.1.4")
):
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    output_path = input_path.replace(".wav", f"_{output_format}.wav")

    # 업믹스 처리 (당신의 로직 함수 사용)
    upmix(input_path, output_path, output_format)

    return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")
