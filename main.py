from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from DetectAudio import predict
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict(file_path)  # 결과: "siren", "announcement", "normal"
        is_danger = (result == "siren" or result == "announcement") # 🚨 사이렌만 True 처리
        return JSONResponse(content={"detected": is_danger})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})