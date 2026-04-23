# backend/main.py （FastAPI，新增端点）
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import uvicorn
import sys
sys.path.append("/data/unet-attention-dsconv_github")
from unet.inference import predict_and_visualize
from langchain_pipeline.chain import generate_report

app = FastAPI()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.post("/brain-detect")
async def brain_detect(file: UploadFile = File(...)):
    try:
        # 保存上传文件
        image_path = os.path.join(TEMP_DIR, file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 预测 + 可视化
        mask, vis_path = predict_and_visualize(image_path)

        # 生成报告（带可视化图像）
        report = generate_report(mask, vis_path)

        return JSONResponse({
            "status": "success",
            "visualized_image": vis_path,   # 或返回 base64
            "report": report,
            "mask_shape": mask.shape
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001,reload=True)
