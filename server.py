import os
import json
import io
import base64
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MP Studio - Local Ultra HD Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "references")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

@app.get("/api/examples")
async def list_examples():
    if os.path.exists(EXAMPLES_DIR):
        files = [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        return files
    return []

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/references")
async def get_references():
    return catalog

@app.post("/api/generate")
async def generate_image(
    prompt: str = Form(""),
    ref_image_name: str = Form(""),
    main_image: UploadFile = File(None),
    second_image: UploadFile = File(None),
    mask_image: UploadFile = File(None)
):
    try:
        ref_path = None
        if ref_image_name:
            chosen_path = os.path.join(EXAMPLES_DIR, ref_image_name)
            if os.path.exists(chosen_path):
                ref_path = chosen_path
                
        temp_main = "temp_main_input.jpg"
        temp_mask = "temp_mask_input.png"
        temp_out = f"static/generations/v4_out_{int(time.time())}.jpg"
        os.makedirs("static/generations", exist_ok=True)

        if main_image:
            main_bytes = await main_image.read()
            with open(temp_main, "wb") as f:
                f.write(main_bytes)
        else:
            temp_main = ref_path if ref_path else os.path.join(EXAMPLES_DIR, "Marcos 01 leve sorriso.png")

        temp_mask_path = None
        if mask_image:
            mask_bytes = await mask_image.read()
            with open(temp_mask, "wb") as f:
                f.write(mask_bytes)
            temp_mask_path = temp_mask

        from local_engine_v4 import process_unified_v4
        print(f"[UNIFIED SERVER v5.1] Executando processamento local Kimi AI (GPU RTX 3050)...", flush=True)
        res_file, logs = process_unified_v4(temp_main, ref_path, temp_out, mask_img_path=temp_mask_path)
        
        clean_url = "/" + res_file.replace("\\", "/")
        return JSONResponse({
            "status": "success",
            "image_url": clean_url,
            "images": [clean_url],
            "logs": logs
        })

    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[{time.strftime('%H:%M:%S')}] ERRO NO SERVER.PY:\n{err_detail}", flush=True)
        return JSONResponse({"status": "error", "message": f"{type(e).__name__}: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
