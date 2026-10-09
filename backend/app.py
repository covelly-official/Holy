from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: we don't strictly load at startup so the server boots instantly, 
    # and loads on the first request.
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"]
)

app.mount("/outputs", StaticFiles(directory=engine.OUTPUT_DIR), name="outputs")

@app.post("/generate")
async def generate_endpoint(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    negative_prompt: str = Form(None),
    steps: int = Form(30),
    strength: float = Form(0.8),
    seed: int = Form(None)
):
    image_bytes = await image.read()
    mask_bytes = await mask.read()
    
    seed_val = seed if seed else None
    
    kwargs = {
        "prompt": prompt,
        "image_bytes": image_bytes,
        "mask_bytes": mask_bytes,
        "strength": strength,
        "steps": steps,
        "seed": seed_val
    }
    
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
        
    from starlette.concurrency import run_in_threadpool
    result = await run_in_threadpool(engine.generate, **kwargs)
    
    if result.get("status") == "aborted":
        return result
        
    result["url"] = f"/outputs/{result['file']}"
    return result

@app.get("/progress")
async def progress():
    return engine._current_progress

@app.post("/abort")
async def abort():
    engine.abort_generation()
    return {"status": "aborting"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/history")
def get_history():
    files = list(engine.OUTPUT_DIR.glob("*.png"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return {"images": [f.name for f in files]}

import os
ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)