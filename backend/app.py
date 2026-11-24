from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import engine

# Filter out high-frequency polling from Uvicorn terminal access log
class QuietPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/progress" not in msg and "/health" not in msg

logging.getLogger("uvicorn.access").addFilter(QuietPollingFilter())

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
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    model_id: str = Form("Uminosachi/realisticVisionV51_v51VAE-inpainting"),
    negative_prompt: str = Form(None),
    steps: int = Form(30),
    strength: float = Form(0.8),
    seed: int = Form(None),
    aspect_ratio: str = Form("original"),
    width: int | None = Form(None),
    height: int | None = Form(None),
    allow_nsfw: bool = Form(True),
):
    image_bytes = None
    if image is not None:
        try:
            content = await image.read()
            if content and len(content) > 0:
                image_bytes = content
        except Exception:
            image_bytes = None

    mask_bytes = None
    if mask is not None:
        try:
            content = await mask.read()
            if content and len(content) > 0:
                mask_bytes = content
        except Exception:
            mask_bytes = None

    seed_val = seed if seed else None

    kwargs = {
        "prompt": prompt,
        "image_bytes": image_bytes,
        "mask_bytes": mask_bytes,
        "model_id": model_id,
        "strength": strength,
        "steps": steps,
        "seed": seed_val,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "allow_nsfw": allow_nsfw,
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
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=9000,
        reload=True,
        reload_excludes=["outputs/*", "outputs", "**/outputs/**"]
    )
