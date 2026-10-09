import threading
import time
import uuid
import gc
from io import BytesIO
from pathlib import Path
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_model = None
_lock = threading.Lock()
_last_used_time = time.time()
_UNLOAD_TIMEOUT = 300  # 5 minutes
_current_progress = {"step": 0, "total_steps": 1}
_abort_flag = False


def abort_generation():
    global _abort_flag
    _abort_flag = True


def _unload_timer_thread():
    global _model, _last_used_time
    while True:
        time.sleep(30)
        with _lock:
            if _model is not None and (time.time() - _last_used_time > _UNLOAD_TIMEOUT):
                print("Model has been idle for 5 minutes. Unloading to free memory...")
                _model = None
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Memory freed successfully.")


threading.Thread(target=_unload_timer_thread, daemon=True).start()


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    global _model
    if _model is None:
        print("Loading Realistic Vision V5.1 Inpainting Model...")
        model_id = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
        device = _get_device()

        _model = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        _model = _model.to(device)
        _model.enable_attention_slicing()
        _model.vae.enable_slicing()  # extra memory headroom, updated to new API
        print(f"Model loaded successfully on {device}!")
    return _model


def generate(prompt: str, image_bytes: bytes, mask_bytes: bytes,
             negative_prompt: str = "deformed, blurry, bad anatomy, low quality",
             strength: float = 0.8, steps: int = 30, seed: int | None = None) -> dict:

    global _last_used_time, _current_progress, _abort_flag
    _abort_flag = False
    if seed is None:
        seed = int(time.time()) % (2**31)

    with _lock:
        _last_used_time = time.time()
        pipe = load_model()
        device = pipe.device.type

        # Load full resolution images
        init_image_full_res = Image.open(BytesIO(image_bytes)).convert("RGB")
        mask_image_full_res = Image.open(BytesIO(mask_bytes)).convert("L")

        # First, ensure the full resolution is a multiple of 8 so we don't have alignment issues later
        w, h = init_image_full_res.size
        new_w, new_h = (w // 8) * 8, (h // 8) * 8
        if new_w != w or new_h != h:
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            init_image_full_res = init_image_full_res.crop((left, top, left + new_w, top + new_h))
            mask_image_full_res = mask_image_full_res.crop((left, top, left + new_w, top + new_h))
            
        original_size_cropped = init_image_full_res.size

        # Create working copies for the AI
        init_image = init_image_full_res.copy()
        mask_image = mask_image_full_res.copy()

        # MASSIVE FIX: Cap AI generation at 768px. 1024px in float32 consumes 8GB+ of RAM 
        # and violently crashes 16GB Macs due to swap memory thrashing.
        max_size = 768
        if init_image.width > max_size or init_image.height > max_size:
            init_image.thumbnail((max_size, max_size), Image.LANCZOS)
            mask_image = mask_image.resize(init_image.size, Image.NEAREST)

        w, h = init_image.size
        new_w, new_h = (w // 8) * 8, (h // 8) * 8
        if new_w != w or new_h != h:
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            init_image = init_image.crop((left, top, left + new_w, top + new_h))
            mask_image = mask_image.crop((left, top, left + new_w, top + new_h))
            
        w, h = init_image.size
        # re-binarize in case of any interpolation drift
        mask_image = mask_image.point(lambda p: 255 if p > 127 else 0)
        mask_image_full_res = mask_image_full_res.point(lambda p: 255 if p > 127 else 0)

        generator = torch.Generator(device).manual_seed(seed)

        actual_steps = int(steps * strength)
        _current_progress["step"] = 0
        _current_progress["total_steps"] = actual_steps

        def step_callback(pipe, i, t, callback_kwargs):
            global _abort_flag
            if _abort_flag:
                pipe._interrupt = True
            _current_progress["step"] = i + 1
            time.sleep(0.001)  # Force GIL release so FastAPI can breathe!
            return callback_kwargs

        result_img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            height=h,
            width=w,
            strength=strength,
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=step_callback,
        ).images[0]
        
        # DEBUG: Save the raw, un-composited, low-res output from the AI
        result_img.save(str(OUTPUT_DIR / "debug_raw_output.png"))

        # FIX: when _interrupt stops the pipeline early, pipe() still returns
        # normally with a (partial/garbage) image — without this check the
        # caller has no way to know the run was aborted rather than completed
        if _abort_flag:
            return {"status": "aborted"}

    # Scale the AI output back up to the original un-shrunk resolution
    result_img = result_img.resize(original_size_cropped, Image.LANCZOS)

    # Composite the generated image back onto the original full-res image using the full-res mask
    # This guarantees zero quality loss for pixels outside the mask!
    soft_mask = mask_image_full_res.filter(ImageFilter.GaussianBlur(4))
    result_img = Image.composite(result_img, init_image_full_res, soft_mask)

    name = f"{int(time.time())}_{seed}_{uuid.uuid4().hex[:6]}.png"
    result_img.save(str(OUTPUT_DIR / name))

    return {"file": name, "seed": seed, "prompt": prompt}