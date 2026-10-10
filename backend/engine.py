import os
import threading
import time
import uuid
import gc
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set SAVE_DEBUG_OUTPUT=1 to also dump the raw (pre-composite) generation to disk
# on every call. Off by default now -- it was an unconditional disk write done
# on every single request, inside the model lock.
_SAVE_DEBUG_OUTPUT = os.environ.get("SAVE_DEBUG_OUTPUT", "0") == "1"

_model = None
_lock = threading.Lock()
_last_used_time = time.time()
_UNLOAD_TIMEOUT = 300  # 5 minutes
_current_progress = {"step": 0, "total_steps": 1}
_abort_flag = False

_watcher_started = False
_watcher_start_lock = threading.Lock()


def abort_generation():
    global _abort_flag
    _abort_flag = True


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_cache(device: str):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _unload_timer_thread():
    global _model, _last_used_time
    while True:
        time.sleep(30)
        with _lock:
            if _model is not None and (time.time() - _last_used_time > _UNLOAD_TIMEOUT):
                print("Model has been idle for 5 minutes. Unloading to free memory...")
                device = _model.device.type
                _model = None
                _empty_cache(device)
                print("Memory freed successfully.")


def _ensure_watcher_started():
    # Started lazily (on first model load) and guarded, instead of unconditionally
    # at import time. Under an auto-reloading dev server (uvicorn --reload, etc.)
    # the old module-level start() could stack up a new polling thread on every
    # reload, each one independently acquiring the lock and calling gc.collect()
    # every 30s -- a real, silent resource leak during development.
    global _watcher_started
    if _watcher_started:
        return
    with _watcher_start_lock:
        if not _watcher_started:
            threading.Thread(target=_unload_timer_thread, daemon=True).start()
            _watcher_started = True


class _Float16LatentDist:
    """
    Forwards to a float32 latent distribution but returns float16 samples/stats.

    This replaces the old approach of monkey-patching `sample`/`mean`/`std`
    directly onto `res.latent_dist` (the same object the closure captured).
    That created a reference cycle -- the object held a bound method / closure
    that pointed straight back at itself -- on every single VAE encode() call.
    CPython's refcounter can't free cycles; only a full gc.collect() can, which
    is why the original code needed one after every generation just to keep
    memory from creeping up. This wrapper is a plain forward reference
    (res -> wrapper -> real dist), so it's freed immediately when dereferenced.
    """
    __slots__ = ("_dist",)

    def __init__(self, dist):
        self._dist = dist

    def sample(self, *args, **kwargs):
        return self._dist.sample(*args, **kwargs).to(torch.float16)

    def __getattr__(self, name):
        value = getattr(self._dist, name)
        if name in ("mean", "std") and torch.is_tensor(value):
            return value.to(torch.float16)
        return value


class VAEUpcastWrapper:
    """
    Wraps a diffusers VAE. Intercepts float16 inputs from the pipeline,
    casts them to float32 so the VAE can process them safely (avoiding MPS NaNs),
    and casts the results back to float16 so the pipeline doesn't crash with a type mismatch.
    """
    def __init__(self, vae):
        self.vae = vae
        # The underlying VAE is converted to float32
        self.vae.to(dtype=torch.float32)

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, x, *args, **kwargs):
        # Pipeline sends Half; we cast to Float for the VAE
        x = x.to(torch.float32)
        res = self.vae.encode(x, *args, **kwargs)
        # Cast the latent output back to Half without creating a reference cycle
        res.latent_dist = _Float16LatentDist(res.latent_dist)
        return res

    def decode(self, z, *args, **kwargs):
        # Pipeline sends Half; we cast to Float for the VAE
        z = z.to(torch.float32)
        res = self.vae.decode(z, *args, **kwargs)
        # Cast the image output back to Half
        if isinstance(res, tuple):
            return tuple(r.to(torch.float16) if hasattr(r, 'to') else r for r in res)
        else:
            res.sample = res.sample.to(torch.float16)
            return res


def load_model():
    global _model
    if _model is None:
        print("Loading Realistic Vision V5.1 Inpainting Model...")
        model_id = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
        device = _get_device()

        # CPU does not support HalfTensor operations well. Fallback to float32 if no GPU.
        pipeline_dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

        if model_id.endswith(".safetensors"):
            _model = AutoPipelineForInpainting.from_single_file(
                model_id,
                torch_dtype=pipeline_dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            # Loading an fp16 variant (when the repo publishes one) means the fp16
            # weights are downloaded and materialized directly -- no intermediate
            # fp32 copy in RAM that then gets cast down. That fp32->fp16 cast on
            # load is the single biggest RAM spike in this whole file, and the
            # most likely reason the very first run took the machine down.
            # low_cpu_mem_usage streams weights in rather than fully materializing
            # before moving to device. Falls back cleanly if no fp16 variant exists.
            try:
                _model = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=pipeline_dtype,
                    use_safetensors=True,
                    variant="fp16" if pipeline_dtype == torch.float16 else None,
                    low_cpu_mem_usage=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            except Exception as e:
                print(f"fp16 variant not available ({e}); loading default weights.")
                _model = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=pipeline_dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
        _model = _model.to(device)

        # Wrap the VAE to avoid NaNs only if we are running the UNet in float16
        if pipeline_dtype == torch.float16:
            _model.vae = VAEUpcastWrapper(_model.vae)

        # float16 uses massively less RAM than float32.
        # DO NOT enable attention slicing on MPS! It forces mathematical attention which accumulates in fp16 and causes NaNs.
        # Letting it use PyTorch 2.0 SDPA automatically accumulates in fp32 and prevents the NaN!
        _model.vae.enable_tiling()
        _model.vae.enable_slicing()

        print(f"Model loaded successfully on {device}! (dtype: {pipeline_dtype})")

    _ensure_watcher_started()
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

        init_image_full_res = Image.open(BytesIO(image_bytes)).convert("RGB")
        mask_image_full_res = Image.open(BytesIO(mask_bytes)).convert("L")

        w, h = init_image_full_res.size
        new_w, new_h = (w // 8) * 8, (h // 8) * 8
        if new_w != w or new_h != h:
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            init_image_full_res = init_image_full_res.crop((left, top, left + new_w, top + new_h))
            mask_image_full_res = mask_image_full_res.crop((left, top, left + new_w, top + new_h))

        original_size_cropped = init_image_full_res.size

        init_image = init_image_full_res.copy()
        mask_image = mask_image_full_res.copy()

        # Start conservative. Once this is stable, raise toward 768.
        max_size = 512
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
        mask_image = mask_image.point(lambda p: 255 if p > 127 else 0)
        mask_image_full_res = mask_image_full_res.point(lambda p: 255 if p > 127 else 0)

        generator = torch.Generator("cpu").manual_seed(seed)

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

        try:
            with torch.inference_mode():
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
        finally:
            _empty_cache(device)

        if np.asarray(result_img).std() < 1e-3:
            raise RuntimeError(
                "Generation produced a blank image — not expected in fp32. "
                "Check prompt/mask inputs if this happens."
            )

        if _SAVE_DEBUG_OUTPUT:
            result_img.save(str(OUTPUT_DIR / "debug_raw_output.png"))

        if _abort_flag:
            return {"status": "aborted"}

    result_img = result_img.resize(original_size_cropped, Image.LANCZOS)

    soft_mask = mask_image_full_res.filter(ImageFilter.GaussianBlur(4))
    result_img = Image.composite(result_img, init_image_full_res, soft_mask)

    name = f"{int(time.time())}_{seed}_{uuid.uuid4().hex[:6]}.png"
    result_img.save(str(OUTPUT_DIR / name))

    del result_img, init_image_full_res, mask_image_full_res, init_image, mask_image

    return {"file": name, "seed": seed, "prompt": prompt}