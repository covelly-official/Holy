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

_model = None
_lock = threading.Lock()
_last_used_time = time.time()
_UNLOAD_TIMEOUT = 300  # 5 minutes
_current_progress = {"step": 0, "total_steps": 1}
_abort_flag = False


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


threading.Thread(target=_unload_timer_thread, daemon=True).start()


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
        # Cast the latent output back to Half
        res.latent_dist.sample = lambda *a, **k: res.latent_dist.sample_raw(*a, **k).to(torch.float16)
        res.latent_dist.sample_raw = res.latent_dist.sample
        res.latent_dist.mean = res.latent_dist.mean.to(torch.float16)
        res.latent_dist.std = res.latent_dist.std.to(torch.float16)
        return res

    def decode(self, z, *args, **kwargs):
        # Pipeline sends Half; we cast to Float for the VAE
        z = z.to(torch.float32)
        res = self.vae.decode(z, *args, **kwargs)
        # Cast the image output back to Half
        res.sample = res.sample.to(torch.float16)
        return res


def load_model():
    global _model
    if _model is None:
        print("Loading Realistic Vision V5.1 Inpainting Model...")
        model_id = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
        device = _get_device()

        _model = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        _model = _model.to(device)

        # Wrap the VAE in our translator to avoid MPS NaNs AND type mismatches
        _model.vae = VAEUpcastWrapper(_model.vae)

        # We no longer need to restrict memory or use max slicing because
        # float16 uses massively less RAM than float32.
        _model.enable_attention_slicing()
        _model.vae.enable_tiling()
        _model.vae.enable_slicing()

        print(f"Model loaded successfully on {device}! (fp16 with fp32 VAE Wrapper)")
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
        device = _get_device()
        _empty_cache(device)  # clear any fragmentation left from a previous run
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

        result_img.save(str(OUTPUT_DIR / "debug_raw_output.png"))

        if _abort_flag:
            return {"status": "aborted"}

    result_img = result_img.resize(original_size_cropped, Image.LANCZOS)

    soft_mask = mask_image_full_res.filter(ImageFilter.GaussianBlur(4))
    result_img = Image.composite(result_img, init_image_full_res, soft_mask)

    name = f"{int(time.time())}_{seed}_{uuid.uuid4().hex[:6]}.png"
    result_img.save(str(OUTPUT_DIR / name))

    del result_img, init_image_full_res, mask_image_full_res, init_image, mask_image
    _empty_cache(device)

    return {"file": name, "seed": seed, "prompt": prompt}