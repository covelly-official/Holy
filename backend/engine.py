import os
import threading
import time
import uuid
import gc
import queue
import multiprocessing as mp
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter

try:
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.common.resolution.config_resolution import ConfigResolution
    MFLUX_AVAILABLE = True
except ImportError:
    MFLUX_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set SAVE_DEBUG_OUTPUT=1 to also dump the raw (pre-composite) generation to disk
# on every call. Off by default now -- it was an unconditional disk write done
# on every single request, inside the model lock.
_SAVE_DEBUG_OUTPUT = os.environ.get("SAVE_DEBUG_OUTPUT", "0") == "1"

# ---------------------------------------------------------------------------
# Tunables for the Flux/mflux safety net. All overridable with env vars so
# you can tighten/loosen them without touching code. Defaults are chosen for
# a 16GB Apple Silicon machine (M3 Air) that also has to keep Cursor, a
# browser, etc. alive at the same time.
#
#   FLUX_MLX_MEMORY_LIMIT_GB  - soft cap we ask MLX's own allocator to
#                               respect. First line of defense.
#   FLUX_MAX_RSS_GB           - hard cap on the worker *process's* RSS. If
#                               MLX ignores the soft cap above, this is the
#                               backstop: we SIGKILL the worker the instant
#                               it crosses this line.
#   FLUX_MAX_SYSTEM_PERCENT   - hard cap on *system-wide* memory used%.
#                               Catches cases where the worker's own RSS
#                               looks fine but the whole Mac is still under
#                               pressure (compressed memory, other apps).
#   FLUX_MIN_AVAILABLE_GB     - hard floor on *system-wide* available memory.
#   FLUX_MAX_SECONDS          - wall-clock backstop in case a run just hangs.
#   FLUX_QUANTIZE_BITS        - "4", "8", or "0"/"none" to disable.
# ---------------------------------------------------------------------------
_FLUX_MLX_MEMORY_LIMIT_GB = float(os.environ.get("FLUX_MLX_MEMORY_LIMIT_GB", "8.0"))
_FLUX_MAX_RSS_GB = float(os.environ.get("FLUX_MAX_RSS_GB", "9.5"))
_FLUX_MAX_SYSTEM_PERCENT = float(os.environ.get("FLUX_MAX_SYSTEM_PERCENT", "95"))
_FLUX_MIN_AVAILABLE_GB = float(os.environ.get("FLUX_MIN_AVAILABLE_GB", "0.8"))
_FLUX_MAX_SECONDS = float(os.environ.get("FLUX_MAX_SECONDS", "1200"))
_FLUX_QUANTIZE_BITS = os.environ.get("FLUX_QUANTIZE_BITS", "4")
_FLUX_POLL_INTERVAL = 0.25

_model = None
_current_model_id = None
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


def _model_device_type(model) -> str:
    # MLX models (Flux2Klein etc.) don't expose a torch-style `.device.type`.
    # Fall back to "mps" (the only realistic device for mflux today) instead
    # of letting this throw.
    return getattr(getattr(model, "device", None), "type", "mps")


def _empty_cache(device: str):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# System/process memory readers. psutil is used when available; otherwise we
# fall back to zero-dependency macOS tools (`vm_stat`, `ps`) so the safety
# net still works even before you `pip install psutil`.
# ---------------------------------------------------------------------------
def _system_memory_percent():
    if _PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().percent
        except Exception:
            pass
    return None


def _system_memory_available_gb():
    if _PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass
    try:
        import subprocess
        import re
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        page_size = 4096
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            page_size = int(m.group(1))
        free_pages = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            m = re.search(rf"{re.escape(key)}:\s+(\d+)\.", out)
            if m:
                free_pages += int(m.group(1))
        return (free_pages * page_size) / (1024 ** 3)
    except Exception:
        return None


def _process_rss_gb(pid):
    if _PSUTIL_AVAILABLE:
        try:
            return psutil.Process(pid).memory_info().rss / (1024 ** 3)
        except Exception:
            return None
    try:
        import subprocess
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=2)
        return int(out.strip()) / (1024 ** 2)  # `ps` reports RSS in KB
    except Exception:
        return None


def _unload_timer_thread():
    global _model, _last_used_time
    while True:
        time.sleep(30)
        with _lock:
            if _model is not None and (time.time() - _last_used_time > _UNLOAD_TIMEOUT):
                print("Model has been idle for 5 minutes. Unloading to free memory...")
                device = _model_device_type(_model)
                _model = None
                _empty_cache(device)
                print("Memory freed successfully.")
        _flux_worker.maybe_idle_shutdown()


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


def load_model(model_id: str):
    """
    PyTorch/diffusers model loader. Unchanged from before, with one addition:
    if the Flux worker process is sitting warm in memory, we shut it down
    first. On a 16GB machine you can't comfortably keep both backends loaded
    at once, and there's no point letting an idle Flux worker hold several
    GB hostage while you're about to load a diffusion checkpoint.
    """
    global _model, _current_model_id

    if "flux" in model_id.lower() or "mflux" in model_id.lower():
        raise RuntimeError(
            "Flux/mflux models are no longer loaded through load_model() -- "
            "they run in an isolated worker process. Call generate() as usual; "
            "it routes Flux model_ids to the worker automatically."
        )

    if _flux_worker.is_alive():
        _flux_worker.shutdown(hard=False)

    if _model is not None and _current_model_id != model_id:
        print(f"Switching models from {_current_model_id} to {model_id}. Unloading old model...")
        device = _model_device_type(_model)
        _model = None
        _empty_cache(device)

    if _model is None:
        _current_model_id = model_id

        print(f"Loading PyTorch Model: {model_id}...")
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

        _model.vae.enable_tiling()
        _model.vae.enable_slicing()

        print(f"Model loaded successfully on {device}! (dtype: {pipeline_dtype})")

        # Load negative embeddings (e.g. bad-hands-5, badhandv4) if present
        emb_dir = Path(__file__).parent.parent / "models" / "embeddings"
        if emb_dir.exists():
            for emb_file in sorted(emb_dir.glob("*.pt")):
                token_name = emb_file.stem
                try:
                    _model.load_textual_inversion(str(emb_file), token=token_name)
                    print(f"Loaded textual inversion embedding: {token_name}")
                except Exception as e:
                    print(f"Could not load embedding {emb_file.name}: {e}")

    _ensure_watcher_started()
    return _model


# ---------------------------------------------------------------------------
# Flux / mflux: isolated worker process.
#
# Why a whole process instead of just running mflux in-thread like before:
# a single `pipe.generate_image(...)` call is one big blocking native call.
# Nothing inside this process -- no watchdog thread, no timeout, no flag --
# can interrupt it mid-flight or force it to give memory back. The old code's
# only "safety net" was whatever happened *after* that call returned, by
# which point the damage (swapping the whole Mac, freezing Cursor) was
# already done.
#
# Running it in a separate process fixes that: a watchdog in the main
# process can watch the worker's real memory use from the outside and
# SIGKILL it the instant it crosses a threshold. A killed process returns
# all of its memory to macOS immediately -- that's the only mechanism that
# can genuinely "terminate the model at any point."  The cost is that if we
# ever do have to kill it, the next Flux request pays a model-reload. That's
# the correct trade on a 16GB machine: a slower next request beats a frozen
# Mac.
# ---------------------------------------------------------------------------

def _set_mlx_memory_limit(limit_gb: float):
    # The exact API for capping MLX's memory has moved between mlx releases
    # (mx.metal.set_memory_limit -> mx.set_wired_limit / mx.set_cache_limit).
    # Try every spelling we know of; silently skip whichever don't exist on
    # the installed version rather than hard-failing.
    try:
        import mlx.core as mx
    except ImportError:
        return
    limit_bytes = int(limit_gb * 1024 ** 3)
    candidates = (
        ("set_memory_limit",),
        ("set_cache_limit",),
        ("set_wired_limit",),
        ("metal", "set_memory_limit"),
        ("metal", "set_cache_limit"),
    )
    for path in candidates:
        obj = mx
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        try:
            obj(limit_bytes, relaxed=True)
        except TypeError:
            try:
                obj(limit_bytes)
            except Exception:
                pass
        except Exception:
            pass


def _clear_mlx_cache():
    try:
        import mlx.core as mx
    except ImportError:
        return
    for path in (("clear_cache",), ("metal", "clear_cache")):
        obj = mx
        try:
            for attr in path:
                obj = getattr(obj, attr)
            obj()
            return
        except AttributeError:
            continue
        except Exception:
            return


def _flux_worker_main(cmd_q, result_q):
    """
    Entry point run inside the isolated worker process. Keeps its own mflux
    model loaded for as long as the process lives (so repeat requests stay
    fast); the parent kills/respawns this process to reclaim memory or to
    pick up a different model.
    """
    _set_mlx_memory_limit(_FLUX_MLX_MEMORY_LIMIT_GB)

    worker_model = {"pipe": None, "model_id": None}

    def _load(model_id):
        if worker_model["pipe"] is not None and worker_model["model_id"] == model_id:
            return worker_model["pipe"]

        if not MFLUX_AVAILABLE:
            raise RuntimeError("mflux is not installed. Run `pip install mflux` to use this model.")

        print(f"[flux-worker] Loading MLX Model: {model_id}...")

        mflux_path = None
        if any(k in model_id.lower() for k in ("flux2-klein-4b", "flux.2-klein-4b", "ar9av", "klein")):
            mflux_path = str(Path(__file__).parent.parent / "models" / "flux2-klein-4b-mlx")

        if mflux_path and not Path(mflux_path).exists():
            raise RuntimeError(
                f"Expected model files at {mflux_path} but nothing is there. "
                f"Did the `hf download` finish into this exact folder?"
            )

        model_config = ConfigResolution.resolve_restricted(
            "flux2-klein-4b",
            "flux2-klein-4b",
            model_path=mflux_path,
            extra_keys=(),
            base_model=None,
        )

        if mflux_path is not None:
            # mflux_path points at the known local download
            # (ar9av/FLUX.2-klein-4B-mflux-4bit, or whatever's at that exact
            # folder) -- those weights are already packed to 4-bit on disk.
            # Passing quantize=N here tells mflux to quantize *again* at
            # load time, which for an already-quantized checkpoint typically
            # means dequantizing back up to a working precision first and
            # requantizing -- a large, pointless memory spike right at load.
            # FLUX_QUANTIZE_BITS is ignored for this known-local-path case;
            # it only applies when loading some other, non-pre-quantized
            # model_id from the hub.
            quantize_val = None
        else:
            bits = _FLUX_QUANTIZE_BITS.strip().lower()
            quantize_val = None if bits in ("0", "none", "") else int(bits)

        pipe = Flux2Klein(
            model_config=model_config,
            quantize=quantize_val,
            model_path=mflux_path,
        )
        worker_model["pipe"] = pipe
        worker_model["model_id"] = model_id
        print(f"[flux-worker] MLX Model {model_id} loaded successfully! (quantize={quantize_val})")
        return pipe

    while True:
        msg = cmd_q.get()  # blocks until the parent sends something

        if msg["op"] == "shutdown":
            return

        if msg["op"] != "generate":
            continue

        try:
            pipe = _load(msg["model_id"])

            flux_steps = 4 if "klein" in msg["model_id"].lower() else msg["steps"]

            width = msg.get("width") or 512
            height = msg.get("height") or 512
            width = max(256, (int(width) // 16) * 16)
            height = max(256, (int(height) // 16) * 16)

            gen_kwargs = {
                "seed": msg["seed"],
                "prompt": msg["prompt"],
                "width": width,
                "height": height,
                "guidance": 1.0,
                "num_inference_steps": flux_steps,
                "scheduler": "flow_match_euler_discrete",
            }
            if msg.get("image_path"):
                gen_kwargs["image_path"] = msg["image_path"]
                gen_kwargs["image_strength"] = msg.get("strength", 0.8)

            result_img_mflux = pipe.generate_image(**gen_kwargs)

            if hasattr(result_img_mflux, "image"):
                img = result_img_mflux.image.convert("RGB")
            elif hasattr(result_img_mflux, "sq_image"):
                img = result_img_mflux.sq_image.convert("RGB")
            elif hasattr(result_img_mflux, "pil_image"):
                img = result_img_mflux.pil_image.convert("RGB")
            else:
                import tempfile
                out_fd, out_path = tempfile.mkstemp(suffix=".png")
                os.close(out_fd)
                try:
                    result_img_mflux.save(path=out_path, export_json_metadata=False)
                    real_path = out_path
                    if os.path.exists(out_path + ".png") and os.path.getsize(out_path) == 0:
                        real_path = out_path + ".png"
                    img = Image.open(real_path).convert("RGB")
                finally:
                    for p in (out_path, out_path + ".png"):
                        if os.path.exists(p):
                            os.unlink(p)

            buf = BytesIO()
            img.save(buf, format="PNG")
            result_q.put({"ok": True, "png_bytes": buf.getvalue()})

        except Exception as e:
            result_q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})

        finally:
            _clear_mlx_cache()
            gc.collect()


class _FluxWorkerManager:
    """
    Owns the isolated Flux/mflux worker process from the parent side: starts
    it lazily, keeps it warm between requests, watches its memory while a
    job is running, and hard-kills it the moment that memory looks
    dangerous (see the module docstring above for why it has to be a kill,
    not a softer signal).
    """

    def __init__(self):
        self._ctx = mp.get_context("spawn")  # fork + Metal/GPU contexts is unsafe
        self.process = None
        self.cmd_q = None
        self.result_q = None
        self.last_used = time.time()
        self.lock = threading.Lock()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def ensure_started(self):
        if self.is_alive():
            return
        _unload_diffusers_model()  # no-op if nothing is loaded
        self.cmd_q = self._ctx.Queue()
        self.result_q = self._ctx.Queue()
        self.process = self._ctx.Process(
            target=_flux_worker_main, args=(self.cmd_q, self.result_q), daemon=True
        )
        self.process.start()
        _ensure_watcher_started()

    def shutdown(self, hard: bool):
        if not self.is_alive():
            self.process = None
            return
        if hard:
            self.process.kill()
        else:
            try:
                self.cmd_q.put({"op": "shutdown"})
            except Exception:
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
        self.process.join(timeout=5)
        self.process = None
        print("[flux-worker] process stopped, memory released.")

    def maybe_idle_shutdown(self):
        # Called every 30s from the shared watcher thread. Skips the check
        # (rather than blocking) if a generation is actively in flight.
        if not self.is_alive():
            return
        if (time.time() - self.last_used) <= _UNLOAD_TIMEOUT:
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.is_alive() and (time.time() - self.last_used) > _UNLOAD_TIMEOUT:
                print("[flux-worker] idle for 5 minutes, shutting down to free memory...")
                self.shutdown(hard=False)
        finally:
            self.lock.release()

    def generate(self, **params) -> dict:
        global _abort_flag
        with self.lock:
            self.last_used = time.time()
            self.ensure_started()
            self.cmd_q.put({"op": "generate", **params})

            start = time.time()
            while True:
                try:
                    result = self.result_q.get(timeout=_FLUX_POLL_INTERVAL)
                    self.last_used = time.time()
                    return result
                except queue.Empty:
                    pass

                if not self.process.is_alive():
                    exitcode = self.process.exitcode
                    self.process = None
                    return {
                        "ok": False,
                        "error": (
                            f"Flux worker process exited unexpectedly (exit code {exitcode}). "
                            "This usually means macOS itself killed it for using too much memory."
                        ),
                    }

                if _abort_flag:
                    self.shutdown(hard=True)
                    return {"ok": False, "aborted": True}

                rss_gb = _process_rss_gb(self.process.pid)
                sys_percent = _system_memory_percent()
                sys_avail_gb = _system_memory_available_gb()
                elapsed = time.time() - start

                over_rss = rss_gb is not None and rss_gb > _FLUX_MAX_RSS_GB
                over_percent = sys_percent is not None and sys_percent > _FLUX_MAX_SYSTEM_PERCENT
                under_avail = sys_avail_gb is not None and sys_avail_gb < _FLUX_MIN_AVAILABLE_GB
                timed_out = elapsed > _FLUX_MAX_SECONDS

                if over_rss or over_percent or under_avail or timed_out:
                    if over_rss:
                        reason = f"worker RSS {rss_gb:.1f}GB exceeded the {_FLUX_MAX_RSS_GB:.1f}GB cap"
                    elif over_percent:
                        reason = f"system memory at {sys_percent:.0f}%, over the {_FLUX_MAX_SYSTEM_PERCENT:.0f}% cap"
                    elif under_avail:
                        reason = f"only {sys_avail_gb:.1f}GB available, under the {_FLUX_MIN_AVAILABLE_GB:.1f}GB floor"
                    else:
                        reason = f"generation ran past the {_FLUX_MAX_SECONDS:.0f}s safety timeout"
                    print(f"[flux-worker] SAFETY KILL: {reason}. Terminating worker to protect the Mac.")
                    self.shutdown(hard=True)
                    return {
                        "ok": False,
                        "error": f"Stopped to protect your Mac's memory ({reason}). Try again; "
                                 f"consider lowering steps or image size.",
                    }


def _unload_diffusers_model():
    global _model
    with _lock:
        if _model is not None:
            print("[flux-worker] unloading PyTorch model to free RAM for Flux on this machine...")
            device = _model_device_type(_model)
            _model = None
            _empty_cache(device)


_flux_worker = _FluxWorkerManager()


def _preprocess(image_bytes: bytes, mask_bytes: bytes):
    """Shared by both backends: load, pad-crop to a multiple of 8, downscale
    to max_size, pad-crop again, and binarize the mask. Identical to what the
    original single generate() function did inline."""
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

    mask_image = mask_image.point(lambda p: 255 if p > 127 else 0)
    mask_image_full_res = mask_image_full_res.point(lambda p: 255 if p > 127 else 0)

    return init_image_full_res, mask_image_full_res, init_image, mask_image, original_size_cropped


def _finish(result_img, init_image_full_res, mask_image_full_res, original_size_cropped, seed, prompt):
    result_img = result_img.resize(original_size_cropped, Image.LANCZOS)

    soft_mask = mask_image_full_res.filter(ImageFilter.GaussianBlur(4))
    result_img = Image.composite(result_img, init_image_full_res, soft_mask)

    name = f"{int(time.time())}_{seed}_{uuid.uuid4().hex[:6]}.png"
    result_img.save(str(OUTPUT_DIR / name))

    del result_img

    return {"file": name, "seed": seed, "prompt": prompt}


def _generate_flux(prompt, image_bytes, mask_bytes, model_id, strength, steps, seed,
                   aspect_ratio="original", width=None, height=None):
    global _last_used_time, _current_progress, _abort_flag

    _last_used_time = time.time()
    has_image = image_bytes is not None and len(image_bytes) > 0

    if has_image:
        init_image_full_res, mask_image_full_res, init_image, mask_image, original_size_cropped = \
            _preprocess(image_bytes, mask_bytes)
        w, h = init_image.size

        actual_steps = int(steps * strength)
        flux_steps = 4 if "klein" in model_id.lower() else actual_steps
        _current_progress["step"] = 0
        _current_progress["total_steps"] = flux_steps

        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        init_image.save(temp_path)

        try:
            result = _flux_worker.generate(
                model_id=model_id,
                prompt=prompt,
                seed=seed,
                width=w,
                height=h,
                image_path=temp_path,
                steps=flux_steps,
                strength=strength,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if result.get("aborted"):
            return {"status": "aborted"}
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Flux generation failed."))

        result_img = Image.open(BytesIO(result["png_bytes"])).convert("RGB")

        if _SAVE_DEBUG_OUTPUT:
            result_img.save(str(OUTPUT_DIR / "debug_raw_output.png"))

        if _abort_flag:
            return {"status": "aborted"}

        return _finish(result_img, init_image_full_res, mask_image_full_res, original_size_cropped, seed, prompt)
    else:
        # PURE TEXT-TO-IMAGE
        if width and height:
            w, h = int(width), int(height)
        elif aspect_ratio == "square":
            w, h = 512, 512
        elif aspect_ratio == "portrait":
            w, h = 512, 768
        elif aspect_ratio == "landscape":
            w, h = 768, 512
        else:
            w, h = 512, 512

        flux_steps = 4 if "klein" in model_id.lower() else steps
        _current_progress["step"] = 0
        _current_progress["total_steps"] = flux_steps

        result = _flux_worker.generate(
            model_id=model_id,
            prompt=prompt,
            seed=seed,
            width=w,
            height=h,
            image_path=None,
            steps=flux_steps,
            strength=strength,
        )

        if result.get("aborted"):
            return {"status": "aborted"}
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Flux generation failed."))

        result_img = Image.open(BytesIO(result["png_bytes"])).convert("RGB")

        if _SAVE_DEBUG_OUTPUT:
            result_img.save(str(OUTPUT_DIR / "debug_raw_output.png"))

        if _abort_flag:
            return {"status": "aborted"}

        name = f"{int(time.time())}_{seed}_{uuid.uuid4().hex[:6]}.png"
        result_img.save(str(OUTPUT_DIR / name))
        return {"file": name, "seed": seed, "prompt": prompt, "is_txt2img": True}


def _generate_diffusers(prompt, image_bytes, mask_bytes, model_id, negative_prompt, strength, steps, seed):
    global _last_used_time, _current_progress, _abort_flag

    with _lock:
        _last_used_time = time.time()
        pipe = load_model(model_id)
        device = _model_device_type(pipe)

        init_image_full_res, mask_image_full_res, init_image, mask_image, original_size_cropped = \
            _preprocess(image_bytes, mask_bytes)
        w, h = init_image.size

        actual_steps = int(steps * strength)
        _current_progress["step"] = 0
        _current_progress["total_steps"] = actual_steps

        generator = torch.Generator("cpu").manual_seed(seed)

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

    return _finish(result_img, init_image_full_res, mask_image_full_res, original_size_cropped, seed, prompt)


def generate(prompt: str, image_bytes: bytes | None = None, mask_bytes: bytes | None = None,
             model_id: str = "Uminosachi/realisticVisionV51_v51VAE-inpainting",
             negative_prompt: str = "bad-hands-5, badhandv4, (extra fingers:1.2), (missing fingers:1.2), (fused fingers:1.2), deformed, blurry, bad anatomy, low quality",
             strength: float = 0.8, steps: int = 30, seed: int | None = None,
             aspect_ratio: str = "original", width: int | None = None, height: int | None = None,
             allow_nsfw: bool = True) -> dict:

    global _abort_flag
    _abort_flag = False
    if seed is None:
        seed = int(time.time()) % (2**31)

    print(f"[Holy Engine] Generation: allow_nsfw={allow_nsfw}, prompt='{prompt}'")

    # When NSFW is disabled (Safe Mode), inject strong anti-nudity and safety tokens into negative prompt
    if not allow_nsfw:
        safe_negatives = "nsfw, nude, naked, nudity, nipples, breasts, sexual, explicit, pornography, topless, erotic, genitals, adult content"
        if negative_prompt:
            negative_prompt = f"{negative_prompt}, {safe_negatives}"
        else:
            negative_prompt = safe_negatives
        print(f"[Holy Engine] Safe Mode ACTIVE: Injected safety negatives.")
    else:
        print(f"[Holy Engine] Uncensored Mode ACTIVE: Safety filters bypassed.")

    is_flux = "flux" in model_id.lower() or "mflux" in model_id.lower()

    if is_flux:
        return _generate_flux(
            prompt=prompt,
            image_bytes=image_bytes,
            mask_bytes=mask_bytes,
            model_id=model_id,
            strength=strength,
            steps=steps,
            seed=seed,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
        )
    else:
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError(
                "Realistic Vision requires an uploaded photo for inpainting. "
                "Switch to FLUX.2 Klein for Text-to-Image generation!"
            )
        return _generate_diffusers(prompt, image_bytes, mask_bytes, model_id, negative_prompt, strength, steps, seed)