# Holy Virtual Try-On Studio - Troubleshooting Handbook

This document logs all the technical issues, crashes, and bugs encountered while building the Apple Silicon (MPS) Inpainting pipeline for the Holy Virtual Try-On app.

### 1. Zombie Server Processes (Address already in use)
**Issue:** `[Errno 48] error while attempting to bind on address ('127.0.0.1', 8000): address already in use`
**What Caused It:** A previous Uvicorn backend process was orphaned or crashed into the background without freeing up the port it was listening on.
**What Fixed It:** First, forcefully killed the zombie process using the terminal (`lsof -i :8000` -> `kill -9`), and subsequently migrated the frontend and backend to use port `9000` to avoid conflicts entirely.
**Note for Future:** If a port is mysteriously blocked on macOS, there is almost always a hidden background process holding it. Use `pkill -9 -f uvicorn` to clear all Python servers.

### 2. Hugging Face Download Freezing
**Issue:** Terminal model download speed plummeted to `1.94kB/s` and stalled indefinitely at 80%.
**What Caused It:** When downloading multi-gigabyte models without authentication, Hugging Face Hub servers occasionally throttle or drop the CDN connection.
**What Fixed It:** Terminated the frozen server (`Ctrl+C`) and restarted it. The Hugging Face local cache system is smart and automatically resumed the download exactly from 80%.
**Note for Future:** Never delete the cache if a download freezes. Just restart the script. If throttling happens repeatedly, create a free Hugging Face token and run `huggingface-cli login` in the terminal to remove rate limits.

### 3. NSFW Filter False Positives (Black Image Output)
**Issue:** The AI generated successfully, but the UI displayed a completely black square. Terminal showed: `Potential NSFW content was detected in one or more images.`
**What Caused It:** The default `diffusers` safety checker is notoriously over-aggressive, especially on highly detailed models. It falsely flagged safe test images (like a donkey) and standard skin exposures (necks/arms) as explicit content.
**What Fixed It:** Explicitly disabled the filter when loading the model by passing `safety_checker=None` and `requires_safety_checker=False` to `AutoPipelineForInpainting.from_pretrained`.
**Note for Future:** Always disable the safety checker when building and testing local generative AI tools to avoid frustrating false positives.

### 4. Mac MPS Mathematical Overflow (Black Image / NaNs)
**Issue:** Even after the safety checker was removed, the AI still outputted black images. Terminal showed: `RuntimeWarning: invalid value encountered in cast` (NaNs).
**What Caused It:** Apple Silicon's Metal Performance Shaders (MPS) have a hardware/driver bug where `float16` (half-precision) cross-attention math in the UNet and VAE decoding overflows its limits, corrupting the pixels into `NaN` (Not a Number).
**What Fixed It:** Abandoned memory-saving `float16` hacks and brute-forced the entire pipeline to load and execute in `torch.float32` (standard precision). Because the M3 Mac has 16GB of unified memory, it easily absorbed the ~4GB RAM requirement of `float32` without crashing.
**Note for Future:** Never trust `float16` on Mac MPS for older Stable Diffusion 1.5 pipelines. If you have 16GB+ of RAM, always default to `float32` to guarantee stable math and prevent silent black-image corruption.

### 5. PyTorch Dtype Mismatch (c10::Half vs float)
**Issue:** App crashed instantly on generation with `RuntimeError: Input type (c10::Half) and bias type (float) should be the same`.
**What Caused It:** During an attempt to fix the NaNs (Issue 4), only the VAE was cast to `float32` while the rest of the pipeline remained in `float16`. The `diffusers` inpainting pipeline fed a `float16` (c10::Half) image into the VAE encoder, which now only accepted `float32`.
**What Fixed It:** Removed the partial VAE casting and explicitly cast the *entire* pipeline uniformly using `_model = _model.to("mps", dtype=torch.float32)`.
**Note for Future:** Never mix `float16` and `float32` submodules within a single PyTorch pipeline unless you are explicitly writing the intermediate tensor casting code yourself. 

### 6. Gradient Tracking Crash (Can't call numpy() on Tensor)
**Issue:** App crashed with `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`
**What Caused It:** During a manual VAE decode hack, we called `pipe.vae.decode()` directly. Unlike the main `pipe()` call which disables training math, calling the sub-module directly caused PyTorch to track gradients, preventing the tensor from being exported to a static PIL image.
**What Fixed It:** Wrapped the manual decode step in a `with torch.no_grad():` block and appended `.detach()` to the output tensor. (This code was later removed when the pipeline was upgraded to full `float32`).
**Note for Future:** Whenever you bypass a high-level library API to manipulate raw neural network components (like Encoders/Decoders) for inference, always wrap your code in `torch.no_grad()` to prevent memory leaks and gradient errors.

### 7. FastAPI Progress Bar Freezing & Unresponsive Cancel Button
**Issue:** Frontend UI progress bar stayed stuck at "Synthesizing..." and the Cancel button did absolutely nothing during generation.
**What Caused It:** The backend FastAPI route for `/generate` was an `async def` function, but it executed `engine.generate()`, a heavy, synchronous PyTorch model that hijacked the Python Global Interpreter Lock (GIL). Because it ran on the main thread, it blocked the entire FastAPI asyncio event loop, causing `/progress` and `/abort` requests to pile up and time out until generation finished.
**What Fixed It:** 
1. Used Starlette's `run_in_threadpool` to offload the heavy `engine.generate()` call to a background worker thread.
2. Injected a microscopic `time.sleep(0.001)` in PyTorch's `callback_on_step_end` loop to physically force PyTorch to release the GIL for 1 millisecond every step, allowing the main thread to instantly intercept abort commands and serve progress JSON.
**Note for Future:** Never run synchronous ML inferences directly inside an `async def` FastAPI endpoint. Always offload them to a threadpool.

### 8. Conda vs Homebrew Environment Clashes
**Issue:** `python app.py` threw `ModuleNotFoundError: No module named 'fastapi'` even after activating the `(holy)` conda environment.
**What Caused It:** The terminal's `$PATH` was scrambled. While `pip` mapped to the conda environment (`/opt/miniconda3/envs/holy`), running `python` unexpectedly mapped to the global Homebrew installation (`/opt/homebrew/...`). Thus, the dependencies were installed in one environment, but the script ran in a completely different one.
**What Fixed It:** Reverted to using `uvicorn app:app --port 9000`. The global `uvicorn` binary correctly resolved its own python path context, entirely bypassing the broken `python` mapping.
**Note for Future:** When a basic `import` fails in a conda environment despite a successful `pip install`, run `which python` and `which pip` to instantly verify if they point to the exact same directory tree.

### 9. Hard Seam Artifacts on Inpainted Images
**Issue:** AI inpainting slightly altered the color and noise profile of the original pixels *outside* the masked area, creating an ugly invisible square seam around the edited zone.
**What Caused It:** VAE compression and decompression inherently modifies pixels slightly. When the model stitches the output back into the unmasked image, the 32x32 tensor chunking leaves a visible boundary.
**What Fixed It:** Added manual PIL compositing post-generation: `Image.composite(generated, original, soft_mask)`. Used a Gaussian Blur feather of `4` on the mask edges. This perfectly preserves all original unmasked pixels and smoothly blends the AI output only inside the feathered region.
**Note for Future:** Always use feathered composite blending when building professional inpainting/outpainting tools. Never rely exclusively on the model's native tensor output for untouched regions.

### 10. Browser Timeout on Long-Running API Requests (Double Generation)
**Issue:** After the progress bar hit 100%, the backend suddenly started generating a completely new image from 1/38 without the user clicking the button again.
**What Caused It:** The 32-bit generation was taking 4–5 minutes on the Mac. Web browsers (Chrome/Safari) have built-in timeout limits (usually ~2-3 minutes) for pending HTTP requests. When the browser hit its timeout, it assumed the connection dropped and silently retried the POST `/generate` request in the background. As soon as the backend finished the first run, it immediately processed the browser's hidden retry request.
**What Fixed It:** In standard architecture, long-running ML tasks should return a `task_id` instantly (async pattern) rather than blocking the HTTP request. However, by brutally optimizing the model's memory footprint and resolution (Issue 11), generation times dropped under the browser timeout limit, mitigating the issue for now.
**Note for Future:** Never hold an HTTP POST request open for longer than 60 seconds. Always implement a Task Queue (e.g., Celery) or an async Job/Polling pattern for heavy generative AI endpoints.

### 11. Extreme Memory Swapping on Apple Silicon (SSD Thrashing)
**Issue:** During an attempt to use `float16`, generation slowed to a crawl (from 6 seconds/iteration down to 19 seconds/iteration) and the entire Mac OS locked up and lagged heavily.
**What Caused It:** To prevent `float16` NaNs on MPS, `_model.enable_attention_slicing()` was removed. Without attention slicing, the UNet's attention maps fully materialized in memory during the forward pass. This massive memory spike exceeded the Mac's physical Unified Memory. To survive, macOS brutally swapped memory to the solid-state drive (SSD thrashing), causing catastrophic generation slowdowns and UI lockups.
**What Fixed It:** Abandoned the `float16` experiment entirely. Reverted to pure `float32` and re-enabled maximum memory slicing: `_model.enable_attention_slicing(1)`. Added `torch.mps.set_per_process_memory_fraction(0.6)` to strictly cap the Python process to 60% of the Mac's RAM, ensuring the OS never has to swap to SSD again.
**Note for Future:** On Apple Silicon, peak *activation memory* (memory used temporarily during a single layer's computation) is far more dangerous than model weight memory. Always use `enable_attention_slicing(1)` to slice attention maps into smaller blocks and keep the activation memory footprint tiny.
