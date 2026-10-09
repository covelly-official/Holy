# Project Scope: Holy (Full AI Image Editor)

> Status: DRAFT v3. (Pivoted to full AI Editor suite)

## 1. What this is

Holy is a powerful, local-first **AI Image Editor** (an "AI Photoshop"). The core contract is:

**Input: one image (or more) + a text prompt + optional mask → Output: one edited image.**

Instead of just Virtual Try-On, it acts as a comprehensive local studio for Generative Fill, Object Removal, Background Replacement, and Image-to-Image styling. Everything runs locally on the user's own Mac. No cloud, no subscriptions, no privacy leaks.

## 2. Core behavior (the contract)

- The primary mode is **image + prompt → image** (image-conditioned editing).
- The user supplies a reference image and an instruction ("change the jacket to red", "make it evening light", "add glasses").
- Optional: a second reference image (for example, an object or garment to apply to the first).
- The output is always a new image. The input file is never modified.
- Every result stores its input image, prompt, seed and settings, so it can be reproduced.
- **[TBD]** Whether to also offer plain text-to-image (no input image) as a secondary mode.

## 3. Goals

- Preserve the subject's identity, pose and anatomy during edits (faces, hands, proportions).
- Run fully offline on Apple Silicon (target: MacBook Air M3, 16 GB RAM).
- Browsable history of every edit with input, output, prompt and seed.
- Simple UI: drop in an image, type an instruction, get a result. Few controls (seed, steps, size).

## 4. Non-goals (for v1)

- No cloud generation or hosted service.
- No multi-user support or accounts.
- No video generation.
- No mobile app.
- No model training or fine-tuning inside the app.

## 5. Target user

- **[TBD]** Just me, or others too? This decides licensing, packaging and polish.

## 6. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Model runtime | mflux (MLX) | Apple Silicon native |
| Model | FLUX.2 Klein edit (`Flux2KleinEdit`), start with the 4B distilled variant, quantized | mflux documents image-conditioned editing with one or more reference images, ~4 steps. 9B and 9B-kv are higher-quality options, but their full weights are ~32 GB, so they need heavy quantization on 16 GB. Test 4B first. |
| Backend | Python 3.11 + FastAPI + uvicorn | Conda env `holy`. `/edit` takes a multipart upload (image + prompt + settings). |
| Frontend | Plain HTML/CSS/JS | Image drop zone, prompt box, before/after view. Vite + React later if needed. |
| Storage | Inputs and outputs as files in `backend/outputs/`; UI history in IndexedDB | **[TBD]** move history to SQLite? |
| Packaging | Tauri or Electron with a Python sidecar | After the web version works |

The earlier Z-Image Turbo plan was text-to-image only, so it is no longer the primary model for this contract.

## 7. Constraints

- 16 GB unified memory shared with macOS: one job at a time, ~1024px max, and an input image is resized before use.
- Fanless Air: long batches will throttle.
- The FLUX.2 Klein 4B weights are ~15 GB unquantized and the 9B ~32 GB (per the mflux docs). Use quantization, and download on first launch rather than bundling.
- Memory use for editing (reference image tokens) is higher than for plain generation. **[Unmeasured]**: test on the actual device before committing to a size.
- **Licensing:** check each model's license before distributing the app. The 4B and 9B variants may have different licenses (the 4B is, to my knowledge, more permissive than the 9B). Verify on their Hugging Face pages.

## 8. Content and safety policy

**[TBD]** Image editing raises the stakes beyond generation: users can upload photos of real people. Decide before building:
- Whether uploading photos of real people is allowed, and under what conditions (for example, only your own photos or ones you have consent for).
- Whether to embed metadata or a visible marker that the output is AI-edited.
- Prompt and input filtering, especially for sexual content and anything involving minors, which should be a hard block.
- Whether to store inputs permanently or let users delete them.

## 9. Project structure

```
Holy/
├── backend/
│   ├── app.py          FastAPI app and routes (/edit, /history, /health)
│   ├── engine.py       model loading + edit pipeline
│   ├── history.py      history store (input, output, prompt, seed)
│   ├── importer.py     import images/settings
│   ├── exporter.py     export images/settings
│   ├── outputs/        generated images
│   ├── inputs/         uploaded source images
│   └── requirements.txt
├── ui/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── logic/
│       ├── api.js      calls to the backend
│       ├── db.js       IndexedDB history
│       ├── media.js    image upload, preview, download
│       └── stuff.js    helpers
└── meta/
    ├── projectscope.md this file
    ├── hand_book.md    how to run/develop
    ├── agent.md        notes/instructions for AI assistants
    ├── features.md     feature list and roadmap
    └── bugs.md         known issues
```

## 10. Milestones

1. **M1: Engine works.** `engine.py` takes an image and a prompt and returns an edited image. *(in progress)*
2. **M2: API.** FastAPI `/edit` endpoint with multipart upload and a model-loaded-once queue.
3. **M3: Minimal UI.** Image upload, prompt, seed/steps controls, before/after preview.
4. **M4: History.** Gallery of past edits with input, output, prompt and seed; reuse settings.
5. **M5: Import/export.** Save and load settings; export images with metadata.
6. **M6: Packaging.** Desktop wrapper, first-launch model download.

## 11. Success criteria for v1

- Given an image and a prompt, returns an edited image in under ~2 minutes on a MacBook Air M3 16 GB. **[Target, not yet measured]**
- Identity and anatomy of the subject are preserved in typical edits.
- Runs with no internet after the first model download.
- Every output can be reproduced from its stored input, prompt and seed.
- Doesn't crash or swap heavily during normal use.

## 12. Open questions

1. Personal use or distribution?
2. Main use case (portrait retouching, outfit/background changes, product shots, something else)?
3. Plain text-to-image as a secondary mode, yes or no?
4. Content policy for photos of real people (section 8)?
5. Native Mac app (SwiftUI) or web-tech wrapper (Tauri/Electron)?