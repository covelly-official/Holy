const API_BASE = '';

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('generate-form');
    const stepsInput = document.getElementById('steps');
    const stepsVal = document.getElementById('steps-val');
    const strengthInput = document.getElementById('strength');
    const strengthVal = document.getElementById('strength-val');
    const generateBtn = document.getElementById('generate-btn');
    const btnText = generateBtn.querySelector('span');
    const loader = generateBtn.querySelector('.loader-spinner');
    
    // Canvas & File Upload Elements
    const imageUpload = document.getElementById('image-upload');
    const canvasContainer = document.getElementById('canvas-container');
    const baseCanvas = document.getElementById('base-canvas');
    const maskCanvas = document.getElementById('mask-canvas');
    const baseCtx = baseCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');
    const placeholder = document.getElementById('canvas-placeholder');
    
    // Results & Library
    const resultContainer = document.getElementById('result-container');
    const resultImage = document.getElementById('result-image');
    const newTryonBtn = document.getElementById('new-tryon-btn');
    const libraryGrid = document.getElementById('library-grid');
    
    // Before/After
    const baContainer = document.getElementById('before-after-container');
    const baWrapper = document.getElementById('before-wrapper');
    const baHandle = document.getElementById('ba-handle');
    const originalImagePreview = document.getElementById('original-image-preview');

    const contextMenu = document.getElementById('library-context-menu');
    const ctxSendCanvas = document.getElementById('ctx-send-canvas');
    const ctxDownload = document.getElementById('ctx-download');
    let contextTargetUrl = null;

    async function loadLibrary() {
        if (!libraryGrid) return;
        try {
            const res = await fetch(`${API_BASE}/history`);
            const data = await res.json();
            libraryGrid.innerHTML = '';
            data.images.forEach(imgFile => {
                const img = document.createElement('img');
                const imgUrl = `${API_BASE}/outputs/${imgFile}`;
                img.src = imgUrl;
                
                // Left click -> View full screen
                img.onclick = () => window.open(imgUrl, '_blank');
                
                // Right click -> Open Context Menu
                img.oncontextmenu = (e) => {
                    e.preventDefault();
                    contextTargetUrl = imgUrl;
                    contextMenu.style.display = 'flex';
                    // Position menu at mouse coordinates (relative to page)
                    contextMenu.style.left = `${e.pageX}px`;
                    contextMenu.style.top = `${e.pageY}px`;
                };
                
                libraryGrid.appendChild(img);
            });
        } catch (e) { console.error(e); }
    }
    loadLibrary();

    // Hide context menu when clicking outside
    document.addEventListener('click', (e) => {
        if (contextMenu && e.target.closest('#library-context-menu') === null) {
            contextMenu.style.display = 'none';
        }
    });

    // Handle "Download" action
    if (ctxDownload) {
        ctxDownload.addEventListener('click', () => {
            if (!contextTargetUrl) return;
            const a = document.createElement('a');
            a.href = contextTargetUrl;
            a.download = contextTargetUrl.split('/').pop(); // Extracts filename
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            contextMenu.style.display = 'none';
        });
    }

    // Handle "Send to Canvas" action
    if (ctxSendCanvas) {
        ctxSendCanvas.addEventListener('click', async () => {
            if (!contextTargetUrl) return;
            
            try {
                // Fetch the image as a blob so we can populate the actual HTML file input
                const res = await fetch(contextTargetUrl);
                const blob = await res.blob();
                
                // Construct a File object and forcefully inject it into the <input type="file">
                // This prevents the HTML "Please select a file" validation error!
                const file = new File([blob], contextTargetUrl.split('/').pop(), { type: blob.type });
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                const imageUpload = document.getElementById('image-upload');
                if (imageUpload) {
                    imageUpload.files = dataTransfer.files;
                }

                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => {
                    currentRawImage = img;
                    renderImageToCanvas();
                    canvasPlaceholder.style.display = 'none';
                    
                    // Jump to Canvas view if we are on Result view
                    resultContainer.style.display = 'none';
                    canvasContainer.style.display = 'flex';
                    
                    // Scroll to top
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                };
                img.src = URL.createObjectURL(blob);
                contextMenu.style.display = 'none';
            } catch (err) {
                console.error("Error loading image from library:", err);
                contextMenu.style.display = 'none';
            }
        });
    }

    let isSliding = false;
    
    function onSlide(e) {
        if (!isSliding || !baContainer) return;
        let clientX = e.clientX;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
        }
        const rect = baContainer.getBoundingClientRect();
        let x = clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        const percent = (x / rect.width) * 100;
        baWrapper.style.width = `${percent}%`;
        baHandle.style.left = `${percent}%`;
    }

    if (baContainer) {
        baContainer.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isSliding = true;
            onSlide(e);
        });
        baContainer.addEventListener('touchstart', (e) => {
            isSliding = true;
            onSlide(e);
        });
        window.addEventListener('mousemove', onSlide);
        window.addEventListener('touchmove', onSlide, { passive: true });
        window.addEventListener('mouseup', () => isSliding = false);
        window.addEventListener('touchend', () => isSliding = false);
    }

    function updateBeforeAfterWidths() {
        const container = document.getElementById('before-after-container');
        if(container && originalImagePreview) {
            originalImagePreview.style.width = `${container.offsetWidth}px`;
        }
    }
    window.addEventListener('resize', updateBeforeAfterWidths);
    
    // Brush Logic
    const brushSizeInput = document.getElementById('brush-size');
    const brushSizeVal = document.getElementById('brush-size-val');
    const clearMaskBtn = document.getElementById('clear-mask-btn');
    const drawModeBtn = document.getElementById('draw-mode-btn');
    const eraseModeBtn = document.getElementById('erase-mode-btn');
    
    let isDrawing = false;
    let brushSize = 20;
    let isErasing = false;

    if (drawModeBtn && eraseModeBtn) {
        drawModeBtn.addEventListener('click', () => {
            isErasing = false;
            drawModeBtn.style.borderColor = 'var(--white)';
            drawModeBtn.style.color = 'var(--white)';
            eraseModeBtn.style.borderColor = 'var(--border-light)';
            eraseModeBtn.style.color = 'var(--gray-300)';
        });
        eraseModeBtn.addEventListener('click', () => {
            isErasing = true;
            eraseModeBtn.style.borderColor = 'var(--white)';
            eraseModeBtn.style.color = 'var(--white)';
            drawModeBtn.style.borderColor = 'var(--border-light)';
            drawModeBtn.style.color = 'var(--gray-300)';
        });
    }
    
    // UI Label Updates
    stepsInput.addEventListener('input', e => stepsVal.textContent = e.target.value);
    strengthInput.addEventListener('input', e => strengthVal.textContent = parseFloat(e.target.value).toFixed(2));
    brushSizeInput.addEventListener('input', e => {
        brushSize = parseInt(e.target.value);
        brushSizeVal.textContent = brushSize;
    });
    // State Persistence
    let currentRawImage = null;
    const aspectRatioSelect = document.getElementById('aspect-ratio');
    
    function saveState() {
        const state = {
            prompt: document.getElementById('prompt').value,
            negativePrompt: document.getElementById('negative-prompt').value,
            aspectRatio: aspectRatioSelect ? aspectRatioSelect.value : 'original',
            steps: stepsInput.value,
            strength: strengthInput.value
        };
        sessionStorage.setItem('holy_state', JSON.stringify(state));
        
        if (currentRawImage) {
            // Save base image and mask as data URLs
            // We use a scaled down version for storage limits if needed, but modern browsers allow 5MB+
            try {
                sessionStorage.setItem('holy_img', currentRawImage.src);
                sessionStorage.setItem('holy_mask', maskCanvas.toDataURL());
            } catch (e) {
                console.warn('Session storage quota exceeded for images.');
            }
        }
    }

    function loadState() {
        const savedState = sessionStorage.getItem('holy_state');
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                if (state.prompt) document.getElementById('prompt').value = state.prompt;
                if (state.negativePrompt) document.getElementById('negative-prompt').value = state.negativePrompt;
                if (state.aspectRatio && aspectRatioSelect) aspectRatioSelect.value = state.aspectRatio;
                if (state.steps) {
                    stepsInput.value = state.steps;
                    stepsVal.textContent = state.steps;
                }
                if (state.strength) {
                    strengthInput.value = state.strength;
                    strengthVal.textContent = parseFloat(state.strength).toFixed(2);
                }
            } catch(e) {}
        }
        
        const savedImg = sessionStorage.getItem('holy_img');
        if (savedImg) {
            try {
                fetch(savedImg).then(res => res.blob()).then(blob => {
                    const file = new File([blob], "restored_session.png", { type: "image/png" });
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    const imageUpload = document.getElementById('image-upload');
                    if (imageUpload) imageUpload.files = dataTransfer.files;
                });
            } catch (e) { console.error(e); }

            const img = new Image();
            img.onload = () => {
                currentRawImage = img;
                renderImageToCanvas();
                placeholder.style.display = 'none';
                
                // Attempt to restore mask
                const savedMask = sessionStorage.getItem('holy_mask');
                if (savedMask) {
                    const maskImg = new Image();
                    maskImg.onload = () => {
                        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                        maskCtx.drawImage(maskImg, 0, 0);
                    };
                    maskImg.src = savedMask;
                }
            };
            img.src = savedImg;
        }
    }

    // Attach listeners to save state automatically
    document.getElementById('prompt').addEventListener('input', saveState);
    document.getElementById('negative-prompt').addEventListener('input', saveState);
    if (aspectRatioSelect) aspectRatioSelect.addEventListener('change', () => { renderImageToCanvas(); saveState(); });
    stepsInput.addEventListener('change', saveState);
    strengthInput.addEventListener('change', saveState);

    // Load state on startup
    loadState();

    
    function renderImageToCanvas() {
        if (!currentRawImage) return;
        const img = currentRawImage;
        const ratioType = aspectRatioSelect ? aspectRatioSelect.value : 'original';
        
        let targetW = img.width;
        let targetH = img.height;
        
        const MAX_DIM = 1024;
        
        if (ratioType === 'original') {
            // Constrain to MAX_DIM while keeping aspect ratio
            if (targetW > MAX_DIM || targetH > MAX_DIM) {
                const scale = Math.min(MAX_DIM / targetW, MAX_DIM / targetH);
                targetW = Math.floor(targetW * scale);
                targetH = Math.floor(targetH * scale);
            }
        } else if (ratioType === 'square') {
            targetW = 512; targetH = 512;
        } else if (ratioType === 'portrait') {
            targetW = 512; targetH = 768;
        } else if (ratioType === 'landscape') {
            targetW = 768; targetH = 512;
        }
        
        // AI requires dimensions to be perfectly divisible by 8
        // By doing this on the frontend BEFORE drawing the image, it crops the 
        // 1-7 pixel overflow perfectly instead of the backend squishing/stretching it.
        targetW = Math.floor(targetW / 8) * 8;
        targetH = Math.floor(targetH / 8) * 8;
        
        baseCanvas.width = targetW;
        baseCanvas.height = targetH;
        maskCanvas.width = targetW;
        maskCanvas.height = targetH;
        
        const imgRatio = img.width / img.height;
        const targetRatio = targetW / targetH;
        let drawW = targetW;
        let drawH = targetH;
        let offsetX = 0;
        let offsetY = 0;
        
        if (imgRatio > targetRatio) {
            drawW = targetH * imgRatio;
            offsetX = (targetW - drawW) / 2;
        } else {
            drawH = targetW / imgRatio;
            offsetY = (targetH - drawH) / 2;
        }
        
        baseCtx.clearRect(0, 0, targetW, targetH);
        baseCtx.drawImage(img, offsetX, offsetY, drawW, drawH);
        clearMask();
        
        const cssRatio = targetH / targetW;
        const containerWidth = canvasContainer.clientWidth;
        canvasContainer.style.height = `${Math.min(containerWidth * cssRatio, window.innerHeight * 0.8)}px`;
    }
    
    if (aspectRatioSelect) {
        aspectRatioSelect.addEventListener('change', renderImageToCanvas);
    }
    
    // Shared file handler
    function handleImageFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        
        // Sync the file to the actual input element so the backend gets it
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageUpload.files = dataTransfer.files;

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                currentRawImage = img;
                renderImageToCanvas();
                placeholder.style.display = 'none';
                saveState();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Handle Image Upload via Button
    imageUpload.addEventListener('change', (e) => {
        handleImageFile(e.target.files[0]);
    });
    
    // Click on placeholder to upload
    placeholder.addEventListener('click', () => {
        imageUpload.click();
    });

    // Drag and drop onto Canvas Area
    canvasContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        canvasContainer.classList.add('drag-active');
    });

    canvasContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        canvasContainer.classList.remove('drag-active');
    });

    canvasContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        canvasContainer.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    
    // Helper to calculate mouse coordinates taking CSS scaling into account
    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }
    
    // Mouse Events for Mask Canvas
    maskCanvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        draw(e);
    });
    
    maskCanvas.addEventListener('mousemove', draw);
    maskCanvas.addEventListener('mouseup', () => {
        if (isDrawing) { isDrawing = false; saveState(); }
    });
    maskCanvas.addEventListener('mouseout', () => {
        if (isDrawing) { isDrawing = false; saveState(); }
    });
    
    // Draw semi-transparent brush stroke
    function draw(e) {
        if (!isDrawing) return;
        const pos = getMousePos(maskCanvas, e);
        
        // Draw smoothly
        maskCtx.beginPath();
        maskCtx.arc(pos.x, pos.y, brushSize, 0, Math.PI * 2);
        
        if (isErasing) {
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.fillStyle = 'rgba(0,0,0,1)';
        } else {
            maskCtx.globalCompositeOperation = 'source-over';
            maskCtx.fillStyle = 'rgba(217, 70, 239, 0.6)'; // Neon purple masking color
        }
        
        maskCtx.fill();
        maskCtx.closePath();
    }
    
    clearMaskBtn.addEventListener('click', clearMask);
    function clearMask() {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        saveState();
    }
    
    // Clear Workspace
    const clearWorkspaceBtn = document.getElementById('clear-workspace-btn');
    if (clearWorkspaceBtn) {
        clearWorkspaceBtn.addEventListener('click', () => {
            sessionStorage.removeItem('holy_img');
            sessionStorage.removeItem('holy_mask');
            sessionStorage.removeItem('holy_state');
            currentRawImage = null;
            baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            placeholder.style.display = 'flex';
            canvasContainer.style.height = 'auto'; // Reset height
            document.getElementById('prompt').value = '';
            document.getElementById('negative-prompt').value = 'deformed, blurry, bad anatomy, low quality';
        });
    }
    
    // Utility: Convert painted mask into a pure B&W mask for backend
    function getMaskBlob() {
        return new Promise(resolve => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = maskCanvas.width;
            tempCanvas.height = maskCanvas.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Solid black background
            tempCtx.fillStyle = 'black';
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
            const tempData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Everywhere the user painted (alpha > 0), make it solid white
            for (let i = 0; i < maskData.data.length; i += 4) {
                if (maskData.data[i+3] > 0) {
                    tempData.data[i] = 255;   // R
                    tempData.data[i+1] = 255; // G
                    tempData.data[i+2] = 255; // B
                }
            }
            tempCtx.putImageData(tempData, 0, 0);
            
            tempCanvas.toBlob(blob => resolve(blob), 'image/png');
        });
    }
    
    function getBaseImageBlob() {
        return new Promise(resolve => {
            baseCanvas.toBlob(blob => resolve(blob), 'image/png');
        });
    }

    const abortBtn = document.getElementById('abort-btn');
    if (abortBtn) {
        abortBtn.addEventListener('click', async () => {
            try {
                await fetch(`${API_BASE}/abort`, { method: 'POST' });
            } catch(e) {}
        });
    }

    // Submit to FastAPI
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!imageUpload.files[0]) {
            alert("Please upload a photo of the person first.");
            return;
        }

        const prompt = document.getElementById('prompt').value;
        const negPromptEl = document.getElementById('negative-prompt');
        const negativePrompt = negPromptEl ? negPromptEl.value : "";
        const steps = document.getElementById('steps').value;
        const strength = document.getElementById('strength').value;
        const seedStr = document.getElementById('seed').value;

        generateBtn.disabled = true;
        btnText.textContent = 'Synthesizing...';
        loader.style.display = 'block';
        
        const genOverlay = document.getElementById('generation-overlay');
        const overlayText = document.getElementById('overlay-text');
        const progressBarFill = document.getElementById('progress-bar-fill');
        if (genOverlay) genOverlay.style.display = 'flex';
        if (overlayText) overlayText.textContent = 'Synthesizing...';
        if (progressBarFill) progressBarFill.style.width = '0%';

        const progressInterval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/progress`);
                if (res.ok) {
                    const data = await res.json();
                    if (data.total_steps > 0 && data.step <= data.total_steps) {
                        const percent = Math.round((data.step / data.total_steps) * 100);
                        const progressMsg = `Generating ${percent}%`;
                        btnText.textContent = progressMsg;
                        if (overlayText) overlayText.textContent = progressMsg;
                        if (progressBarFill) progressBarFill.style.width = `${percent}%`;
                    }
                }
            } catch(e) {}
        }, 500);

        try {
            const imageBlob = await getBaseImageBlob();
            const maskBlob = await getMaskBlob();
            
            const formData = new FormData();
            formData.append('prompt', prompt);
            if (negativePrompt.trim()) {
                formData.append('negative_prompt', negativePrompt.trim());
            }
            formData.append('image', imageBlob, 'original.png');
            formData.append('mask', maskBlob, 'mask.png');
            formData.append('steps', steps);
            formData.append('strength', strength);
            if (seedStr) {
                formData.append('seed', seedStr);
            }

            const response = await fetch(`${API_BASE}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.status === "aborted") {
                console.log("Generation aborted by user");
                return;
            }
            
            originalImagePreview.src = URL.createObjectURL(imageBlob);
            resultImage.src = `${API_BASE}${data.url}`;
            
            canvasContainer.style.display = 'none';
            resultContainer.style.display = 'block';
            
            setTimeout(updateBeforeAfterWidths, 50);
            loadLibrary();
            
        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate image. Ensure the backend (uvicorn) is running!');
        } finally {
            if (genOverlay) genOverlay.style.display = 'none';
            clearInterval(progressInterval);
            generateBtn.disabled = false;
            btnText.textContent = 'Generate';
            loader.style.display = 'none';
        }
    });
    
    newTryonBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resultContainer.style.display = 'none';
        canvasContainer.style.display = 'flex';
    });

    const continueEditingBtn = document.getElementById('continue-editing-btn');
    if (continueEditingBtn) {
        continueEditingBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const resImgSrc = document.getElementById('result-image').src;
            if (!resImgSrc) return;
            try {
                const res = await fetch(resImgSrc);
                const blob = await res.blob();
                
                const file = new File([blob], "continued_edit.png", { type: blob.type || "image/png" });
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                const uploadInput = document.getElementById('image-upload');
                if (uploadInput) uploadInput.files = dataTransfer.files;

                const img = new Image();
                img.onload = () => {
                    currentRawImage = img;
                    renderImageToCanvas();
                    const placeholder = document.getElementById('canvas-placeholder');
                    if (placeholder) placeholder.style.display = 'none';
                    clearMask();
                    saveState();
                    
                    resultContainer.style.display = 'none';
                    canvasContainer.style.display = 'flex';
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                };
                img.src = URL.createObjectURL(blob);
            } catch (err) {
                console.error("Error continuing edit:", err);
            }
        });
    }
});
