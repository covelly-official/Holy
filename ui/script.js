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

    async function loadLibrary() {
        if (!libraryGrid) return;
        try {
            const res = await fetch(`${API_BASE}/history`);
            const data = await res.json();
            libraryGrid.innerHTML = '';
            data.images.forEach(imgFile => {
                const img = document.createElement('img');
                img.src = `${API_BASE}/outputs/${imgFile}`;
                img.onclick = () => window.open(img.src, '_blank');
                libraryGrid.appendChild(img);
            });
        } catch (e) { console.error(e); }
    }
    loadLibrary();

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
    
    let currentRawImage = null;
    const aspectRatioSelect = document.getElementById('aspect-ratio');
    
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
    
    // Handle Image Upload
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                currentRawImage = img;
                renderImageToCanvas();
                placeholder.style.display = 'none';
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
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
    maskCanvas.addEventListener('mouseup', () => isDrawing = false);
    maskCanvas.addEventListener('mouseout', () => isDrawing = false);
    
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
            btnText.textContent = 'Generate Try-On';
            loader.style.display = 'none';
        }
    });
    
    newTryonBtn.addEventListener('click', () => {
        resultContainer.style.display = 'none';
        canvasContainer.style.display = 'flex';
    });
});
