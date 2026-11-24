# Holy.

> **an local AI image editor**  

This is an offlienimage editor made for apple silicon any M serise mac,
---


## 💻 System Requirements

- **Platform:** macOS with Apple Silicon (M1, M2, M3, M4 or later)
- **Memory:** 16 GB Unified Memory recommended
- **Python:** Python 3.10, 3.11, or 3.13

---

##  How to run it

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Holy.git
cd Holy
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch the Studio
```bash
python backend/app.py
```

Open your browser and navigate to:
```text
http://localhost:9000
```
*(On your very first generation, the pipeline will automatically download the Realistic Vision inpainting weights from Hugging Face and cache them locally).*

---

## 📁 Project Structure

```text
Holy/
├── backend/
│   ├── app.py              # FastAPI server & static file host
│   ├── engine.py           # PyTorch MPS & MLX generation engine
│   ├── requirements.txt    # Python dependencies
│   └── outputs/            # Local generation library (git-ignored)
├── ui/
│   ├── index.html          # Studio frontend
│   ├── style.css           # Glassmorphic dark theme
│   └── script.js           # Canvas & generation controller
├── models/
│   └── embeddings/         # Negative textual inversion embeddings (.pt)
└── README.md
```

---

## ⚖️ Legal Disclaimer & Terms of Use

> **PLEASE READ CAREFULLY BEFORE USING THIS SOFTWARE**

### 1. Complete Disclaimer of Developer Liability
This software is provided for research, creative exploration, and personal educational purposes on an **"AS IS"** and **"AS AVAILABLE"** basis, without warranties of any kind, whether express or implied. 

**Under no circumstances shall the developer, creators, maintainers, or contributors of Holy be held liable, legally or otherwise, for any content, images, modifications, or material produced, stored, or distributed through the use of this software.** 

### 2. Sole End-User Responsibility
**Any individual or entity who clones, installs, or executes this software assumes 100% total, exclusive, and unreserved legal and civil liability for all outputs generated.** 
- You are solely responsible for ensuring that your usage of this software complies with all applicable local, national, and international laws, regulations, and statutes.
- You bear sole responsibility for obtaining all necessary rights and explicit consent before processing or inpainting images featuring real people.

### 3. Strictly Prohibited Uses
You agree not to use Holy to generate, process, or distribute:
- Non-consensual sexual imagery (NCII / deepfakes) of any living individual.
- Child Sexual Abuse Material (CSAM) or any exploitation involving minors (under any jurisdiction, this is strictly prohibited and illegal).
- Defamatory, libelous, harassing, hateful, or unlawful materials.

By cloning, running, or using this repository, you explicitly acknowledge, understand, and agree to these terms.
