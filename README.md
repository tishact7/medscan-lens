
<p align="center">
  <img src="https://img.shields.io/badge/Build%20with-Gemma-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Track-Healthcare%20AI-E53935?style=for-the-badge&logo=medical&logoColor=white" />
  <img src="https://img.shields.io/badge/Deploy-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">🏥 MedScan Lens</h1>
<p align="center"><strong>AI-Powered Handwritten Prescription Reader & Drug Interaction Checker</strong></p>

<p align="center">
  <a href="#-problem-statement">Problem</a> •
  <a href="#-solution">Solution</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-kaggle-notebook">Kaggle</a> •
  <a href="#-team">Team</a>
</p>

---

## 🎯 Problem Statement

Millions of patients worldwide — especially the elderly and those in rural or underserved areas — face serious challenges when handling handwritten medical prescriptions:

- **Illegible handwriting** leads to wrong medications being taken
- **Complex medical terminology** is confusing for non-experts
- **Drug interactions** are often overlooked, causing dangerous health risks
- **No digital copy** means patients forget dosage instructions

A single misread prescription can result in hospitalization or worse.

## 💡 Solution

**MedScan Lens** transforms a simple photo of a handwritten prescription into structured, actionable, and safe health information in seconds.

| Step | Feature | Benefit |
|:----:|---------|---------|
| 1 | 📸 **Upload Image** | Patient takes a photo of their prescription |
| 2 | 🧠 **AI Extraction** | Gemma 3 4B reads handwriting and extracts structured data |
| 3 | ⚠️ **Safety Check** | Automatically flags dangerous drug interactions |
| 4 | 💬 **Plain Output** | Clean, easy-to-read summary for patients |

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User uploads   │
│  prescription   │
│     image       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Hugging Face Inference    │
│        API (Gemma 4)     │
│   Multimodal Vision + NLP   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│      Structured JSON        │
│  • Patient Name             │
│  • Medications & Dosage     │
│  • Frequency & Quantity     │
│  • Doctor Name & Date       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Drug Interaction         │
│       Safety Engine         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Patient-Friendly          │
│      Dashboard              │
└─────────────────────────────┘
```

**Key Design Decisions:**
- 🚀 **Serverless AI** — Uses Hugging Face Inference API so the app runs on free Streamlit Cloud with **zero GPU requirement**
- 🔒 **Token Security** — API keys live in Streamlit Secrets, never in source code
- 🔄 **Gemma-Ready** — Single-line model swap upgrades the system to Gemma 4

---

## 🌐 Live Demo

**Try it now:** 👉 [medscan-lens-app.streamlit.app](https://medscan-lens-app.streamlit.app/)

> The deployed app uses Hugging Face's serverless API. Just enter your free Hugging Face token in the sidebar and upload a prescription image.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| AI Model | `unsloth/gemma-4-it` | Multimodal vision & text understanding |
| Quantization | `bitsandbytes` 4-bit | Fits model inside Kaggle GPU limits |
| API Layer | Hugging Face Inference API | Zero-GPU deployment on Streamlit |
| Frontend | Streamlit | Interactive web UI |
| Image Processing | Pillow | Contrast enhancement & resizing |
| Language | Python 3.10+ | Core pipeline |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/tishact7/medscan-lens.git
cd medscan-lens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Hugging Face Token
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Option A — Environment Variable:**
```bash
# Linux / Mac
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"

# Windows
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Option B — Streamlit Secrets (Recommended for deployment):**
Create a file at `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 4. Run locally
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 📓 Kaggle Notebook

The full training, prototyping, and model pipeline is available on Kaggle:

🔗 **[View Kaggle Notebook](https://www.kaggle.com/code/tishachatterjee07/notebook-app)**

The notebook includes:
- Gemma 3 4B loading with 4-bit quantization
- Multimodal prescription extraction pipeline
- JSON parsing & error handling
- Drug interaction checking logic
- Plain-language summary generation

---

## 📝 Note on Gemma 4

Due to **Kaggle GPU storage constraints** (15.5 GB VRAM limit) during the 24-hour hackathon window, we optimized our prototype using **Gemma 4** with 4-bit quantization (~12 GB VRAM). 

Our architecture is intentionally designed for a **seamless one-line upgrade** to Gemma 4:
```python
# Current
MODEL_ID = "unsloth/gemma-4-it"

# Future upgrade
MODEL_ID = "google/gemma-4-it"  # or applicable Gemma 4 variant
```

---

## 📂 Repository Structure

```
prescription-ai/
├── app.py                      # Streamlit frontend + Hugging Face API integration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Excludes secrets and cache
├── secrets.toml.example        # Template for local API token setup
└── notebook/
    └── medscan-lens.ipynb   # Full Kaggle prototype notebook
```

---

## ⚠️ Medical Disclaimer

> **This project is built strictly for educational and hackathon demonstration purposes.**  
> It is **NOT** a certified medical device, diagnostic tool, or substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a licensed physician, pharmacist, or qualified healthcare provider before taking, stopping, or changing any medication.

---

## 👥 Team

Built with ❤️ for **Build with Gemma: GDG TIU Buildathon**  
**Track:** AI for Healthcare

| Name | Role |
|------|------|
| [Tisha Chatterjee] | ML Engineering & Pipeline |
| [Lekhoni Sarkar] | UI/UX & Research |

---

## 📄 License

This project is released for hackathon and educational use. See repository for details.

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Passion-ff69b4?style=flat-square" />
  <img src="https://img.shields.io/badge/Powered%20by-Gemma-4285F4?style=flat-square" />
</p>
```

---

