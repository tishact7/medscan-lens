# MedScan Lens

**AI-Powered Medication Safety Assistant**  
Built for **Build with Gemma: GDG TIU Buildathon**  
Track: **AI for Healthcare**

## Problem
80% of patients in India cannot read their handwritten prescriptions. This leads to:
- Wrong medication taken
- Dangerous drug interactions missed
- Anxiety and confusion about treatment

## Solution
MedScan Lens uses Google's Gemma 3 4B vision model to:
1. **Read** handwritten drug names from prescription photos
2. **Explain** what each drug does in plain language
3. **Warn** about dangerous drug interactions
4. **Summarize** the entire prescription for patients

## Demo
[(https://medscan-lens-app.streamlit.app/)]

## Tech Stack
- **AI Model:** Gemma 3 4B (4-bit quantized, ~3GB VRAM)
- **Vision:** Multimodal image understanding for handwritten OCR
- **Safety:** Local drug interaction database
- **UI:** Streamlit web app
- **Deployment:** Kaggle (prototype) + Streamlit Cloud

## Note on Gemma 4
Due to Kaggle GPU storage constraints during the 24-hour hackathon, we optimized for Gemma 3 4B with 4-bit quantization. The architecture is designed to seamlessly upgrade to Gemma 4 via model ID swap with zero code changes.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
