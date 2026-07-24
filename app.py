import streamlit as st
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image, ImageEnhance
import json
import re
import os

# ─── Page Config ───
st.set_page_config(page_title="MedScan Lens", page_icon="🏥", layout="wide")
st.title("🏥 MedScan Lens")
st.caption("AI-Powered Medication Safety Assistant | Built with Gemma 3 4B")

# ─── Load Model (cached) ───
@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "unsloth/gemma-3-4b-it"
    
    # Try 4-bit first
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    except:
        # Fallback: no quantization
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    return processor, model

with st.spinner("⏳ Loading Gemma 3 4B... (~2 minutes)"):
    processor, model = load_model()

# ─── Drug Database ───
DRUG_DB = {
    "paracetamol": {"cat": "Pain Reliever", "use": "Fever, headache", "warn": "Max 4g/day. Avoid alcohol."},
    "acetaminophen": {"cat": "Pain Reliever", "use": "Fever, headache", "warn": "Same as Paracetamol (US name)."},
    "aspirin": {"cat": "Blood Thinner", "use": "Pain, heart protection", "warn": "May cause bleeding. Avoid if pregnant."},
    "metformin": {"cat": "Diabetes Med", "use": "Type 2 diabetes", "warn": "Take with food. Monitor kidneys."},
    "atorvastatin": {"cat": "Cholesterol", "use": "High cholesterol", "warn": "Avoid grapefruit. Report muscle pain."},
    "amoxicillin": {"cat": "Antibiotic", "use": "Bacterial infections", "warn": "Complete full course."},
    "warfarin": {"cat": "Blood Thinner", "use": "Prevent clots", "warn": "Regular blood tests. Avoid vitamin K foods."},
    "lisinopril": {"cat": "Blood Pressure", "use": "Hypertension", "warn": "May cause dry cough."},
    "ibuprofen": {"cat": "Pain Reliever", "use": "Pain, inflammation", "warn": "Take with food. Avoid if kidney issues."},
    "omeprazole": {"cat": "Acid Reducer", "use": "GERD, ulcers", "warn": "Long-term use affects calcium/B12."},
}

IX_DB = {
    tuple(sorted(["metformin", "aspirin"])): {"sev": "major", "txt": "Lactic acidosis risk. Monitor kidney function."},
    tuple(sorted(["warfarin", "aspirin"])): {"sev": "contraindicated", "txt": "HIGH BLEEDING RISK. Never combine without doctor."},
    tuple(sorted(["atorvastatin", "aspirin"])): {"sev": "moderate", "txt": "Increased bleeding risk."},
    tuple(sorted(["ibuprofen", "aspirin"])): {"sev": "moderate", "txt": "Ibuprofen may reduce aspirin's heart protection."},
}

# ─── Helper Functions ───
def preprocess(img, max_size=512):
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def read_drug(img):
    prompt = 'Read the handwritten drug name. Return ONLY JSON: {"drug_name": "text", "confidence": "high|medium|low"}'
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                            return_tensors="pt", add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        return json.loads(m.group()) if m else {"drug_name": text.strip(), "confidence": "low"}
    except:
        return {"drug_name": text.strip(), "confidence": "low"}

def lookup(name):
    clean = name.lower().strip().replace(".", "").replace(",", "")
    for k, v in DRUG_DB.items():
        if k in clean or clean in k:
            return {"name": k.title(), **v, "ok": True}
    return {"name": name, "cat": "Unknown", "use": "Unknown", "warn": "Verify with doctor.", "ok": False}

def check_ix(names):
    found = []
    n = [d.lower().strip() for d in names if d.strip()]
    for i, a in enumerate(n):
        for b in n[i+1:]:
            key = tuple(sorted([a, b]))
            if key in IX_DB and IX_DB[key] not in found:
                found.append(IX_DB[key])
    return found

def summarize(drugs, ix):
    med_txt = "\n".join([f"- {d['name']}: {d['cat']}. {d['use']}. {d['warn']}" for d in drugs])
    alert_txt = "\n".join([f"[{x['sev'].upper()}] {x['txt']}" for x in ix]) if ix else "No dangerous interactions."
    prompt = f"""You are a friendly medical assistant. Explain simply:

MEDICATIONS:
{med_txt}

SAFETY:
{alert_txt}

Write 4-5 warm sentences. End with: "This is not medical advice. Consult your doctor."

Response:"""
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True, top_p=0.9)
    gen = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

# ─── UI Tabs ───
tab1, tab2 = st.tabs(["📸 Single Drug Scanner", "📄 Prescription Analyzer"])

# ─── Tab 1: Single Drug ───
with tab1:
    st.header("Scan a Handwritten Drug Name")
    st.write("Upload a cropped image of a drug name from your prescription. Gemma reads it and explains what it does.")
    
    uploaded = st.file_uploader("Upload drug image", type=["jpg", "png", "jpeg"], key="single")
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        img = Image.open(uploaded)
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("Gemma is reading..."):
                proc = preprocess(img)
                result = read_drug(proc)
            
            st.subheader(f"📝 Detected: '{result['drug_name']}'")
            st.caption(f"Confidence: {result['confidence']}")
            
            info = lookup(result['drug_name'])
            
            st.subheader(f"💊 {info['name']}")
            st.write(f"**Category:** {info['cat']}")
            st.write(f"**Used for:** {info['use']}")
            
            if info['ok']:
                st.info(f"⚠️ **Caution:** {info['warn']}")
            else:
                st.warning(f"❓ {info['warn']}")
            
            # Interaction demo
            st.divider()
            st.write("**Simulated Safety Check** (with Aspirin):")
            ix = check_ix([result['drug_name'], "aspirin"])
            if ix:
                for x in ix:
                    if x['sev'] == 'contraindicated':
                        st.error(f"🚫 {x['txt']}")
                    elif x['sev'] == 'major':
                        st.warning(f"⚠️ {x['txt']}")
                    else:
                        st.info(f"ℹ️ {x['txt']}")
            else:
                st.success("✅ No dangerous interactions")

# ─── Tab 2: Full Prescription ───
with tab2:
    st.header("Full Prescription Analysis")
    st.write("Enter all medications from your prescription. The AI checks interactions and explains everything.")
    
    st.subheader("Enter Medications (one per line):")
    default_meds = "Metformin\nAtorvastatin\nAspirin"
    drugs_input = st.text_area("Drug names", value=default_meds, height=120)
    drugs = [d.strip() for d in drugs_input.split("\n") if d.strip()]
    
    if st.button("🔍 Analyze with Gemma", type="primary"):
        with st.spinner("Analyzing..."):
            infos = [lookup(d) for d in drugs]
            ix = check_ix([d['name'] for d in infos])
            
            # Display cards
            st.subheader("📋 Your Medications")
            cols = st.columns(len(infos))
            for col, info in zip(cols, infos):
                with col:
                    st.metric(info['name'], info['cat'])
                    st.caption(f"Use: {info['use']}")
            
            # Safety
            st.subheader("⚠️ Safety Alerts")
            if ix:
                for x in ix:
                    if x['sev'] == 'contraindicated':
                        st.error(f"🚫 **CONTRAINDICATED:** {x['txt']}")
                    elif x['sev'] == 'major':
                        st.warning(f"⚠️ **MAJOR:** {x['txt']}")
                    else:
                        st.info(f"ℹ️ **MODERATE:** {x['txt']}")
            else:
                st.success("✅ No known dangerous interactions between these medications.")
            
            # Summary
            st.subheader("💬 What This Means")
            summary = summarize(infos, ix)
            st.markdown(f"> {summary}")
            
            st.divider()
            st.caption("⚕️ **Disclaimer:** This is an AI assistant, not a doctor. Always consult a healthcare professional before taking or changing medications.")

# ─── Sidebar ───
with st.sidebar:
    st.title("About")
    st.write("**MedScan Lens** helps patients understand handwritten prescriptions.")
    st.write("**Problem:** 80% of patients can't read their prescriptions.")
    st.write("**Solution:** AI reads drug names, explains them, and checks for dangerous interactions.")
    st.divider()
    st.write("**Model:** Gemma 3 4B (4-bit quantized)")
    st.write("**Track:** AI for Healthcare")
    st.write("**Hackathon:** Build with Gemma — GDG TIU")
    st.divider()
    st.write("⚠️ **Not medical advice.** Consult your doctor.")