import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image, ImageEnhance
import json, re, io, base64

st.set_page_config(page_title="MedScan Lens", page_icon="🏥", layout="centered")

# ========== GITHUB VERSION: Token from Secrets or Sidebar ==========
with st.sidebar:
    st.header("🔐 API Setup")
    st.markdown("Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
    
    try:
        hf_token = st.secrets["HF_TOKEN"]
        st.success("✅ Token loaded from Streamlit Secrets")
    except Exception:
        hf_token = st.text_input("Hugging Face Token", type="password")
    
    st.caption("🔒 Your token is never saved to GitHub.")
# =====================================================================

st.title("🏥 MedScan Lens")
st.caption("Healthcare Assistant | AI for Healthcare | Built with Gemma 4")
st.markdown("---")

if not hf_token:
    st.info("👈 Please enter your Hugging Face token in the sidebar to start.")
    st.stop()

client = InferenceClient(model="google/gemma-4-e2b-it", token=hf_token)

def preprocess(img):
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img.thumbnail((896, 896), Image.Resampling.LANCZOS)
    return img

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_from_image(img):
    prompt = """Analyze this prescription image. Extract all readable information.
Return ONLY valid JSON. No markdown, no explanation.
Format:
{"document_type":"prescription","patient_name":"name or null","patient_age":"age or null",
"medications":[{"name":"drug","dosage":"dose","frequency":"freq","quantity":"qty"}],
"doctor_name":"name or null","date":"date or null","raw_text":"other text"}
If unreadable, use null."""

    base64_img = image_to_base64(img)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            {"type": "text", "text": prompt}
        ]
    }]
    
    try:
        with st.spinner("🧠 AI is reading the prescription... (10-20 sec)"):
            response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.1)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {e}")
        return '{"error": "API call failed"}'

def parse_json(text):
    text = text.strip()
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for m in matches:
            try: return json.loads(m)
            except: continue
    s, e = text.find('{'), text.rfind('}')
    if s >= 0 and e > s:
        try: return json.loads(text[s:e+1])
        except: pass
    return {}

uploaded = st.file_uploader("📤 Upload Prescription Image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns([1, 1])
    img = Image.open(uploaded)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("📋 Extracted Data")
        proc = preprocess(img)
        raw = extract_from_image(proc)
        data = parse_json(raw)
        
        st.json(data)
        
        meds = data.get("medications", [])
        if meds:
            st.success(f"💊 Found {len(meds)} medication(s)")
            for m in meds:
                name = m.get('name', 'Unknown')
                dosage = m.get('dosage', 'N/A')
                freq = m.get('frequency', 'N/A')
                qty = m.get('quantity', 'N/A')
                st.markdown(f"**{name}**  \n`Dosage:` {dosage} | `Freq:` {freq} | `Qty:` {qty}")
        else:
            st.warning("No medications extracted. Try a clearer image.")
        
        names = [m.get("name", "").lower() for m in meds if m.get("name")]
        risky = [("warfarin", "aspirin"), ("metformin", "aspirin"), ("lisinopril", "potassium")]
        alerts = []
        for a, b in risky:
            if a in names and b in names:
                alerts.append(f"⚠️ **{a.upper()} + {b.upper()}**: Potential interaction!")
        
        if alerts:
            st.subheader("🚨 Safety Alerts")
            for alert in alerts:
                st.error(alert)
        else:
            st.success("✅ No common interactions detected")

st.markdown("---")
st.caption("⚠️ Disclaimer: This is not medical advice. Always consult your doctor.")
st.caption("Built for Build with Gemma: GDG TIU Buildathon | Track: AI for Healthcare")
