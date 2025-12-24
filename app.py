import streamlit as st
import tempfile
import os
import json
import re
from pydub import AudioSegment
from textblob import TextBlob
from streamlit_mic_recorder import mic_recorder

# Use the NEW SDK explicitly
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ===================================================================
# CONFIG & SESSION STATE
# ===================================================================
DEFAULT_MODEL = "gemini-2.0-flash"

st.set_page_config(
    page_title="Gemini Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📞 Real-Time Sales Intelligence Assistant")
st.caption("Powered by Gemini 2.0 Flash for Multimodal Diarization & Coaching")

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

# ===================================================================
# NLP UTILITIES (Local)
# ===================================================================
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1: return "POSITIVE", "🟢"
    if score < -0.1: return "NEGATIVE", "🔴"
    return "NEUTRAL", "🟡"

def classify_intent(text):
    t = text.lower()
    rules = {
        "Price Objection": ["expensive", "cost", "price", "budget"],
        "Exchange/Return": ["exchange", "return", "refund"],
        "Complain/Issue": ["not working", "broken", "issue", "problem", "damaged", "leaking"]
    }
    for intent, keys in rules.items():
        if any(k in t for k in keys): return intent
    return "Product Inquiry"

def extract_entities(text):
    t_lower = text.lower()
    return {
        "timeline": re.findall(r"(\b\d+\s*(?:day|days|week|hr|hrs)\b|\bnext\s*(?:day|week)\b)", t_lower),
        "prices": re.findall(r"(?:₹|rs\.?|rupees|\$)\s*\d+", t_lower),
        "issues": re.findall(r"\b(?:damaged|leaking|broken|delay|missing)\b", t_lower)
    }

# ===================================================================
# GEMINI CORE LOGIC
# ===================================================================
def process_audio_with_gemini(filepath, api_key):
    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=filepath)
    
    prompt = """
    Listen to this sales call. 
    1. Transcribe the conversation accurately.
    2. Identify 'Sales Rep' and 'Customer' by voice/context.
    3. Return a JSON list of objects with keys: "speaker" and "text".
    """

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        client.files.delete(name=uploaded_file.name)
        return json.loads(response.text)
    except Exception as e:
        client.files.delete(name=uploaded_file.name)
        raise e

def get_sales_guidance(turn_text, context, crm, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"Coach this rep. Customer said: {turn_text}. History: {context}. CRM: {json.dumps(crm)}. Return JSON with keys: follow_up, objection, recommendation, insight."
    
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# ===================================================================
# SIDEBAR UI
# ===================================================================
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("👤 CRM Context")
    c_name = st.text_input("Customer Name", "Mrunal")
    c_pref = st.text_input("Preferences", "Premium, Customization")
    crm_profile = {"name": c_name, "prefs": c_pref}

# ===================================================================
# MAIN UI
# ===================================================================
tab1, tab2 = st.tabs(["📁 Upload", "🎤 Record"])

with tab1:
    upl = st.file_uploader("Audio File", type=["wav", "mp3"])
    if upl:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(upl.read())
            st.session_state.audio_path = f.name

with tab2:
    mic = mic_recorder(start_prompt="🎙️ Start", stop_prompt="⏹️ Stop", key="mic")
    if mic:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(mic["bytes"])
            st.session_state.audio_path = f.name

if st.button("🚀 Analyze Call"):
    if st.session_state.audio_path and api_key:
        st.session_state.analysis_started = True
    else:
        st.error("Provide API Key and Audio.")

if st.session_state.analysis_started:
    with st.spinner("Processing..."):
        try:
            transcript = process_audio_with_gemini(st.session_state.audio_path, api_key)
            history = []
            for i, turn in enumerate(transcript):
                speaker, text = turn["speaker"], turn["text"]
                history.append(f"{speaker}: {text}")
                
                if "Customer" in speaker:
                    sent, icon = analyze_sentiment(text)
                    guidance = get_sales_guidance(text, history[-3:], crm_profile, api_key)
                    
                    with st.chat_message("user", avatar="👤"):
                        st.write(text)
                        with st.expander("✨ Guidance"):
                            st.write(f"**Sentiment:** {sent} {icon}")
                            st.info(f"💡 {guidance.get('insight')}")
                            st.success(f"🙋 {guidance.get('follow_up')}")
                else:
                    with st.chat_message("assistant", avatar="💼"):
                        st.write(text)
        except Exception as e:
            st.error(f"Error: {e}")
