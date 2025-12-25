import streamlit as st
import tempfile
import os
import json
import re
from pydub import AudioSegment
from textblob import TextBlob
from streamlit_mic_recorder import mic_recorder 
from google import genai
from google.genai import types

# ===================================================================
# CONFIG & SESSION STATE
# ===================================================================
DEFAULT_MODEL = "gemini-2.0-flash"

st.set_page_config(
    page_title="Gemini Sales Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ===================================================================
# LOCAL NLP UTILITIES
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
        "Complain/Issue": ["not working", "broken", "issue", "problem", "damaged", "leaking", "delay"]
    }
    for intent, keys in rules.items():
        if any(k in t for k in keys): return intent
    return "Product Inquiry"

def extract_entities(text):
    t_lower = text.lower()
    return {
        "timeline": re.findall(r"(\b\d+\s*(?:day|days|week|hr|hrs)\b|\bnext\s*(?:day|week)\b|\btomorrow\b)", t_lower),
        "prices": re.findall(r"(?:₹|rs\.?|rupees|\$)\s*\d+", t_lower),
        "issues": re.findall(r"\b(?:damaged|leaking|broken|delay|missing|tight|large)\b", t_lower)
    }

# ===================================================================
# GEMINI CORE LOGIC (Batch Optimized)
# ===================================================================
def process_audio_with_gemini(filepath, api_key):
    """Step 1: Get Diarized Transcript (1 API Call)"""
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

def get_full_analysis(transcript, crm, api_key):
    """Step 2: Get All Guidance in One Batch (1 API Call)"""
    client = genai.Client(api_key=api_key)
    formatted_transcript = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
    
    prompt = f"""
    You are a sales coach. Analyze this full transcript.
    CRM Context: {json.dumps(crm)}
    
    Transcript:
    {formatted_transcript}
    
    For every time the 'Customer' spoke, provide exactly 4 points:
    'follow_up', 'objection', 'recommendation', 'insight'.
    Return a JSON list of objects matching the sequence of customer turns only.
    """
    
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# ===================================================================
# UI LAYOUT
# ===================================================================
st.title("📞 Real-Time Sales Intelligence Assistant")
st.caption("Batch Processing Mode: Maximum Insights, Minimum API Quota Usage")

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.header("👤 CRM Context")
    c_name = st.text_input("Customer Name", "Mrunal")
    c_pref = st.text_input("Preferences", "Premium, Customization, Silk")
    crm_profile = {"name": c_name, "prefs": c_pref}

# Audio Input Section
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Live Record"])

with tab1:
    upl = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    if upl:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(upl.read())
            st.session_state.audio_path = f.name
        st.success("File Uploaded!")

with tab2:
    mic = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop", key="mic")
    if mic:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(mic["bytes"])
            st.session_state.audio_path = f.name
        st.success("Recording Captured!")

# Processing Trigger
if st.button("🚀 Analyze Call", type="primary", use_container_width=True):
    if not api_key or not st.session_state.audio_path:
        st.error("Missing API Key or Audio Source.")
    else:
        with st.spinner("Step 1: Transcribing and Identifying Voices..."):
            try:
                transcript = process_audio_with_gemini(st.session_state.audio_path, api_key)
                
                with st.spinner("Step 2: Generating Batch Sales Guidance..."):
                    all_guidance = get_full_analysis(transcript, crm_profile, api_key)
                    st.session_state.analysis_results = (transcript, all_guidance)
                    st.toast("Analysis Complete!", icon="✅")
            except Exception as e:
                st.error(f"Error: {e}")

# ===================================================================
# DISPLAY RESULTS
# ===================================================================
if st.session_state.analysis_results:
    transcript, all_guidance = st.session_state.analysis_results
    st.subheader("Conversation Analysis")
    st.divider()

    guidance_idx = 0
    for turn in transcript:
        speaker = turn["speaker"]
        text = turn["text"]
        
        if "Customer" in speaker:
            sent, icon = analyze_sentiment(text)
            intent = classify_intent(text)
            ents = extract_entities(text)
            
            # Map batch guidance to the turn
            coach = all_guidance[guidance_idx] if guidance_idx < len(all_guidance) else {}
            guidance_idx += 1

            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Customer:** {text}")
                with st.expander("✨ Sales Intelligence & Guidance", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Sentiment", f"{sent} {icon}")
                        st.caption(f"**Intent:** {intent}")
                        st.caption(f"**Entities:** {ents}")
                    with col2:
                        st.success(f"**🙋 Next Question:** {coach.get('follow_up', 'N/A')}")
                        st.info(f"**🛡️ Objection Fix:** {coach.get('objection', 'N/A')}")
                        st.warning(f"**🏷️ Recommendation:** {coach.get('recommendation', 'N/A')}")
                        st.markdown(f"**🎯 Insight:** *{coach.get('insight', 'N/A')}*")
        else:
            with st.chat_message("assistant", avatar="💼"):
                st.markdown(f"**Sales Rep:** *{text}*")

st.divider()
st.caption("Final Note: This analysis uses exactly 2 API calls per session.")
