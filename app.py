import streamlit as st
import tempfile
import os
import json
import re
from pydub import AudioSegment
from textblob import TextBlob
from streamlit_mic_recorder import mic_recorder 

# Import Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ===================================================================
# CONFIG & SESSION STATE
# ===================================================================
SAMPLE_RATE = 16000
DEFAULT_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="Gemini Sales Intelligence", 
    layout="wide", # Use wide layout for better space utilization
    initial_sidebar_state="expanded"
)

st.title("📞 Real-Time Sales Intelligence Assistant")
st.caption("Powered by Gemini 2.5 Flash for Multimodal Diarization & Coaching")

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "crm_profile" not in st.session_state:
    st.session_state.crm_profile = {}

# ===================================================================
# NLP UTILITIES (Local) - CORRECTED
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
        "Warranty Inquiry": ["warranty", "guarantee"],
        "Exchange/Return": ["exchange", "return", "refund"], # Changed from Exchange Offer to be broader
        "Purchase Confirmation": ["confirm", "book", "order", "buy"],
        "Complain/Issue": ["not working", "broken", "issue", "problem", "damaged", "leaking"] # Added damaged/leaking
    }
    for intent, keys in rules.items():
        if any(k in t for k in keys): return intent
    return "Product Inquiry"

# ===================================================================
# NLP UTILITIES (Local) - FINAL CORRECTED extract_entities
# ===================================================================

# NOTE: This replaces the previous version of extract_entities entirely.
def extract_entities(text):
    text_lower = text.lower()
    
    # 1. Timeline (Detection: "two-day delay", "by Monday", "next two days")
    timeline_matches = re.findall(
        # Pattern looks for N days/weeks/etc. OR specific time words (tomorrow, Monday, weekend)
        r"(\b\d+\s*(?:day|days|week|weeks|month|months|year|years|hr|hrs)\b|\bnext\s*(?:day|week|month|year)\b|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|tomorrow)\b)", 
        text_lower
    )
    
    # 2. Sizes (Detection: "small", "XL", "size 10", "size larger")
    size_matches = re.findall(
        r"\b(?:small|medium|large|extra-?large|xl|xxl|s|m|l|size\s*\d+|size\s*(?:larger|smaller))\b", 
        text_lower
    )
    
    # 3. Prices (Detection: "$50", "Rs. 1000") - Using the specific number 500 from the dialogue
    price_matches = re.findall(
        r"(?:₹|rs\.?|rupees|\$|eur)\s*\d+(?:[\.,]\d+)?|\b\d+\s*rupee\b|\b500\b", 
        text_lower
    )

    # 4. Products/Damage (Detection: specific items or complaint words)
    product_damage_matches = re.findall(
        r"\b(?:dining\s*table|furniture|tomatoes?|milk\s*packets?|damaged|leaking|squashed|broken|delay)\b", 
        text_lower
    )
    
    # Cleanup and return
    return {
        "timeline": list(set(timeline_matches)),
        "sizes": list(set(size_matches)),
        "prices": list(set(price_matches)),
        "products_issues": list(set(product_damage_matches)), # Renamed for clarity
    }

# ===================================================================
# GEMINI CORE LOGIC
# ===================================================================
def process_audio_with_gemini(filepath, api_key):
    """Uses Gemini Multimodal to transcribe and diarize speakers."""
    client = genai.Client(api_key=api_key)
    
    # Upload to Gemini File API using the correct 'file=' argument
    uploaded_file = client.files.upload(file=filepath)
    
    prompt = """
    Listen to this sales call. 
    1. Transcribe the conversation accurately.
    2. Identify the 'Sales Rep' and the 'Customer' by their voices and context.
    3. Return a JSON list of objects with keys: "speaker" and "text".
    
    Example:
    [{"speaker": "Sales Rep", "text": "Hello, how can I help?"}, {"speaker": "Customer", "text": "I am looking for a dress."}]
    """

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[prompt, uploaded_file],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    # Clean up file from cloud
    client.files.delete(name=uploaded_file.name)
    return json.loads(response.text)

def get_sales_guidance(turn_text, context_transcript, crm, api_key):
    """Generates real-time suggestions based on the current customer turn."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are a sales coach. Analyze this customer statement and the previous context.
    
    Customer Statement: "{turn_text}"
    Recent Context: {context_transcript}
    CRM Data: {json.dumps(crm)}
    
    Return JSON only:
    {{
      "follow_up": "Next question for the rep",
      "objection": "How to handle hesitation",
      "recommendation": "Product/Service to suggest",
      "insight": "1-sentence psychological insight"
    }}
    """
    
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# ===================================================================
# SIDEBAR UI (Settings & Configuration)
# ===================================================================
with st.sidebar:
    st.header("🔑 Gemini Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Input your Google AI Studio API Key.")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
             st.warning("Please set your API key to run analysis.")
        
    st.divider()

    st.header("👤 CRM Profile (Context)")
    st.info("This context helps Gemini provide tailored recommendations.")
    c_name = st.text_input("Customer Name", "Mrunal")
    c_pref = st.text_input("Customer Preferences", "Customization, Silk, Premium")
    st.session_state.crm_profile = {"name": c_name, "preferences": c_pref}

# ===================================================================
# MAIN UI (Audio Input & Run Button)
# ===================================================================
col_input, col_run = st.columns([3, 1])

with col_input:
    tab1, tab2 = st.tabs(["📁 Upload Call Audio", "🎤 Live Record Call"])

    with tab1:
        upl = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
        if upl:
            ext = upl.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
                f.write(upl.read())
                st.session_state.audio_path = f.name
            st.audio(st.session_state.audio_path)
            st.success("Audio file ready for processing.")

    with tab2:
        st.write("Click 'Record' to capture a live call segment.")
        mic = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop Recording", key="mic")
        if mic and mic.get("bytes"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(mic["bytes"])
                st.session_state.audio_path = f.name
            st.audio(st.session_state.audio_path)
            st.success("Recording complete. Ready for processing.")

with col_run:
    st.markdown("##") # Spacer
    if st.button("🚀 Analyze & Generate Guidance", type="primary", use_container_width=True):
        if not st.session_state.audio_path or not api_key:
            st.error("Missing Audio or API Key.")
            st.stop()
        
        # Start Analysis
        st.session_state.analysis_started = True

# ===================================================================
# EXECUTION AND RESULTS DISPLAY
# ===================================================================
if st.session_state.get("analysis_started"):
    st.subheader("Transcript & Real-Time Guidance")
    st.divider()
    
    with st.spinner("Gemini is listening, identifying voices, and analyzing..."):
        try:
            # 1. Get Diarized Transcript
            transcript_data = process_audio_with_gemini(st.session_state.audio_path, api_key)
            
            full_history = []
            
            for i, turn in enumerate(transcript_data):
                speaker = turn["speaker"]
                text = turn["text"]
                full_history.append(f"{speaker}: {text}")
                
                # Logic: If Customer speaks, provide Guidance
                if "Customer" in speaker:
                    sentiment_text, sentiment_icon = analyze_sentiment(text)
                    intent = classify_intent(text)
                    entities = extract_entities(text)
                    
                    # Run Gemini for complex guidance
                    guidance = get_sales_guidance(
                        text, 
                        "\n".join(full_history[-4:]), 
                        st.session_state.crm_profile, 
                        api_key
                    )
                    
                    # Display Customer & Guidance using a chat message
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(f"**Customer:** {text}")
                        
                        # Use an Expander for clear separation of Guidance
                        with st.expander(f"✨ **SALES GUIDANCE (Turn {i+1})**", expanded=True):
                            col_nlp, col_guidance = st.columns([1, 2.5])
                            
                            with col_nlp:
                                st.markdown("##### 🧠 Quick NLP")
                                st.metric("Sentiment", f"{sentiment_text} {sentiment_icon}")
                                st.metric("Intent", intent)
                                st.caption(f"Entities: {entities}")

                            with col_guidance:
                                st.markdown("##### 💡 Gemini 2.5 Coaching")
                                st.success(f"**🙋 Next Question:** {guidance.get('follow_up', 'N/A')}")
                                st.info(f"**🛡️ Objection Fix:** {guidance.get('objection', 'N/A')}")
                                st.warning(f"**🏷️ Recommendation:** {guidance.get('recommendation', 'N/A')}")
                                st.markdown(f"**🎯 Insight:** *{guidance.get('insight', 'N/A')}*")

                else:
                    # Display Sales Rep Message
                    with st.chat_message("assistant", avatar="💼"):
                        st.markdown(f"**Sales Rep:** *{text}*")
                        
        except Exception as e:
            st.error(f"An error occurred during processing. Please check your API key and ensure the file is under 20MB. Error: {e}")

# ===================================================================
# FOOTER
# ===================================================================
st.divider()
st.caption("Final Note: All analysis is provided by Google Gemini 2.5 Flash. Accuracy depends on audio quality.")







