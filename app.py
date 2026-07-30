import streamlit as st
import numpy as np
import joblib
import os
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up page styling and configurations
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f9fbfd;
    }
    .stButton>button {
        background-color: #2b5c8f;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1e4366;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 AI Fake News Detector")
st.markdown("""
This application analyzes news text using a trained **LSTM Neural Network** 
to classify whether the input content is **Real** or **Fake**.
""")

# ---- Text Preprocessing Helper Function ----
def clean_text(text):
    text = text.lower()  # Lowercase conversion
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text.strip()

# ---- Model & Asset Loading Pipeline ----
@st.cache_resource
def load_assets():
    assets = {"model": None, "tokenizer": None, "columns": None, "error": None}
    try:
        # Get exact directory path of app.py
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "model_ann.h5")
        tokens_path = os.path.join(BASE_DIR, "tokenizer.pkl")
        column_path = os.path.join(BASE_DIR, "columns.pkl")
        
        # Check if all files exist
        missing_files = [f for f in [model_path, tokens_path, column_path] if not os.path.exists(f)]
        if missing_files:
            missing_names = [os.path.basename(f) for f in missing_files]
            raise FileNotFoundError(f"Missing required pipeline files: {', '.join(missing_names)}")

        from tensorflow.keras.models import load_model
        
        # Load artifacts
        assets["model"] = load_model(model_path)
        assets["tokenizer"] = joblib.load(tokens_path)
        assets["columns"] = joblib.load(column_path)
        
    except FileNotFoundError as fnf_error:
        assets["error"] = f"📁 File Error: {fnf_error}"
    except Exception as e:
        assets["error"] = f"❌ Error loading assets: {str(e)}"
        
    return assets

# Load assets
assets = load_assets()

# ---- User Input Area ----
news_text = st.text_area(
    "Paste the news headline or short news content below:", 
    height=180, 
    placeholder="Enter short news headline or 1-2 sentences here..."
)

# Predict Action Trigger
if st.button("Verify News Authenticity", use_container_width=True):
    if assets["error"] is not None:
        st.error(assets["error"])
        st.warning("⚠️ Application pipeline cannot process requests because files failed to load.")
    elif assets["model"] is None or assets["tokenizer"] is None:
        st.error("❌ Critical pipeline error: Architecture artifacts are missing or corrupted.")
    elif not news_text.strip():
        st.warning("⚠️ Input validation failed: Please enter or paste text inside the content box first!")
    else:
        with st.spinner("LSTM Model is computing token sequence probabilities..."):
            try:
                # 1. Clean the raw text using preprocessing rules
                cleaned_input = clean_text(news_text)
                
                # 2. Text to Sequence via Keras Tokenizer
                sequences = assets["tokenizer"].texts_to_sequences([cleaned_input])
                
                # 3. Sequence Padding (MAX_LEN ko training ke waqt rakhe gaye size se match karein)
                MAX_LEN = 200
                processed_text = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
                
                # 4. Model Prediction
                prediction = assets["model"].predict(processed_text)
                prediction_prob = float(prediction[0][0]) if hasattr(prediction[0], "__len__") else float(prediction[0])
                
                # Show Raw Score for debugging
                st.info(f"📊 **Raw Model Output Score:** `{prediction_prob:.4f}`")

                # Decision Logic (Assuming > 0.5 is REAL, <= 0.5 is FAKE)
                # Agar score ulta ho, toh condition ko (prediction_prob <= 0.5) kar lein.
                is_real = prediction_prob > 0.5
                
                st.success("✨ Analysis Completed Successfully!")
                
                # Render results dynamically
                if is_real:
                    st.balloons()
                    st.markdown("""
                    <div style="background-color:#e6f4ea; padding:22px; border-radius:10px; border-left: 8px solid #137333; margin-top:15px;">
                        <h3 style="color:#137333; margin:0; font-size: 20px;">✅ Authentic Content (REAL)</h3>
                        <p style="color:#1d1d1d; margin:10px 0 0 0; font-size:15px;">According to our LSTM Deep Learning model, the sequence patterns match credible news structures.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color:#fce8e6; padding:22px; border-radius:10px; border-left: 8px solid #c5221f; margin-top:15px;">
                        <h3 style="color:#c5221f; margin:0; font-size: 20px;">🚨 Suspicious Content (FAKE)</h3>
                        <p style="color:#1d1d1d; margin:10px 0 0 0; font-size:15px;">Warning! The Deep Learning model detected patterns strongly associated with unverified or fake text.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as pred_error:
                st.error(f"⚙️ Runtime Prediction Error: {str(pred_error)}")

# Visual Status Bar
if assets["error"] is not None:
    st.sidebar.error("System Status: Offline")
    st.sidebar.info(assets["error"])
else:
    st.sidebar.success("System Status: Operational")
    st.sidebar.write("• Model: `model_ann.h5` Loaded")
    st.sidebar.write("• Tokenizer: `tokenizer.pkl` Loaded")
    st.sidebar.write("• Columns: `columns.pkl` Loaded")

# Footer 
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 13px;'>LSTM Deep Learning Infrastructure Layer</p>", unsafe_allow_html=True)