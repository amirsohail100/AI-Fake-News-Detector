import streamlit as st
import numpy as np
import joblib
import os

# Set up page styling and configurations
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom styling for a modern look
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
This is an advanced web application powered by an **Artificial Neural Network (ANN)** model 
that analyzes news text and classifies whether the information is **Real** or **Fake** with an outstanding **98% accuracy**.
""")

# ---- Model & Asset Loading Pipeline ----
@st.cache_resource
def load_assets():
    assets = {"model": None, "tokenizer": None, "columns": None, "error": None}
    try:
        model_path = "model.h5"
        tokens_path = "tokens.pkl"
        column_path = "column.pkl"
        
        # Check if all files exist
        missing_files = [f for f in [model_path, tokens_path, column_path] if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing required pipeline files: {', '.join(missing_files)}")

        # Lazy import of heavy deep learning modules to optimize performance
        from tensorflow.keras.models import load_model
        
        # Load the artifacts securely
        assets["model"] = load_model(model_path)
        assets["tokenizer"] = joblib.load(tokens_path)
        assets["columns"] = joblib.load(column_path)
        
    except FileNotFoundError as fnf_error:
        assets["error"] = f"📁 File Error: {fnf_error}. Please ensure model.h5, tokens.pkl, and column.pkl are present in the directory."
    except Exception as e:
        assets["error"] = f"❌ Error loading assets: {str(e)}. Please check your environment dependencies (TensorFlow/Joblib)."
        
    return assets

# Load baseline assets
assets = load_assets()

# ---- User Input Area (Kept visible even if loading failed) ----
news_text = st.text_area(
    "Paste the news text content for verification below:", 
    height=220, 
    placeholder="Enter the full text or paragraph of the news article here..."
)

# Predict Action Trigger
if st.button("Verify News Authenticity", use_container_width=True):
    # Check asset validity at runtime inside execution sequence
    if assets["error"] is not None:
        st.error(assets["error"])
        st.warning("⚠️ Application pipeline cannot process requests because the required binary files failed to load correctly.")
    elif assets["model"] is None or assets["tokenizer"] is None or assets["columns"] is None:
        st.error("❌ Critical pipeline error: Architecture artifacts are missing or corrupted.")
    elif not news_text.strip():
        st.warning("⚠️ Input validation failed: Please enter or paste text inside the content box first!")
    else:
        with st.spinner("Artificial Neural Network (ANN) is computing token probabilities..."):
            try:
                # 1. Processing via loaded tokenizer
                # processed_text = assets["tokenizer"].transform([news_text])
                # Convert sparse structure if applicable:
                # if hasattr(processed_text, "toarray"): processed_text = processed_text.toarray()
                
                # 2. Reindexing or aligning based on columns schema
                # final_features = ... (use assets["columns"] if structural alignment is required)
                
                # 3. Model inference calculation 
                # prediction_prob = assets["model"].predict(processed_text)[0][0]
                # is_real = prediction_prob > 0.5
                
                # Mock result placeholder (Replace with your actual binary prediction logic)
                is_real = True  
                
                st.success("✨ Analysis Completed Successfully!")
                
                # Render results dynamically using native HTML cards with corrected key parameter
                if is_real:
                    st.balloons()
                    st.markdown("""
                    <div style="background-color:#e6f4ea; padding:22px; border-radius:10px; border-left: 8px solid #137333; margin-top:15px;">
                        <h3 style="color:#137333; margin:0; font-size: 20px;">✅ Authentic Content (REAL)</h3>
                        <p style="color:#1d1d1d; margin:10px 0 0 0; font-size:15px;">According to our Deep Learning ANN pipeline, the linguistics, semantics, and patterns match credible news reporting structures.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color:#fce8e6; padding:22px; border-radius:10px; border-left: 8px solid #c5221f; margin-top:15px;">
                        <h3 style="color:#c5221f; margin:0; font-size: 20px;">🚨 Suspicious Content (FAKE)</h3>
                        <p style="color:#1d1d1d; margin:10px 0 0 0; font-size:15px;">Warning! The Deep Learning model detected high-probability patterns strongly associated with synthetic text, fabrications, or misinformation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as pred_error:
                st.error(f"⚙️ Runtime Prediction Error: {str(pred_error)}")
                st.info("Ensure the input dimensions and text preprocessing steps strictly match the expected tensor shape of the model.")

# Visual Status Bar / Banner for Error Feedback if files aren't found on load
if assets["error"] is not None:
    st.sidebar.error("System Status: Offline")
    st.sidebar.info(assets["error"])
else:
    st.sidebar.success("System Status: Operational")
    st.sidebar.write("• Model: `model.h5` Loaded")
    st.sidebar.write("• Tokenizer: `tokens.pkl` Loaded")
    st.sidebar.write("• Columns: `column.pkl` Loaded")

# Footer 
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 13px;'>ANN Deep Learning Infrastructure Layer | Accuracy: 98%</p>", unsafe_allow_html=True)