import streamlit as st
import ollama
import json
from PIL import Image

# 1. UI Configuration
st.set_page_config(page_title="Vani-Check AI", page_icon="🌾", layout="wide")

# Custom CSS for a "Mandi-Tech" look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_stdio=True)

# 2. Mock Data (From your Agmarknet Screenshot)
MANDI_DATA = {
    "Tomato": {"price": 14.32, "trend": "Down 📉", "arrival": "2505 MT"},
    "Potato": {"price": 4.96, "trend": "Up 📈", "arrival": "1350 MT"},
    "Onion": {"price": 13.50, "trend": "Stable ⚖️", "arrival": "1280 MT"}
}

# 3. Sidebar for System Status
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("🟢 Ollama: Llama-3 Active")
    st.info("📡 Mode: 100% Offline")
    st.divider()
    selected_item = st.selectbox("Select Commodity", list(MANDI_DATA.keys()))

# 4. Main Dashboard Layout
st.title("🌾 Vani-Check: Multimodal Agri-Auditor")
st.caption("Empowering rural vendors with Edge-AI Pricing Intelligence")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📸 Scan Produce")
    # This captures the image for the YOLO model
    img_file = st.camera_input("Place produce in front of camera")
    
    if img_file:
        st.image(img_file, caption="Processing Frame...", use_container_width=True)
        # Placeholder for Member 1's YOLO Grade
        quality_score = 82 
        st.progress(quality_score/100, text=f"Quality Score: {quality_score}/100")

with col2:
    st.subheader("📊 Market Intelligence")
    data = MANDI_DATA[selected_item]
    
    m1, m2 = st.columns(2)
    m1.metric("Mandi Price", f"₹{data['price']}/kg", data['trend'])
    m2.metric("Daily Arrival", data['arrival'])
    
    st.divider()
    
    st.subheader("🤖 Vani-AI Advice")
    if img_file:
        with st.spinner("Llama-3 is reasoning..."):
            # Constructing the Prompt for Ollama
            prompt = f"As a Mandi expert, give 1 short Hinglish sentence advice for {selected_item} with quality {quality_score}/100 and price {data['price']}."
            
            try:
                response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
                st.chat_message("assistant").write(response['message']['content'])
            except Exception as e:
                st.error("Ollama connection failed. Ensure 'ollama serve' is running.")
    else:
        st.write("Waiting for camera input to generate advice...")

# 5. Footer
st.divider()
st.caption("Developed for Hackathon 2026 | Optimized for AMD Ryzen 7 8000")