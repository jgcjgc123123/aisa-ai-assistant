import streamlit as st
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="😼", layout="wide")

# 2. Sidebar Layout
with st.sidebar:
    st.title("😼 Aisa Settings")
    st.markdown("---")
    st.info("Aisa is your smart upperclassman for Applied AI, Capstone, and Networking.")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Built for CIT-U Technologians ⚡")

# 3. API Configuration
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Aisa is offline 😴")
    st.stop()

SYSTEM_PROMPT = """
You are Aisa (Applied Intelligence Studies Assistant), a dedicated AI companion for students at Cebu Institute of Technology - University.
Your tone is human-like, supportive, and slightly casual—like a smart upperclassman. 
You are an expert in app development, coding, and general IT courses and subjects. 

Key guidelines:
1. Be concise but insightful.
2. Use relatable student language, but stay professional enough.
3. If asked about CIT-U specifically, show school spirit (Technologian pride!).
4. Always prioritize clarity in technical explanations.
"""

# 4. Top Header & Stats Layout
st.title("😼 Aisa AI")
col1, col2, col3 = st.columns(3)
col1.metric("Status", "Online")
col2.metric("Model", "Gemini 2.5 Flash")
col3.metric("School", "CIT-U")
st.markdown("---")

# 5. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help with your studies today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Simple string concatenation for the system prompt
        response = model.generate_content(f"{SYSTEM_PROMPT}\n\nUser: {prompt}")
        st.markdown(response.text)
        
    st.session_state.messages.append({"role": "assistant", "content": response.text})
