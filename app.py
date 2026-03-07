import streamlit as st
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="🎓")

# 1. Setup API Key (Pulling from Streamlit Secrets)
# Go to Streamlit Cloud Settings > Secrets to add: GEMINI_API_KEY = "your_key"
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Please add your Gemini API Key to Streamlit Secrets!")
    st.stop()

# 2. Define Aisa's Personality (The System Prompt)
SYSTEM_PROMPT = """
You are Aisa (Applied Intelligence Studies Assistant), a helpful and 
knowledgeable AI assistant for students at CIT-U. You are encouraging, 
human-like, and expert in Applied AI and Networking.
"""

st.title("🎓 Aisa AI")
st.caption("Your Applied AI Studies Assistant")

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input Logic
if prompt := st.chat_input("How can I help with your studies today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        model = genai.GenerativeModel('gemini-1.5-flash')
        # We send the system prompt + the user's message
        response = model.generate_content(f"{SYSTEM_PROMPT}\n\nUser: {prompt}")
        st.markdown(response.text)
        
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.text})
