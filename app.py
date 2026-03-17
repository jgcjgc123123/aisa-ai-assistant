import streamlit as st
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="😼", layout="wide")

# 2. Sidebar Layout
with st.sidebar:
    st.subheader("😼 Aisa does not handle [enrollment](https://cit.edu/enrollment/) or [payments](https://cit.edu/payment-options/)!")
        
    st.markdown("---")
    
    # Study Modes Feature
    st.subheader("🎯 Study Modes")
    study_topic = st.text_input("What topic are we focusing on?", placeholder="e.g., OSI Model, Subnetting")
    
    quiz_mode = st.toggle("Enable Quiz Mode")
    
    if st.button("Generate Flashcards", use_container_width=True):
        if study_topic:
            user_msg = f"Can you give me 5 study flashcards for {study_topic}? Format them clearly with Q: and A:"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            with st.spinner("Generating flashcards..."):
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(f"You are a helpful tutor.\n\nUser: {user_msg}")
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        else:
            st.warning("Please enter a topic first!")

    st.markdown("---")
    
    # Resource Links Feature
    st.subheader("🔗 Resource Links")
    st.markdown("- [CIT-U Homepage](https://cit.edu/)")
    st.markdown("- [Academic Calendar 25-26](https://cit.edu/collegiate-calendar-for-academic-year-2025-2026/)")
    st.markdown("- [Vision-Mission](https://cit.edu/cit-vision-mission-primer/)")
    st.markdown("- [College Programs](https://cit.edu/cit-university-programs/)")
    st.markdown("- [LAIR](https://lair.education/)")
    st.markdown("- [WITS](https://student.cituwits.com/)")
    
    st.markdown("---")

    # Download Notes Feature
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_history = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="📥 Download Notes as TXT",
            data=chat_history,
            file_name="aisa_study_notes.txt",
            mime="text/plain",
            use_container_width=True
        )
    
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

# Adjust prompt if Quiz Mode is active
if quiz_mode:
    if study_topic:
        SYSTEM_PROMPT += f"\n\nQUIZ MODE ACTIVE: The user is studying '{study_topic}'. Ask them one question at a time to test their knowledge. Wait for their answer, evaluate it, and then ask the next question."
    else:
        SYSTEM_PROMPT += "\n\nQUIZ MODE ACTIVE: Ask the user what specific topic they want to be quizzed on, then start asking them questions about it one by one."

# 4. Top Header & Stats Layout
st.title("😼 Aisa AI")
st.caption("Your Applied AI Studies Assistant")

# 5. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with custom HTML
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #0078D7; color: white; padding: 10px 15px; border-radius: 15px 15px 0px 15px; margin-bottom: 10px; max-width: 75%;'>
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start;'>
                <div style='background-color: #2D2D2D; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 0px; margin-bottom: 10px; max-width: 75%;'>
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Chat Input with built-in file upload (Requires Streamlit 1.43.0+)
if prompt := st.chat_input("How can I help with your studies today?", accept_file=True, file_type=["txt", "pdf", "png", "jpg", "jpeg"]):
    
    # Extract text and files from the new prompt object
    user_text = prompt.text
    user_files = prompt.files
    
    # Define what to show in the chat history
    display_text = user_text if user_text else "📎 Attached files"
    
    st.session_state.messages.append({"role": "user", "content": display_text})
    
    st.markdown(
        f"""
        <div style='display: flex; justify-content: flex-end;'>
            <div style='background-color: #0078D7; color: white; padding: 10px 15px; border-radius: 15px 15px 0px 15px; margin-bottom: 10px; max-width: 75%;'>
                {display_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.spinner("Aisa is thinking..."):
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare parts for Gemini
        parts = [SYSTEM_PROMPT]
        if user_text:
            parts.append(user_text)
            
        if user_files:
            for f in user_files:
                parts.append({
                    "mime_type": f.type,
                    "data": f.getvalue()
                })
                
        contents = [{"role": "user", "parts": parts}]
        response = model.generate_content(contents)
        
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    
    st.rerun()
