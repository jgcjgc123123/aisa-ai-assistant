import streamlit as st
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="😼", layout="wide")

# Move SYSTEM_PROMPT here so sidebar buttons can use it
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

# 2. Sidebar Layout
with st.sidebar:
    st.subheader("😼 Aisa does not handle [enrollment](https://cit.edu/enrollment/) or [payments](https://cit.edu/payment-options/)!")
        
    st.markdown("---")
    
    # Study Modes Feature
    st.subheader("🎯 Study Modes")
    study_topic = st.text_input("What topic are we focusing on?", placeholder="e.g., OSI Model, Subnetting")
    
    # Buttons are now stacked vertically
    quiz_btn = st.button("Generate Quiz", use_container_width=True)
    flashcard_btn = st.button("Generate Flashcards", use_container_width=True)
    
    if quiz_btn:
        if study_topic:
            user_msg = f"Let's start a quiz on {study_topic}. Ask me the first question to test my knowledge. Wait for my answer before asking the next one."
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            with st.spinner("Starting quiz..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
                    
                    # Build history for the quiz button
                    formatted_history = []
                    for msg in st.session_state.messages[:-1]:
                        role = "model" if msg["role"] == "assistant" else "user"
                        formatted_history.append({"role": role, "parts": [msg["content"]]})
                    
                    formatted_history.append({"role": "user", "parts": [user_msg]})
                    
                    response = model.generate_content(formatted_history)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error("Aisa is overloaded! Please wait a minute and try again. ⏳")
        else:
            st.warning("Please enter a topic first!")
            
    if flashcard_btn:
        if study_topic:
            user_msg = f"Can you give me 5 study flashcards for {study_topic}?"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            with st.spinner("Generating flashcards..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
                    
                    flashcard_prompt = f"""
                    You are a helpful tutor. Provide 5 study flashcards about {study_topic}. 
                    Format EACH flashcard strictly using this exact HTML structure:
                    <details style="background-color: #2D2D2D; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444;">
                      <summary style="font-weight: bold; cursor: pointer; font-size: 16px;">💡 Question: [Your Question]</summary>
                      <p style="margin-top: 15px; color: #E0E0E0; font-size: 15px;">[Your Answer]</p>
                    </details>
                    Do not include markdown code blocks. Just output the raw HTML.
                    """
                    
                    response = model.generate_content(flashcard_prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error("Aisa is overloaded! Please wait a minute and try again. ⏳")
        else:
            st.warning("Please enter a topic first!")
            
    st.markdown("---")
    
    # Quick Links Feature
    st.subheader("🔗 Quick Links")
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
    st.caption("March 2026 ©")

# 3. API Configuration
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Aisa is offline 😴")
    st.stop()

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

# Chat Input with built-in file upload
if prompt := st.chat_input("How can I help with your studies today?", accept_file=True, file_type=["txt", "pdf", "png", "jpg", "jpeg"]):
    
    user_text = prompt.text
    user_files = prompt.files
    
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
        try:
            model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
            
            # Build history for the main chat
            formatted_history = []
            for msg in st.session_state.messages[:-1]:
                role = "model" if msg["role"] == "assistant" else "user"
                formatted_history.append({"role": role, "parts": [msg["content"]]})
                
            current_parts = []
            if user_text:
                current_parts.append(user_text)
                
            if user_files:
                for f in user_files:
                    current_parts.append({
                        "mime_type": f.type,
                        "data": f.getvalue()
                    })
                    
            formatted_history.append({"role": "user", "parts": current_parts})
            
            response = model.generate_content(formatted_history)
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        except Exception as e:
            st.session_state.messages.pop()
            st.error("Aisa is overloaded! Please wait a minute and try again. ⏳")
