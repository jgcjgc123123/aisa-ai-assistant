import os
import streamlit as st

# 1. Langfuse Environment Configuration
if "LANGFUSE_PUBLIC_KEY" in st.secrets:
    os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
    os.environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
    os.environ["LANGFUSE_HOST"] = st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    os.environ["LANGFUSE_MAX_BATCH_SIZE"] = "1"
    os.environ["LANGFUSE_FLUSH_INTERVAL_MS"] = "100"

from langfuse.decorators import observe, langfuse_context
from groq import Groq
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import re
import io
import requests
import json
import pdfplumber

# 2. Page Configuration
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="😼", layout="wide")

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

ADMIN_KEYWORDS = [r"\benroll", r"\benrollment", r"\btuition", r"\bpay", r"\bpayment", r"\bfee", r"\bfees", r"\bcost"]

# 3. API & Database Configuration
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Aisa is offline (Groq API Key missing) 😴")
    st.stop()

if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    st.error("Aisa is offline (Supabase credentials missing) 😴")
    st.stop()

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")

# 4. Centralized LLM Generation & Tracing
@observe(as_type="generation", name="aisa-groq-call")
def generate_llm_response(messages, json_mode=False, temperature=0.5):
    langfuse_context.update_current_observation(
        input=messages,
        model="llama-3.3-70b-versatile"
    )
    
    kwargs = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": temperature
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    response = client.chat.completions.create(**kwargs)
    reply_text = response.choices[0].message.content
    
    if response.usage:
        langfuse_context.update_current_observation(
            usage={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            }
        )
        
    langfuse_context.update_current_observation(output=reply_text)
    return reply_text

# 5. AI Tools Configuration (Embeddings & Database Logic)
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

@observe(as_type="generation", name="process-pdf-supabase")
def process_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted: 
                text += extracted + "\n"
    
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 250)]
    
    if not chunks:
        return False
        
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    
    # Package the data for Supabase insertion
    data_to_insert = []
    for i, chunk in enumerate(chunks):
        data_to_insert.append({
            "document_name": uploaded_file.name,
            "chunk_content": chunk,
            "embedding": embeddings[i].tolist() # Supabase requires standard python lists
        })
        
    # Push to cloud database
    supabase.table("document_chunks").insert(data_to_insert).execute()
    return True

@observe(as_type="generation", name="retrieve-supabase-context")
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])[0].tolist()
    
    # Call the SQL function in Supabase
    response = supabase.rpc(
        "match_chunks", 
        {
            "query_embedding": query_vec,
            "match_threshold": 0.4, # 40% similarity minimum threshold
            "match_count": top_k
        }
    ).execute()
    
    if response.data:
        return "\n...\n".join([doc["chunk_content"] for doc in response.data])
    
    return ""

@observe(as_type="generation", name="agent-routing")
def agent_decide_search(user_message):
    prompt = f"""
    You are an AI routing agent determining if a user's prompt requires a live web search.
    The user has access to a dedicated static knowledge base (uploaded reference files, project structures, and study documents).
    
    Strict Rules:
    1. Set "needs_search" to false if the question can be resolved via an internal project reference, project team listing, course syllabus, or static educational materials.
    2. Set "needs_search" to true ONLY if the question explicitly demands current real-time events, live internet data, or up-to-date programming framework documentation that a reference file wouldn't contain.
    
    Message: "{user_message}"
    Reply strictly in JSON format: {{"needs_search": true/false, "query": "optimized web search query if true"}}
    """
    try:
        reply = generate_llm_response([{"role": "user", "content": prompt}], json_mode=True, temperature=0.0)
        return json.loads(reply)
    except:
        return {"needs_search": False, "query": ""}

@observe(name="tavily-web-search")
def web_search(query):
    if not TAVILY_API_KEY: 
        return "[Search disabled: Missing Tavily API Key]"
    try:
        res = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3},
            timeout=10
        ).json()
        return "\n".join([f"- {r['content']}" for r in res.get("results", [])])
    except Exception as e:
        return f"[Search failed: {e}]"

# 6. Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am Aisa. Send a message, upload a PDF, or generate a quiz to begin studying."}
    ]

# 7. Sidebar Layout
with st.sidebar:
    st.subheader("😼 Aisa does not handle [enrollment](https://cit.edu/enrollment/) or [payments](https://cit.edu/payment-options/)!")
    st.markdown("---")
    
    st.subheader("📄 Upload Study Material")
    uploaded_pdf = st.file_uploader("Upload a PDF for Aisa to read", type=["pdf"])
    
    if uploaded_pdf:
        if "pdf_name" not in st.session_state or st.session_state.pdf_name != uploaded_pdf.name:
            with st.spinner("Syncing PDF to cloud database..."):
                success = process_pdf(uploaded_pdf)
                if success:
                    st.session_state.pdf_name = uploaded_pdf.name
                    st.success("PDF indexed to cloud and ready!")
                else:
                    st.error("Failed to read PDF.")

    st.markdown("---")
    st.subheader("🎯 Study Modes")
    study_topic = st.text_input("What topic are we focusing on?", placeholder="e.g., OSI Model, Subnetting")
    
    quiz_btn = st.button("Generate Quiz", use_container_width=True)
    flashcard_btn = st.button("Generate Flashcards", use_container_width=True)
    
    if quiz_btn:
        if study_topic:
            user_msg = f"Let's start a quiz on {study_topic}. Ask me the first question to test my knowledge. Wait for my answer before asking the next one."
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            with st.spinner("Starting quiz..."):
                try:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                    reply = generate_llm_response(messages)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"System Error: {e}")
        else:
            st.warning("Please enter a topic first!")
            
    if flashcard_btn:
        if study_topic:
            user_msg = f"Can you give me 5 study flashcards for {study_topic}?"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            with st.spinner("Generating flashcards..."):
                try:
                    flashcard_prompt = f"""
                    You are a helpful tutor. Provide 5 study flashcards about {study_topic}. 
                    Format EACH flashcard strictly using this exact HTML structure:
                    <details style="background-color: #2D2D2D; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444;">
                      <summary style="font-weight: bold; cursor: pointer; font-size: 16px;">💡 Question: [Your Question]</summary>
                      <p style="margin-top: 15px; color: #E0E0E0; font-size: 15px;">[Your Answer]</p>
                    </details>
                    Do not include markdown code blocks. Just output the raw HTML.
                    """
                    
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": flashcard_prompt}]
                    reply = generate_llm_response(messages)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"System Error: {e}")
        else:
            st.warning("Please enter a topic first!")
            
    st.markdown("---")
    st.subheader("🔗 Quick Links")
    st.markdown("- [CIT-U Homepage](https://cit.edu/)")
    st.markdown("- [Academic Calendar 25-26](https://cit.edu/collegiate-calendar-for-academic-year-2025-2026/)")
    st.markdown("- [Vision-Mission](https://cit.edu/cit-vision-mission-primer/)")
    st.markdown("- [College Programs](https://cit.edu/cit-university-programs/)")
    st.markdown("- [LAIR](https://lair.education/)")
    st.markdown("- [WITS](https://student.cituwits.com/)")
    
    st.markdown("---")

    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        chat_history = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="📥 Download chat as .txt",
            data=chat_history,
            file_name="aisa_study_notes.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I am Aisa. Send a message, upload a PDF, or generate a quiz to begin studying."}
        ]
        if "pdf_name" in st.session_state:
            del st.session_state.pdf_name
        st.rerun()
        
    st.markdown("---")
    st.caption("May 2026 ©")

# 8. Top Header Layout
st.markdown(
    """
    <div style='text-align: left; margin-bottom: 20px;'>
        <h1 style='color: #F2A900; margin-bottom: 0px;'>😼 Aisa AI</h1>
        <p style='color: #888; font-size: 18px;'>Your Applied AI Studies Assistant</p>
    </div>
    <hr>
    """, 
    unsafe_allow_html=True
)

# 9. Render Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; align-items: flex-end; margin-bottom: 10px;'>
                <div style='background-color: #0078D7; color: white; padding: 10px 15px; border-radius: 15px 15px 0px 15px; max-width: 75%;'>
                    {message["content"]}
                </div>
                <div style='font-size: 24px; margin-left: 10px;'>🧑‍💻</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; align-items: flex-end; margin-bottom: 10px;'>
                <div style='font-size: 24px; margin-right: 10px;'>😼</div>
                <div style='background-color: #2D2D2D; color: white; padding: 10px 15px; border-radius: 15px 15px 15px 0px; max-width: 75%;'>
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# 10. Chat Input & Main Processing Logic
if prompt := st.chat_input("How can I help with your studies today?"):
    
    user_text = prompt
    st.session_state.messages.append({"role": "user", "content": user_text})
    
    needs_admin_redirect = False
    for keyword in ADMIN_KEYWORDS:
        if re.search(keyword, user_text.lower()):
            needs_admin_redirect = True
            break
    
    if needs_admin_redirect:
        admin_msg = "It looks like you're asking about enrollment or payments! For the most accurate information, please visit the official CIT-U pages:\n\n* [Enrollment Guide](https://cit.edu/enrollment/)\n* [Payment Options](https://cit.edu/payment-options/)"
        st.session_state.messages.append({"role": "assistant", "content": admin_msg})
        st.rerun()
    else:
        with st.spinner("Aisa is thinking..."):
            try:
                context_blocks = []
                
                # Document Retrieval from Supabase
                rag_text = retrieve_context(user_text)
                if rag_text:
                    context_blocks.append(f"[KNOWLEDGE BASE CONTEXT]:\n{rag_text}")
                
                # Web Search Agent Routing
                decision = agent_decide_search(user_text)
                if decision.get("needs_search"):
                    search_data = web_search(decision.get("query"))
                    context_blocks.append(f"[WEB SEARCH CONTEXT]:\n{search_data}")
                
                # Compile Augmented Context
                augmented_system = SYSTEM_PROMPT
                if context_blocks:
                    augmented_system += "\n\nUse the following context to answer the user:\n" + "\n\n".join(context_blocks)
                
                # Assemble Payload
                messages = [{"role": "system", "content": augmented_system}] + st.session_state.messages
                
                # Single abstracted call for tracing
                reply = generate_llm_response(messages)
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
            except Exception as e:
                if st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                st.error(f"System Error: {e}")

# 11. CRITICAL: Force Langfuse to flush before Streamlit kills the script thread
langfuse_context.flush()
