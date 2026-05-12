import streamlit as st
from groq import Groq
from langfuse import Langfuse
import re
import io
import os
import json
import requests

# PDF & RAG dependencies
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["LANGFUSE_SECRET_KEY"] = st.secrets["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_HOST"] = st.secrets["LANGFUSE_BASE_URL"]

langfuse_client = Langfuse()

# ─────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(page_title="Aisa - AI Studies Assistant", page_icon="😼", layout="wide")

# ─────────────────────────────────────────────
# 2. Prompts & Keywords
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Aisa (Applied Intelligence Studies Assistant), a dedicated AI companion for students at Cebu Institute of Technology - University.
Your tone is human-like, supportive, and slightly casual—like a smart upperclassman.
You are an expert in app development, coding, and general IT courses and subjects.

Key guidelines:
1. Be concise but insightful.
2. Use relatable student language, but stay professional enough.
3. If asked about CIT-U specifically, show school spirit (Technologian pride!).
4. Always prioritize clarity in technical explanations.
5. When context from uploaded PDFs is provided, use it to give accurate, document-grounded answers.
6. When web search results are provided, cite and synthesize them naturally.
"""

ADMIN_KEYWORDS = [r"\benroll", r"\benrollment", r"\btuition", r"\bpay", r"\bpayment", r"\bfee", r"\bfees", r"\bcost"]

# ─────────────────────────────────────────────
# 3. API Configuration
# ─────────────────────────────────────────────
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Aisa is offline 😴 — GROQ_API_KEY missing.")
    st.stop()

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", None)

# ─────────────────────────────────────────────
# 4. RAG Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_vector_store(chunks: list[str], embedder) -> np.ndarray:
    return embedder.encode(chunks, show_progress_bar=False)

def retrieve_top_chunks(query: str, chunks: list[str], chunk_embeddings: np.ndarray, embedder, top_k: int = 4) -> list[str]:
    query_vec = embedder.encode([query])
    sims = cosine_similarity(query_vec, chunk_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# ─────────────────────────────────────────────
# 5. Agentic Web Search
# ─────────────────────────────────────────────
def web_search(query: str) -> str:
    """Search the web using Tavily API (fallback: DuckDuckGo HTML scrape)."""
    if TAVILY_API_KEY:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 4},
                timeout=10
            )
            data = resp.json()
            results = data.get("results", [])
            if results:
                snippets = [f"• {r['title']}: {r['content'][:300]}" for r in results]
                return "\n".join(snippets)
        except Exception as e:
            return f"[Tavily search error: {e}]"
    else:
        # Fallback: DuckDuckGo Instant Answer API (no key needed)
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
                timeout=8
            )
            data = resp.json()
            abstract = data.get("AbstractText", "")
            related = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:4] if "Text" in r]
            parts = []
            if abstract:
                parts.append(f"• {abstract}")
            parts.extend([f"• {t}" for t in related if t])
            return "\n".join(parts) if parts else "[No results found]"
        except Exception as e:
            return f"[Search error: {e}]"

# Agent decision: should we search the web?
AGENT_DECISION_PROMPT = """You are a strict routing agent. Decide if a web search is needed to answer the user's message.
Answer ONLY with valid JSON: {"needs_search": true, "search_query": "..."} or {"needs_search": false}

DO NOT SEARCH IF:
- The message is a greeting, small talk, or conversational filler (e.g., "hi", "hello", "how are you", "thanks", "ok").
- The user is asking a personal question about you.
- The user is explicitly asking to generate a quiz, test, or flashcards.

YOU MUST SEARCH IF (Return {"needs_search": true}):
- The user asks any factual, educational, or informational question (e.g., "What is networking?", "How does an API work?", "History of the internet").
- The user asks for an explanation of a concept, term, or process.
- The user asks about current events, tech, or real-time data.
- If the message is a question and is NOT small talk or a quiz request, default to searching.

User message: {user_message}
"""

def agent_decide_search(user_message: str) -> dict:
    # Short-circuit basic greetings to prevent unnecessary API calls and over-triggering
    clean_msg = user_message.lower().strip()
    greetings = ["hi", "hello", "hey", "how are you", "whats up", "what's up", "sup", "good morning", "good evening"]
    if clean_msg in greetings:
        return {"needs_search": False}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": AGENT_DECISION_PROMPT.format(user_message=user_message)}],
            max_tokens=120,
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"needs_search": False}

# ─────────────────────────────────────────────
# 6. Session State Init
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am Aisa. Send a message, upload a PDF, or generate a quiz to begin studying."}
    ]
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

embedder = load_embedder()

# ─────────────────────────────────────────────
# 7. Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("😼 Aisa does not handle [enrollment](https://cit.edu/enrollment/) or [payments](https://cit.edu/payment-options/)!")
    st.markdown("---")

    # ── RAG: PDF Upload ──
    st.subheader("📄 Upload Study Material (RAG)")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_pdf:
        if uploaded_pdf.name != st.session_state.pdf_name:
            with st.spinner("Reading and indexing your PDF..."):
                raw_text = extract_text_from_pdf(uploaded_pdf)
                chunks = chunk_text(raw_text)
                embeddings = build_vector_store(chunks, embedder)
                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_embeddings = embeddings
                st.session_state.pdf_name = uploaded_pdf.name
            st.success(f"✅ **{uploaded_pdf.name}** indexed! ({len(chunks)} chunks)")
    
    if st.session_state.pdf_name:
        st.caption(f"📎 Active PDF: **{st.session_state.pdf_name}**")
        if st.button("🗑️ Remove PDF", use_container_width=True):
            st.session_state.pdf_chunks = []
            st.session_state.pdf_embeddings = None
            st.session_state.pdf_name = None
            st.rerun()

    st.markdown("---")

    # ── Study Modes ──
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
                    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
                    reply = response.choices[0].message.content
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
                    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
                    reply = response.choices[0].message.content
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
        st.rerun()

    st.markdown("---")
    st.caption("May 2026 ©")

# ─────────────────────────────────────────────
# 8. Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align: left; margin-bottom: 20px;'>
        <h1 style='color: #F2A900; margin-bottom: 0px;'>😼 Aisa AI</h1>
        <p style='color: #888; font-size: 18px;'>Your Applied AI Studies Assistant
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# 9. Render Chat History
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 10. Chat Input & Agent Pipeline
# ─────────────────────────────────────────────
if prompt := st.chat_input("How can I help with your studies today?"):

    user_text = prompt
    st.session_state.messages.append({"role": "user", "content": user_text})

    st.markdown(
        f"""
        <div style='display: flex; justify-content: flex-end; align-items: flex-end; margin-bottom: 10px;'>
            <div style='background-color: #0078D7; color: white; padding: 10px 15px; border-radius: 15px 15px 0px 15px; max-width: 75%;'>
                {user_text}
            </div>
            <div style='font-size: 24px; margin-left: 10px;'>🧑‍💻</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Admin keyword redirect ──
    needs_admin_redirect = any(re.search(kw, user_text.lower()) for kw in ADMIN_KEYWORDS)

    if needs_admin_redirect:
        admin_msg = (
            "It looks like you're asking about enrollment or payments! "
            "For the most accurate information, please visit the official CIT-U pages:\n\n"
            "* [Enrollment Guide](https://cit.edu/enrollment/)\n"
            "* [Payment Options](https://cit.edu/payment-options/)"
        )
        st.session_state.messages.append({"role": "assistant", "content": admin_msg})
        st.rerun()

    else:
        context_blocks = []
        status_labels = []

        # ── STEP 1: RAG — retrieve from uploaded PDF ──
        if st.session_state.pdf_chunks and st.session_state.pdf_embeddings is not None:
            with st.status("📄 Searching your PDF...", expanded=False) as rag_status:
                top_chunks = retrieve_top_chunks(
                    user_text,
                    st.session_state.pdf_chunks,
                    st.session_state.pdf_embeddings,
                    embedder
                )
                rag_context = "\n\n---\n\n".join(top_chunks)
                context_blocks.append(
                    f"[DOCUMENT CONTEXT from '{st.session_state.pdf_name}']:\n{rag_context}"
                )
                rag_status.update(label=f"📄 Retrieved from **{st.session_state.pdf_name}**", state="complete")

        # ── STEP 2: Agentic — decide & run web search ──
        with st.status("🤖 Agent thinking...", expanded=False) as agent_status:
            decision = agent_decide_search(user_text)
            if decision.get("needs_search"):
                search_query = decision.get("search_query", user_text)
                agent_status.update(label=f"🌐 Searching: *{search_query}*", state="running")
                search_results = web_search(search_query)
                context_blocks.append(
                    f"[WEB SEARCH RESULTS for '{search_query}']:\n{search_results}"
                )
                agent_status.update(label=f"🌐 Web search done: *{search_query}*", state="complete")
            else:
                agent_status.update(label="🤖 No web search needed", state="complete")

        # ── STEP 3: Build augmented system prompt ──
        augmented_system = SYSTEM_PROMPT
        if context_blocks:
            augmented_system += "\n\n" + "\n\n".join(context_blocks)

        # ── STEP 4: Call LLM ──
        with st.spinner("😼 Aisa is thinking..."):
            try:
                messages = [{"role": "system", "content": augmented_system}] + st.session_state.messages
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                reply = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"System Error: {e}")
