import os
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from rag import load_vectorstore, build_chain, ask, build_chat_history

load_dotenv()

CHROMA_DIR = "./chroma_db"

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Lexi — Your Learning Advisor",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS — warm, branded feel ───────────
st.markdown("""
<style>
  /* Gradient header */
  .hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1.5rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
    color: white;
  }
  .hero h1 { font-size: 2rem; margin: 0; }
  .hero p  { font-size: 1rem; opacity: 0.9; margin: 0.4rem 0 0; }

  /* Lead capture box */
  .lead-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.9rem;
  }

  /* Subtle assistant bubble tint */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #fafafa;
    border-radius: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Hero header ───────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Hi, I'm Lexi!</h1>
  <p>Your personal learning advisor — let's find the perfect course for you.</p>
</div>
""", unsafe_allow_html=True)


# ── Load resources (cached) ───────────────────
@st.cache_resource(show_spinner="Getting Lexi ready...")
def load_resources():
    if not Path(CHROMA_DIR).exists():
        return None, None
    vectorstore = load_vectorstore()
    chain       = build_chain(vectorstore)
    return vectorstore, chain

vectorstore, chain = load_resources()

if vectorstore is None:
    st.error("Course database not found. Please run `python ingest.py` first.")
    st.stop()


# ── Session state ─────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "leads"        not in st.session_state:
    st.session_state.leads        = []   # captured name/email pairs
if "greeted"      not in st.session_state:
    st.session_state.greeted      = False


# ── Opening greeting (shown once) ─────────────
if not st.session_state.greeted:
    opening = (
        "Hey there! 👋 I'm Lexi, your personal learning advisor. "
        "I'm here to help you find the perfect course to reach your goals.\n\n"
        "What would you love to learn or get better at? "
        "Tell me a bit about yourself and I'll point you in the right direction! 😊"
    )
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.greeted = True


# ── Lead detection helper ─────────────────────
def extract_lead(text):
    """Detect if user shared an email in their message."""
    email_pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group(0) if match else None


# ── Display chat history ───────────────────────
for message in st.session_state.messages:
    avatar = "🎓" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ── Sidebar — leads + stats ───────────────────
with st.sidebar:
    st.markdown("### 📊 Session Stats")
    total_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions asked", total_msgs)
    st.metric("Leads captured", len(st.session_state.leads))

    if st.session_state.leads:
        st.markdown("### 📧 Captured Leads")
        for lead in st.session_state.leads:
            st.markdown(f"- `{lead['email']}` — {lead['time']}")

    st.divider()
    if st.button("🔄 Start new conversation"):
        st.session_state.messages = []
        st.session_state.greeted  = False
        st.rerun()


# ── Chat input ────────────────────────────────
if query := st.chat_input("Ask me anything about our courses..."):

    # Check for email in user message → capture lead
    detected_email = extract_lead(query)
    if detected_email:
        st.session_state.leads.append({
            "email": detected_email,
            "time":  datetime.now().strftime("%H:%M"),
        })

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Build conversation history for memory
    # Exclude the message we just added (it's the current query)
    history_msgs = st.session_state.messages[:-1]
    chat_history = build_chat_history(history_msgs)

    # Generate response
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Lexi is thinking..."):
            try:
                answer, _ = ask(chain, query, vectorstore, chat_history)
            except Exception as e:
                answer = (
                    "Oops, I hit a small hiccup there! 😅 "
                    f"Could you try asking that again? (Error: {e})"
                )

        st.markdown(answer)

        # Show lead capture box if email was just shared
        if detected_email:
            st.markdown(f"""
            <div class="lead-box">
              ✅ <strong>Got it!</strong> We'll send course details to 
              <strong>{detected_email}</strong> shortly.
            </div>
            """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })