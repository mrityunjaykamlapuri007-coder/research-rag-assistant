import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Research Assistant · CBN Ceramics",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Instrument+Sans:ital,wght@0,400;0,500;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp { 
    font-family: 'Instrument Sans', sans-serif;
    background: #fafaf8 !important;
    color: #1a1a18;
}

#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1a18 !important;
    border-right: none !important;
    width: 300px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem !important;
}
section[data-testid="stSidebar"] * { color: #f0efe8 !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f0efe8;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-tagline {
    font-size: 11px;
    color: #666 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.sidebar-section {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555 !important;
    margin: 1.5rem 0 0.6rem;
}
.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12.5px;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #2d2d2a;
    margin-bottom: 1rem;
}
.dot-green { background: #22c55e; width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-red { background: #ef4444; width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

.model-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #2d2d2a;
    font-size: 12px;
}
.model-icon { color: #888 !important; font-size: 13px; flex-shrink: 0; margin-top: 1px; }
.model-name { font-weight: 500; font-size: 12px; }
.model-sub { color: #666 !important; font-size: 10.5px; margin-top: 1px; }

.score-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 0.5rem;
}
.score-item {
    background: #242420;
    border: 1px solid #2d2d2a;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.score-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #d4a843 !important;
}
.score-label {
    font-size: 9.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666 !important;
    margin-top: 2px;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #2d2d2a !important;
    color: #f0efe8 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #d4a843 !important;
    background: #242420 !important;
}

section[data-testid="stSidebar"] .stToggle {
    margin-top: 0.5rem;
}

/* ── Main Content ── */
.main-wrap {
    max-width: 780px;
    margin: 0 auto;
    padding: 2.5rem 1rem 6rem;
}

.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1.5px solid #e8e8e2;
}
.page-eyebrow {
    font-size: 10.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #1a1a18;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.page-title em {
    font-style: italic;
    font-family: 'Instrument Sans', serif;
    font-weight: 400;
    color: #888;
}
.page-desc {
    font-size: 14.5px;
    color: #666;
    line-height: 1.6;
    max-width: 560px;
}
.tag-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.tag {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #e0e0d8;
    color: #666;
    background: #fff;
}

/* ── Example Questions ── */
.examples-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.75rem;
}
.examples-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 2rem;
}

/* Main area streamlit buttons */
.main-wrap .stButton > button {
    background: #fff !important;
    border: 1px solid #e0e0d8 !important;
    color: #444 !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    text-align: left !important;
    line-height: 1.4 !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.main-wrap .stButton > button:hover {
    border-color: #d4a843 !important;
    color: #1a1a18 !important;
    box-shadow: 0 2px 8px rgba(212,168,67,0.12) !important;
}

/* ── Chat Messages ── */
.chat-thread { margin-bottom: 1.5rem; }

.msg-wrap-user {
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}
.msg-bubble-user {
    background: #1a1a18;
    color: #f0efe8;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.55;
}

.msg-wrap-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 1rem 0;
    gap: 12px;
    align-items: flex-start;
}
.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #f0efe8;
    border: 1.5px solid #e8e8e2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}
.msg-bubble-assistant {
    background: #fff;
    border: 1px solid #e8e8e2;
    color: #1a1a18;
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Chunk Cards ── */
.chunk-wrap {
    margin-top: 8px;
    padding-left: 46px;
}
.chunk-card {
    background: #fafaf8;
    border: 1px solid #e8e8e2;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 12px;
}
.chunk-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.chunk-src { color: #d4a843; font-size: 11px; font-weight: 500; }
.chunk-badge {
    background: #fef9ec;
    border: 1px solid #f5e0a0;
    color: #b8860b;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
}
.chunk-body { color: #666; line-height: 1.5; }

/* ── Input Bar ── */
.input-bar-wrap {
    position: fixed;
    bottom: 0;
    left: 300px;
    right: 0;
    background: linear-gradient(to top, #fafaf8 70%, transparent);
    padding: 1rem 2rem 1.5rem;
    z-index: 100;
}
.input-inner {
    max-width: 780px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
    align-items: center;
    background: #fff;
    border: 1.5px solid #e0e0d8;
    border-radius: 14px;
    padding: 8px 8px 8px 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Override streamlit input inside bar */
.input-bar-wrap .stTextInput > div > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.input-bar-wrap .stTextInput input {
    border: none !important;
    background: transparent !important;
    color: #1a1a18 !important;
    font-size: 14px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    padding: 4px 0 !important;
    box-shadow: none !important;
}
.input-bar-wrap .stButton > button {
    background: #1a1a18 !important;
    color: #f0efe8 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    width: auto !important;
    box-shadow: none !important;
}
.input-bar-wrap .stButton > button:hover {
    background: #333 !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #e8e8e2 !important;
    border-radius: 8px !important;
    background: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">⚗️ ResearchRAG</div>
    <div class="sidebar-tagline">Scientific QA System</div>
    """, unsafe_allow_html=True)

    try:
        requests.get(f"{API_URL}/health", timeout=2)
        st.markdown('<div class="status-dot"><div class="dot-green"></div> API connected</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="status-dot"><div class="dot-red"></div> API offline</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="model-row">
            <div class="model-icon">🧠</div>
            <div><div class="model-name">Llama 3.1 8B Instruct</div>
            <div class="model-sub">4-bit NF4 quantization</div></div>
        </div>
        <div class="model-row">
            <div class="model-icon">🔍</div>
            <div><div class="model-name">BGE-Large-en-v1.5</div>
            <div class="model-sub">Dense embeddings</div></div>
        </div>
        <div class="model-row">
            <div class="model-icon">⚡</div>
            <div><div class="model-name">BGE-Reranker-Large</div>
            <div class="model-sub">Cross-encoder reranking</div></div>
        </div>
        <div class="model-row" style="border-bottom:none">
            <div class="model-icon">📄</div>
            <div><div class="model-name">21 Research Papers</div>
            <div class="model-sub">1068 indexed chunks</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="score-grid">
        <div class="score-item"><div class="score-val">0.284</div><div class="score-label">ROUGE-1</div></div>
        <div class="score-item"><div class="score-val">0.611</div><div class="score-label">BERTScore</div></div>
        <div class="score-item"><div class="score-val">0.137</div><div class="score-label">ROUGE-2</div></div>
        <div class="score-item"><div class="score-val">0.840</div><div class="score-label">Sem Sim</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Options</div>', unsafe_allow_html=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">Retrieval-Augmented Generation</div>
    <div class="page-title">Ask the papers.<br><em>Get precise answers.</em></div>
    <div class="page-desc">
        Query 21 research papers on CaBi₂Nb₂O₉ piezoelectric ceramics using
        hybrid BM25 + FAISS retrieval, cross-encoder reranking, and Llama 3.1 8B.
    </div>
    <div class="tag-row">
        <span class="tag">Hybrid Search</span>
        <span class="tag">RRF Fusion</span>
        <span class="tag">Context Compression</span>
        <span class="tag">Query Expansion</span>
        <span class="tag">Multi-turn Chat</span>
    </div>
</div>
""", unsafe_allow_html=True)

EXAMPLES = [
    "What is the Curie temperature of pure CBN ceramics?",
    "What are the advantages of BLSF materials over PZT?",
    "How does Cu/W co-doping improve piezoelectric properties?",
    "What is the role of oxygen vacancies in CBN ceramics?",
    "What is the space group of CaBi₂Nb₂O₉?",
    "How does NaBi modification affect the Curie temperature?",
]

if not st.session_state.messages:
    st.markdown('<div class="examples-label">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}"):
                st.session_state.pending = q
                st.rerun()

st.markdown('<div class="chat-thread">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-wrap-user">
            <div class="msg-bubble-user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-wrap-assistant">
            <div class="msg-avatar">⚗️</div>
            <div class="msg-bubble-assistant">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if show_chunks and msg.get("chunks"):
            with st.expander(f"📄 View {len(msg['chunks'])} retrieved chunks"):
                for i, c in enumerate(msg["chunks"]):
                    score = c.get("rerank_score")
                    score_str = f"{score:.4f}" if score else "—"
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-top">
                            <span class="chunk-src">
                                {c.get('source','?')} · p.{c.get('page','?')} · {c.get('section','?')}
                            </span>
                            <span class="chunk-badge">score {score_str}</span>
                        </div>
                        <div class="chunk-body">{c.get('text','')[:380]}…</div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Fixed Input Bar ───────────────────────────────────────────────────────────
st.markdown('<div class="input-bar-wrap"><div class="input-inner">', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input("", placeholder="Ask a question about the research papers…",
                               label_visibility="collapsed", key="user_input")
with col2:
    send = st.button("Send →", key="send_btn")
st.markdown('</div></div>', unsafe_allow_html=True)


def process(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner(""):
        try:
            res = requests.post(f"{API_URL}/ask", json={
                "question": question,
                "chat_history": st.session_state.chat_history
            }, timeout=120)
            data = res.json()
            answer = data["answer"]
            st.session_state.chat_history = data["chat_history"]
            cr = requests.post(f"{API_URL}/retrieve",
                               json={"question": question}, timeout=30)
            chunks = cr.json().get("chunks", [])
        except requests.exceptions.ConnectionError:
            answer = "⚠️ Could not connect to the API. Run `python api.py` first."
            chunks = []
        except Exception as e:
            answer = f"⚠️ Error: {e}"
            chunks = []
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks})
    st.rerun()


if "pending" in st.session_state:
    q = st.session_state.pop("pending")
    process(q)

if send and user_input.strip():
    process(user_input.strip())