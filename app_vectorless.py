"""
app_vectorless.py  —  Final Streamlit UI
-----------------------------------------
Dark mode fixes:
  ✅ Buttons readable in dark mode (light text on dark bg)
  ✅ Chat message bubbles visible in dark mode
  ✅ Sidebar text visible in dark mode
  ✅ Covers both browser dark mode (prefers-color-scheme)
     and Streamlit dark theme toggle ([data-theme="dark"])
  ✅ Light mode completely unchanged
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

JSON_PATH = str(BASE_DIR / "data" / "flipkart_fashion_products_dataset.json")
TREE_PATH = str(BASE_DIR / "catalog_index" / "catalog_tree.json")

MODEL_OPTIONS = {
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant":    "llama-3.1-8b-instant",
}

EXAMPLE_QUERIES = [
    "Show me the best rated kurtas under ₹500",
    "Find sarees from Libas brand",
    "Compare budget vs premium running shoes",
    "Most discounted women's dresses",
    "Top 5 formal shirts for men with rating above 4",
    "Ethnic wear under ₹1000 with good reviews",
    "Which brands have the best handbags?",
    "Best sneakers under ₹2000",
    "What categories do you have?",
]

st.set_page_config(
    page_title="Shopping AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: light + dark mode ────────────────────────────────────────────────────
st.markdown("""
<style>

/* ══════════════════════════════════════════════
   LIGHT MODE (default — unchanged)
══════════════════════════════════════════════ */

.stApp {
    background-color: #f8f9fa;
}

.main-header {
    background: linear-gradient(135deg, #2193b0, #6dd5ed);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 18px;
}

/* Sidebar example query buttons */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff;
    color: #1a1a2e;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    text-align: left;
    font-size: 13px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e8f4f8;
    border-color: #2193b0;
    color: #2193b0;
}

/* Main area buttons */
.main .stButton > button,
.block-container .stButton > button {
    background-color: #ffffff;
    color: #1a1a2e;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    font-size: 14px;
}
.main .stButton > button:hover,
.block-container .stButton > button:hover {
    background-color: #e8f4f8;
    border-color: #2193b0;
    color: #2193b0;
}


/* ══════════════════════════════════════════════
   DARK MODE — browser level (prefers-color-scheme)
══════════════════════════════════════════════ */

@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #0e1117 !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1e2330 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3f50 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2a3550 !important;
        border-color: #6dd5ed !important;
        color: #6dd5ed !important;
    }

    /* Main area buttons */
    .main .stButton > button,
    .block-container .stButton > button {
        background-color: #1e2330 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3f50 !important;
    }
    .main .stButton > button:hover,
    .block-container .stButton > button:hover {
        background-color: #2a3550 !important;
        border-color: #6dd5ed !important;
        color: #6dd5ed !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1a1f2e !important;
        border-radius: 10px !important;
    }

    /* Caption / subtext */
    .stChatMessage .stCaption,
    [data-testid="stChatMessage"] .stCaption {
        color: #9aa0b0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1f2e !important;
        color: #e0e0e0 !important;
    }
    .streamlit-expanderContent {
        background-color: #12151e !important;
    }

    /* General text */
    .stMarkdown p, .stMarkdown li {
        color: #e0e0e0;
    }

    hr {
        border-color: #3a3f50 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #c0c5d0 !important;
    }
}


/* ══════════════════════════════════════════════
   DARK MODE — Streamlit theme toggle
   (when user picks dark in Streamlit's ☰ menu)
══════════════════════════════════════════════ */

[data-theme="dark"] .stApp {
    background-color: #0e1117 !important;
}

[data-theme="dark"] section[data-testid="stSidebar"] .stButton > button {
    background-color: #1e2330 !important;
    color: #e0e0e0 !important;
    border: 1px solid #3a3f50 !important;
}
[data-theme="dark"] section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2a3550 !important;
    border-color: #6dd5ed !important;
    color: #6dd5ed !important;
}

[data-theme="dark"] .main .stButton > button,
[data-theme="dark"] .block-container .stButton > button {
    background-color: #1e2330 !important;
    color: #e0e0e0 !important;
    border: 1px solid #3a3f50 !important;
}
[data-theme="dark"] .main .stButton > button:hover,
[data-theme="dark"] .block-container .stButton > button:hover {
    background-color: #2a3550 !important;
    border-color: #6dd5ed !important;
    color: #6dd5ed !important;
}

[data-theme="dark"] [data-testid="stChatMessage"] {
    background-color: #1a1f2e !important;
    border-radius: 10px !important;
}

[data-theme="dark"] .stChatMessage .stCaption,
[data-theme="dark"] [data-testid="stChatMessage"] .stCaption {
    color: #9aa0b0 !important;
}

[data-theme="dark"] .streamlit-expanderHeader {
    background-color: #1a1f2e !important;
    color: #e0e0e0 !important;
}

[data-theme="dark"] .streamlit-expanderContent {
    background-color: #12151e !important;
}

[data-theme="dark"] hr {
    border-color: #3a3f50 !important;
}

[data-theme="dark"] section[data-testid="stSidebar"] p,
[data-theme="dark"] section[data-testid="stSidebar"] span,
[data-theme="dark"] section[data-testid="stSidebar"] label {
    color: #c0c5d0 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":   [],
    "traces":     {},
    "agent":      None,
    "cur_model":  None,
    "init_error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Agent loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_agent_cached(model: str):
    from vectorless_rag import get_agent
    return get_agent(JSON_PATH, TREE_PATH, model=model)


def ensure_agent(model: str):
    if st.session_state.agent is not None and st.session_state.cur_model == model:
        return st.session_state.agent

    if not os.path.exists(JSON_PATH):
        st.session_state.init_error = (
            f"❌ Dataset not found: `{JSON_PATH}`\n\n"
            "Put `flipkart_fashion_products_dataset.json` inside the `data/` folder."
        )
        return None

    if not os.path.exists(TREE_PATH):
        with st.spinner("🔨 Building catalog index (first run, ~5s)…"):
            try:
                from vectorless_rag.index_builder import build_catalog_tree
                build_catalog_tree(JSON_PATH, TREE_PATH)
            except Exception as e:
                st.session_state.init_error = f"❌ Index build failed: {e}"
                return None

    if not os.environ.get("GROQ_API_KEY"):
        st.session_state.init_error = (
            "❌ `GROQ_API_KEY` not set.\n\n"
            "Add it to `.env` or set as an environment variable."
        )
        return None

    try:
        with st.spinner("🚀 Loading products into memory…"):
            agent = _load_agent_cached(model)
        st.session_state.agent      = agent
        st.session_state.cur_model  = model
        st.session_state.init_error = None
        return agent
    except Exception as e:
        st.session_state.init_error = f"❌ Failed to load agent: {e}"
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ Shopping AI Assistant")
    st.markdown("*Vectorless RAG · PageIndex style*")
    st.divider()

    st.markdown("### ⚙️ Settings")
    model_keys   = list(MODEL_OPTIONS.keys())
    model_labels = list(MODEL_OPTIONS.values())
    sel_idx = st.selectbox(
        "Model",
        range(len(model_keys)),
        format_func=lambda i: model_labels[i],
        index=0,
    )
    selected_model = model_keys[sel_idx]

    if "compound" in selected_model:
        st.warning(
            "⚠️ **compound models have 30,000 TPM** (tokens/min).\n\n"
            "Complex queries may hit the 429 rate limit. "
            "Switch to **llama-3.3-70b-versatile** for best experience."
        )

    show_trace = st.toggle("🔍 Show reasoning trace", value=False)

    st.divider()
    st.markdown("### 💡 Try these")
    for q in EXAMPLE_QUERIES:
        if st.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
            st.session_state["_pending"] = q

    st.divider()
    st.markdown("### 📊 Status")
    st.markdown(f"{'✅' if os.path.exists(JSON_PATH) else '❌'} Dataset JSON")
    st.markdown(f"{'✅' if os.path.exists(TREE_PATH) else '⚠️'} Catalog index")
    st.markdown(f"{'✅' if os.environ.get('GROQ_API_KEY') else '❌'} GROQ_API_KEY")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.traces   = {}
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🛍️ Shopping AI Assistant</h2>
    <p style="margin:4px 0 0;opacity:.9">Vectorless RAG · 30,000+ Products · Groq LLM</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.init_error:
    st.error(st.session_state.init_error)
    st.stop()

agent = ensure_agent(selected_model)
if agent is None:
    if st.session_state.init_error:
        st.error(st.session_state.init_error)
    st.stop()


# ── Render trace ──────────────────────────────────────────────────────────────
def render_trace(trace: dict, expanded: bool = False):
    if not trace.get("trace"):
        return
    with st.expander(f"🔍 Reasoning — {trace['steps']} tool call(s)", expanded=expanded):
        for step in trace["trace"]:
            st.markdown(f"**Step {step['step']}: `{step['tool']}`**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Arguments")
                st.json(step["args"])
            with c2:
                st.caption("Result preview")
                r = step["result"]
                if isinstance(r, dict):
                    preview = {k: v for k, v in r.items() if k != "products"}
                    if "products" in r:
                        preview["products_count"] = len(r["products"])
                        if r["products"]:
                            preview["first_product"] = r["products"][0].get("name", "")
                    st.json(preview)
                else:
                    st.code(str(r)[:400])


# ── Render history ────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_trace and i in st.session_state.traces:
            render_trace(st.session_state.traces[i])


# ── Chat input ────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("_pending", None)
user_input = st.chat_input("Ask about fashion products… e.g. 'What categories do you have?'")
if pending and not user_input:
    user_input = pending


def _is_error(text: str) -> bool:
    return text.startswith("⚠️") or text.startswith("⏳")


if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    history = st.session_state.messages[:-1]

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Searching the catalog…"):
            t0 = time.time()
            try:
                trace   = agent.get_tool_trace(user_input, history=history)
                answer  = trace["final_answer"]
                elapsed = time.time() - t0
            except Exception as e:
                answer  = f"⚠️ Unexpected error: {e}"
                trace   = {"final_answer": answer, "steps": 0, "trace": []}
                elapsed = 0

        if _is_error(answer):
            st.warning(answer)
            if "429" in answer or "Rate limit" in answer or "⏳" in answer:
                st.info(
                    "💡 **Tip:** Switch to **llama-3.3-70b-versatile** in the sidebar "
                    "(it has ~200× more TPM than compound models)."
                )
            elif "413" in answer or "too large" in answer:
                st.info("💡 **Tip:** Press **🗑️ Clear Chat** in the sidebar, then ask again.")
        else:
            st.markdown(answer)
            st.caption(f"⏱️ {elapsed:.1f}s · {trace['steps']} tool call(s) · {selected_model}")
            if show_trace:
                render_trace(trace, expanded=True)

    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    if trace.get("trace"):
        st.session_state.traces[idx] = trace


# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 👋 Welcome! What are you looking for today?")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES[:6]):
        with cols[i % 2]:
            if st.button(f"💬 {q}", use_container_width=True, key=f"w_{i}"):
                st.session_state["_pending"] = q
                st.rerun()