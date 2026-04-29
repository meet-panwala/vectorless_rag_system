"""
app_vectorless.py  —  Final Streamlit UI
-----------------------------------------
Changes:
  ✅ compound-beta + compound-beta-mini restored with TPM warning label
  ✅ 429 shows orange warning + "Switch model" tip, NOT a crash
  ✅ 413 shows orange warning + "Clear Chat" tip
  ✅ Model descriptions shown in sidebar
  ✅ Auto-builds catalog index on first run
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

# ── Model list with labels ────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "llama-3.3-70b-versatile":  "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile":  "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Fashion AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.main-header {
    background: linear-gradient(135deg, #2193b0, #6dd5ed);
    color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px;
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
    st.markdown("## 🛍️ Flipkart Fashion AI")
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

    # Show TPM warning for compound models
    if "compound" in selected_model:
        st.warning(
            "⚠️ **compound models have 30,000 TPM** (tokens/min).\n\n"
            "This is very low — complex queries may hit the 429 rate limit. "
            "The system will auto-retry, but you may need to wait ~10s. "
            "Switch to **llama-3.3-70b-versatile** for the best experience."
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
    <h2 style="margin:0">🛍️ Flipkart Fashion Shopping Assistant</h2>
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


# ── Render trace helper ───────────────────────────────────────────────────────
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
                        preview["products_count"]  = len(r["products"])
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