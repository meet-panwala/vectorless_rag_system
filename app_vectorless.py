import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

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
    "What categories do you have?",
    "Best sneakers under ₹2000",
    "Top 5 formal shirts for men with rating above 4",
    "Compare budget vs premium running shoes",
    "Most discounted women's dresses",
    "Ethnic wear under ₹1000 with good reviews",
    "Which brands have the best handbags?",
    "give me subcategory of Footwear"
]

st.set_page_config(
    page_title="Shopping AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Style
st.markdown("""
<style>

/* ── Light mode ── */
.stApp { background-color: #f8f9fa; }

.main-header {
    background: linear-gradient(135deg, #2193b0, #6dd5ed);
    color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px;
}

.metric-card {
    background: white; border-radius: 12px; padding: 20px 16px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #2193b0;
}
.metric-card h2 { margin: 0; color: #2193b0; font-size: 2rem; }
.metric-card p  { margin: 4px 0 0; color: #666; font-size: 13px; }

section[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff; color: #1a1a2e;
    border: 1px solid #dee2e6; border-radius: 8px;
    text-align: left; font-size: 13px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e8f4f8; border-color: #2193b0; color: #2193b0;
}
.block-container .stButton > button {
    background-color: #ffffff; color: #1a1a2e;
    border: 1px solid #dee2e6; border-radius: 10px; font-size: 14px;
}
.block-container .stButton > button:hover {
    background-color: #e8f4f8; border-color: #2193b0; color: #2193b0;
}

/* ── Dark mode (browser) ── */
@media (prefers-color-scheme: dark) {
    .stApp { background-color: #0e1117 !important; }
    .metric-card { background: #1a1f2e !important; border-left-color: #6dd5ed; }
    .metric-card h2 { color: #6dd5ed !important; }
    .metric-card p  { color: #9aa0b0 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1e2330 !important; color: #e0e0e0 !important;
        border: 1px solid #3a3f50 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2a3550 !important; border-color: #6dd5ed !important;
        color: #6dd5ed !important;
    }
    .block-container .stButton > button {
        background-color: #1e2330 !important; color: #e0e0e0 !important;
        border: 1px solid #3a3f50 !important;
    }
    .block-container .stButton > button:hover {
        background-color: #2a3550 !important; border-color: #6dd5ed !important;
        color: #6dd5ed !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1a1f2e !important; border-radius: 10px !important;
    }
    [data-testid="stChatMessage"] .stCaption { color: #9aa0b0 !important; }
    .streamlit-expanderHeader { background-color: #1a1f2e !important; color: #e0e0e0 !important; }
    .streamlit-expanderContent { background-color: #12151e !important; }
    hr { border-color: #3a3f50 !important; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: #c0c5d0 !important; }
}

/* ── Dark mode (Streamlit toggle) ── */
[data-theme="dark"] .stApp { background-color: #0e1117 !important; }
[data-theme="dark"] .metric-card { background: #1a1f2e !important; border-left-color: #6dd5ed; }
[data-theme="dark"] .metric-card h2 { color: #6dd5ed !important; }
[data-theme="dark"] .metric-card p  { color: #9aa0b0 !important; }
[data-theme="dark"] section[data-testid="stSidebar"] .stButton > button {
    background-color: #1e2330 !important; color: #e0e0e0 !important;
    border: 1px solid #3a3f50 !important;
}
[data-theme="dark"] section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2a3550 !important; border-color: #6dd5ed !important;
    color: #6dd5ed !important;
}
[data-theme="dark"] .block-container .stButton > button {
    background-color: #1e2330 !important; color: #e0e0e0 !important;
    border: 1px solid #3a3f50 !important;
}
[data-theme="dark"] .block-container .stButton > button:hover {
    background-color: #2a3550 !important; border-color: #6dd5ed !important;
    color: #6dd5ed !important;
}
[data-theme="dark"] [data-testid="stChatMessage"] {
    background-color: #1a1f2e !important; border-radius: 10px !important;
}
[data-theme="dark"] [data-testid="stChatMessage"] .stCaption { color: #9aa0b0 !important; }
[data-theme="dark"] .streamlit-expanderHeader { background-color: #1a1f2e !important; color: #e0e0e0 !important; }
[data-theme="dark"] .streamlit-expanderContent { background-color: #12151e !important; }
[data-theme="dark"] hr { border-color: #3a3f50 !important; }
[data-theme="dark"] section[data-testid="stSidebar"] p,
[data-theme="dark"] section[data-testid="stSidebar"] span,
[data-theme="dark"] section[data-testid="stSidebar"] label { color: #c0c5d0 !important; }

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


# Analytics data loader
@st.cache_data(show_spinner=False)
def load_analytics_data(json_path: str) -> dict:
    import re

    def clean_price(v):
        if not v: return 0.0
        try: return float(str(v).replace("₹","").replace(",","").strip())
        except: return 0.0

    def clean_rating(v):
        if not v: return 0.0
        m = re.search(r"([\d.]+)", str(v))
        return float(m.group(1)) if m else 0.0

    def clean_discount(v):
        if not v: return 0.0
        m = re.search(r"([\d.]+)", str(v))
        return float(m.group(1)) if m else 0.0

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    products = raw if isinstance(raw, list) else raw.get("products", [])

    categories    = defaultdict(int)
    brands        = defaultdict(int)
    prices, ratings, discounts = [], [], []
    price_bands   = {"Budget (<₹500)": 0, "Mid (₹500–₹1,500)": 0, "Premium (>₹1,500)": 0}
    rated_prods   = 0
    cat_avg_price  = defaultdict(list)
    cat_avg_rating = defaultdict(list)
    subcat_counts  = defaultdict(lambda: defaultdict(int))

    for p in products:
        cat    = str(p.get("category", "Unknown")).strip()
        subcat = str(p.get("sub_category", "General")).strip()
        brand  = str(p.get("brand", "Unknown")).strip()
        price  = clean_price(p.get("selling_price", p.get("discounted_price", 0)))
        rating = clean_rating(p.get("average_rating", p.get("rating", 0)))
        disc   = clean_discount(p.get("discount", p.get("discount_percentage", 0)))

        categories[cat]            += 1
        brands[brand]              += 1
        subcat_counts[cat][subcat] += 1

        if price > 0:
            prices.append(price)
            cat_avg_price[cat].append(price)
            if price < 500:    price_bands["Budget (<₹500)"]   += 1
            elif price < 1500: price_bands["Mid (₹500–₹1,500)"] += 1
            else:              price_bands["Premium (>₹1,500)"] += 1

        if rating > 0:
            ratings.append(rating)
            cat_avg_rating[cat].append(rating)
            rated_prods += 1

        if disc > 0:
            discounts.append(disc)

    return {
        "total":            len(products),
        "total_categories": len(categories),
        "total_brands":     len(brands),
        "rated_products":   rated_prods,
        "categories":       dict(sorted(categories.items(), key=lambda x: -x[1])),
        "top_brands":       dict(sorted(brands.items(), key=lambda x: -x[1])[:20]),
        "price_bands":      price_bands,
        "prices":           prices,
        "ratings":          ratings,
        "discounts":        discounts,
        "cat_avg_price":    {k: round(sum(v)/len(v), 2) for k, v in cat_avg_price.items()},
        "cat_avg_rating":   {k: round(sum(v)/len(v), 2) for k, v in cat_avg_rating.items()},
        "subcat_counts":    {k: dict(v) for k, v in subcat_counts.items()},
    }



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

#sidebar code
with st.sidebar:
    st.markdown("## 🛍️ Shopping AI Assistant")
    st.markdown("*Vectorless RAG · PageIndex style*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Chatbot", "📊 Analytics Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()

    if page == "💬 Chatbot":
        st.markdown("### ⚙️ Settings")
        model_keys   = list(MODEL_OPTIONS.keys())
        model_labels = list(MODEL_OPTIONS.values())
        sel_idx = st.selectbox("Model", range(len(model_keys)),
                               format_func=lambda i: model_labels[i], index=0)
        selected_model = model_keys[sel_idx]
        show_trace = st.toggle("🔍 Show reasoning trace", value=False)

        st.divider()
        st.markdown("### 💡 Try these")
        for q in EXAMPLE_QUERIES:
            if st.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
                st.session_state["_pending"] = q

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.traces   = {}
            st.rerun()
    else:
        selected_model = list(MODEL_OPTIONS.keys())[0]
        show_trace     = False


# CHATBOT Tab
if page == "💬 Chatbot":

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

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_trace and i in st.session_state.traces:
                render_trace(st.session_state.traces[i])

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
                if "429" in answer or "⏳" in answer:
                    st.info("💡 Switch to **llama-3.3-70b-versatile** in the sidebar.")
                elif "413" in answer:
                    st.info("💡 Press **🗑️ Clear Chat** in the sidebar, then ask again.")
            else:
                st.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s · {trace['steps']} tool call(s) · {selected_model}")
                if show_trace:
                    render_trace(trace, expanded=True)

        idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if trace.get("trace"):
            st.session_state.traces[idx] = trace

    # Empty / welcome state
    if not st.session_state.messages:
        st.markdown("### 👋 Welcome! What are you looking for today?")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUERIES[:6]):
            with cols[i % 2]:
                if st.button(f"💬 {q}", use_container_width=True, key=f"w_{i}"):
                    st.session_state["_pending"] = q
                    st.rerun()


# EDA Tab
elif page == "📊 Analytics Dashboard":

    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0">📊 Dataset Analytics Dashboard</h2>
        <p style="margin:4px 0 0;opacity:.9">Flipkart Fashion Products · EDA & Insights</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(JSON_PATH):
        st.error(f"❌ Dataset not found: `{JSON_PATH}`")
        st.stop()

    try:
        import plotly.express as px
        PLOTLY = True
    except ImportError:
        PLOTLY = False
        st.warning("⚠️ Install plotly: `pip install plotly`")

    with st.spinner("📊 Loading analytics…"):
        d = load_analytics_data(JSON_PATH)

    avg_price = round(sum(d["prices"])    / len(d["prices"]),    2) if d["prices"]    else 0
    avg_disc  = round(sum(d["discounts"]) / len(d["discounts"]), 1) if d["discounts"] else 0

    st.caption(f"Flipkart Fashion Products · {d['total']:,} products analysed")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{d["total"]:,}</h2><p>Total Products</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{d["total_categories"]}</h2><p>Categories</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{d["total_brands"]:,}</h2><p>Unique Brands</p></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h2>₹{avg_price:,.0f}</h2><p>Avg Price</p></div>',
                    unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><h2>{avg_disc}%</h2><p>Avg Discount</p></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ Category Distribution")
    col1, col2 = st.columns([3, 2])

    with col1:
        if PLOTLY:
            cats = list(d["categories"].keys())
            cnts = list(d["categories"].values())
            fig = px.bar(x=cnts, y=cats, orientation="h",
                         title="Products per Category",
                         color=cnts, color_continuous_scale="Blues",
                         labels={"x": "Products", "y": "Category"})
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=10,r=10,t=40,b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig.update_traces(texttemplate="%{x:,}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(d["categories"])

    with col2:
        if PLOTLY:
            fig2 = px.pie(names=list(d["price_bands"].keys()),
                          values=list(d["price_bands"].values()),
                          title="Price Band Distribution",
                          color_discrete_sequence=["#6dd5ed", "#2193b0", "#1a5276"],
                          hole=0.45)
            fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                               paper_bgcolor="rgba(0,0,0,0)")
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.bar_chart(d["price_bands"])

    # ── Price + Rating Distribution ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💰 Price & ⭐ Rating Distribution")
    col3, col4 = st.columns(2)

    with col3:
        if PLOTLY and d["prices"]:
            capped = [p for p in d["prices"] if p <= 5000]
            fig3 = px.histogram(x=capped, nbins=50,
                                title="Price Distribution (capped ₹5,000)",
                                labels={"x": "Price (₹)", "y": "Products"},
                                color_discrete_sequence=["#2193b0"])
            fig3.update_layout(margin=dict(l=10,r=10,t=40,b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               bargap=0.05)
            fig3.add_vline(x=avg_price, line_dash="dash", line_color="#e74c3c",
                           annotation_text=f"Avg ₹{avg_price:,.0f}")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if PLOTLY and d["ratings"]:
            buckets = {"1.0–2.0":0,"2.0–3.0":0,"3.0–3.5":0,
                       "3.5–4.0":0,"4.0–4.5":0,"4.5–5.0":0}
            for r in d["ratings"]:
                if r < 2:     buckets["1.0–2.0"] += 1
                elif r < 3:   buckets["2.0–3.0"] += 1
                elif r < 3.5: buckets["3.0–3.5"] += 1
                elif r < 4:   buckets["3.5–4.0"] += 1
                elif r < 4.5: buckets["4.0–4.5"] += 1
                else:         buckets["4.5–5.0"] += 1
            fig4 = px.bar(x=list(buckets.keys()), y=list(buckets.values()),
                          title=f"Rating Distribution ({d['rated_products']:,} rated)",
                          labels={"x":"Rating Range","y":"Products"},
                          color=list(buckets.values()), color_continuous_scale="Greens")
            fig4.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=10,r=10,t=40,b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Top Brands ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏷️ Top 20 Brands by Product Count")
    if PLOTLY:
        bs = dict(sorted(d["top_brands"].items(), key=lambda x: x[1]))
        fig5 = px.bar(x=list(bs.values()), y=list(bs.keys()), orientation="h",
                      title="Top 20 Brands",
                      color=list(bs.values()), color_continuous_scale="Teal",
                      labels={"x":"Product Count","y":"Brand"})
        fig5.update_layout(showlegend=False, coloraxis_showscale=False, height=520,
                           margin=dict(l=10,r=10,t=40,b=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig5.update_traces(texttemplate="%{x:,}", textposition="outside")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.bar_chart(d["top_brands"])

    # ── Avg Price per Category + Discount ─────────────────────────────────────
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 💸 Avg Price per Category")
        if PLOTLY and d["cat_avg_price"]:
            cp = dict(sorted(d["cat_avg_price"].items(), key=lambda x: -x[1]))
            fig6 = px.bar(x=list(cp.keys()), y=list(cp.values()),
                          title="Average Selling Price by Category",
                          labels={"x":"Category","y":"Avg Price (₹)"},
                          color=list(cp.values()), color_continuous_scale="Blues")
            fig6.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=10,r=10,t=40,b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig6.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")
            st.plotly_chart(fig6, use_container_width=True)

    with col6:
        st.markdown("### 🎯 Discount Distribution")
        if PLOTLY and d["discounts"]:
            db = {"0–10%":0,"10–25%":0,"25–40%":0,"40–60%":0,"60–75%":0,"75%+":0}
            for disc in d["discounts"]:
                if disc < 10:   db["0–10%"]  += 1
                elif disc < 25: db["10–25%"] += 1
                elif disc < 40: db["25–40%"] += 1
                elif disc < 60: db["40–60%"] += 1
                elif disc < 75: db["60–75%"] += 1
                else:           db["75%+"]   += 1
            fig7 = px.bar(x=list(db.keys()), y=list(db.values()),
                          title="Discount Range Distribution",
                          labels={"x":"Discount Range","y":"Products"},
                          color=list(db.values()), color_continuous_scale="Oranges")
            fig7.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=10,r=10,t=40,b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig7, use_container_width=True)

    # ── Sub-category breakdown ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📦 Sub-category Breakdown")
    sel_cat = st.selectbox("Select category:",
                           options=list(d["subcat_counts"].keys()), index=0)
    if sel_cat and sel_cat in d["subcat_counts"]:
        sub = dict(sorted(d["subcat_counts"][sel_cat].items(), key=lambda x: -x[1]))
        if PLOTLY:
            fig8 = px.bar(x=list(sub.keys()), y=list(sub.values()),
                          title=f"Sub-categories in: {sel_cat}",
                          labels={"x":"Sub-category","y":"Products"},
                          color=list(sub.values()), color_continuous_scale="Purples")
            fig8.update_layout(showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=10,r=10,t=40,b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_tickangle=-35)
            fig8.update_traces(texttemplate="%{y:,}", textposition="outside")
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.bar_chart(sub)

    # Summary table
    st.markdown("---")
    st.markdown("### 📋 Category Summary Table")
    rows = []
    for cat, count in d["categories"].items():
        rows.append({
            "Category":      cat,
            "Products":      f"{count:,}",
            "Avg Price (₹)": f"₹{d['cat_avg_price'].get(cat, 0):,.0f}",
            "Avg Rating":    f"{d['cat_avg_rating'].get(cat, 0):.2f}"
                             if d["cat_avg_rating"].get(cat) else "N/A",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
