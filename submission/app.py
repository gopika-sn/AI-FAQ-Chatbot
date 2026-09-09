import streamlit as st
import pandas as pd
from nlp_engine import FAQEngine
from gemini_fallback import GeminiFallbackHandler

# 1. Page Configuration
st.set_page_config(
    page_title="ShopAssist AI - E-Commerce FAQ Assistant & Benchmark",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Polished E-Commerce CSS Styling
st.markdown("""
<style>
    /* Global App Background & Typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 1.5rem 1rem 1.0rem 1rem;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .brand-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Response Type Tags & Badges */
    .badge-faq {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 8px;
    }

    .badge-ai {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-right: 8px;
    }

    .badge-high {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-top: 8px;
    }

    .badge-medium {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
        margin-top: 8px;
    }

    .badge-low {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        margin-top: 8px;
    }

    /* Footer Container */
    .footer-container {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load & Cache Engine and Gemini Handler
@st.cache_resource
def load_engine():
    return FAQEngine(
        data_path="data/faqs.json",
        model_name="all-MiniLM-L6-v2",
        confidence_threshold=0.55,
        min_margin=0.04
    )

@st.cache_resource
def load_gemini_handler():
    return GeminiFallbackHandler(model_name="gemini-3.6-flash")

try:
    faq_engine = load_engine()
    gemini_handler = load_gemini_handler()
except Exception:
    st.error("ShopAssist AI is currently initializing. Please refresh in a moment.")
    st.stop()

# 4. Navigation & Sidebar Setup
page_selection = st.sidebar.radio("Navigation", ["💬 Chatbot Assistant", "📊 System Evaluation"], index=0)

with st.sidebar:
    st.markdown("---")
    st.title("🛍️ ShopAssist AI")
    st.caption("E-Commerce Customer Support")
    
    # New Chat / Clear Chat Button
    if st.button("🔄 New Chat / Clear History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! 👋 I'm ShopAssist AI. Ask me anything about orders, shipping, payments, returns, refunds, products, accounts, and more."
            }
        ]
        st.rerun()
        
    st.markdown("---")
    
    # Phase 6 & 11: Sidebar Retrieval & Fallback Settings
    st.subheader("⚙️ System Architecture Settings")
    st.markdown("• **FAQ semantic search**: `Always enabled` ✅")
    
    gemini_enabled = st.checkbox(
        "Gemini AI Fallback",
        value=gemini_handler.is_configured(),
        help="Triggers Gemini LLM only when FAQ semantic search confidence is low."
    )
    
    if gemini_enabled and not gemini_handler.is_configured():
        st.caption("⚠️ `GEMINI_API_KEY` not set in `.env`. Gemini fallback will remain inactive until key is provided.")

    # Phase 11: UI Explanation
    st.caption(
        "ℹ️ *FAQ answers are retrieved using semantic similarity. "
        "Gemini is used only when no sufficiently relevant FAQ is found.*"
    )

    st.markdown("---")
    
    # About Section
    st.subheader("ℹ️ About ShopAssist AI")
    st.write(
        "ShopAssist AI is an automated customer support assistant designed to help customers "
        "instantaneously answer questions about orders, payments, deliveries, and returns."
    )
    
    st.markdown("---")
    
    # Model Info & How It Works
    st.subheader("💡 How It Works")
    st.info("""
    - **Primary Retrieval**: `all-MiniLM-L6-v2` + Cosine Similarity
    - **Preprocessing**: spaCy sentence normalization
    - **Optional Fallback**: Gemini 3.6 Flash (`google-genai` SDK)
    """)
    
    st.markdown("---")
    
    # Supported FAQ Categories
    st.subheader("📂 Supported Categories")
    categories = sorted(list(set(f["category"] for f in faq_engine.faqs)))
    st.write(f"Total FAQs Indexed: **{len(faq_engine.faqs)}**")
    
    selected_cat = st.selectbox("Explore Category", ["All Categories"] + categories)
    filtered_faqs = [f for f in faq_engine.faqs if selected_cat == "All Categories" or f["category"] == selected_cat]
    
    for faq in filtered_faqs:
        with st.expander(f"[{faq['category']}] {faq['question']}"):
            st.write(f"**Answer**: {faq['answer']}")

# -----------------------------------------------------------------------------
# PAGE 1: CHATBOT ASSISTANT
# -----------------------------------------------------------------------------
if page_selection == "💬 Chatbot Assistant":
    WELCOME_MSG = "Hi! 👋 I'm ShopAssist AI. Ask me anything about orders, shipping, payments, returns, refunds, products, accounts, and more."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MSG
            }
        ]

    # Main UI Header
    st.markdown("""
    <div class="header-container">
        <div class="brand-title">ShopAssist AI</div>
        <div class="brand-subtitle">Your intelligent e-commerce FAQ assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Clickable Example Question Chips
    st.markdown("**Suggested Questions:**")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("📦 Where is my order?", use_container_width=True):
            st.session_state.pending_input = "Where is my package?"
    with c2:
        if st.button("🔄 Return an item?", use_container_width=True):
            st.session_state.pending_input = "How can I return an item?"
    with c3:
        if st.button("💳 Payment options?", use_container_width=True):
            st.session_state.pending_input = "What payment methods do you accept?"
    with c4:
        if st.button("🚚 Delivery time?", use_container_width=True):
            st.session_state.pending_input = "How long does delivery take?"
    with c5:
        if st.button("🏷️ Coupon code?", use_container_width=True):
            st.session_state.pending_input = "Why isn't my coupon working?"

    st.markdown("---")

    def render_match_details(msg):
        """Helper to render response type indicators, confidence badges, and expandable Match Details section (Phase 8)."""
        response_type = msg.get("response_type", "FAQ-based response")
        
        # Phase 8: Response Badges
        if response_type == "AI-generated response":
            st.markdown('<div class="badge-ai">✨ AI-generated response</div>', unsafe_allow_html=True)
            with st.expander("🔍 Match Details"):
                st.write("**Response Source**: AI-generated response (Gemini Fallback)")
                st.write("**Reason**: No sufficiently confident FAQ match found in knowledge base (similarity score < 0.55)")
        else:
            st.markdown('<div class="badge-faq">📚 FAQ-based response</div>', unsafe_allow_html=True)
            if "score" in msg and msg["score"] is not None and msg["score"] > 0:
                conf_level = msg.get("confidence_level", "Medium confidence")
                if conf_level == "High confidence":
                    badge_class = "badge-high"
                    icon = "🟢"
                elif conf_level == "Medium confidence":
                    badge_class = "badge-medium"
                    icon = "🟡"
                else:
                    badge_class = "badge-low"
                    icon = "🔴"

                st.markdown(
                    f'<div class="{badge_class}">{icon} {conf_level} (Score: {msg["score"]:.4f})</div>',
                    unsafe_allow_html=True
                )

                with st.expander("🔍 Match Details"):
                    if "matched_question" in msg and msg["matched_question"]:
                        st.write(f"**Matched FAQ**: {msg['matched_question']}")
                        st.write(f"**Category**: `{msg.get('matched_category', 'General')}`")
                    else:
                        st.write("**Matched FAQ**: None (Below threshold or ambiguous)")
                    st.write(f"**Similarity Score**: `{msg['score']:.4f}` ({msg.get('confidence', '0%')})")
                    st.write(f"**Confidence Level**: `{conf_level}`")
                    
                    if "top_candidates" in msg and msg["top_candidates"]:
                        st.markdown("**Top 3 Candidates:**")
                        for rank, cand in enumerate(msg["top_candidates"], start=1):
                            st.write(f"{rank}. `[{cand['category']}]` **{cand['question']}** — Score: `{cand['score']:.4f}` ({cand['confidence_percent']})")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg["content"] != WELCOME_MSG:
                render_match_details(msg)

    # Handle User Input & Send Interaction
    user_prompt = st.chat_input("Type your question here...")

    if "pending_input" in st.session_state:
        user_prompt = st.session_state.pending_input
        del st.session_state.pending_input

    if user_prompt:
        clean_user_input = user_prompt.strip()
        if not clean_user_input:
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": clean_user_input})
        with st.chat_message("user"):
            st.markdown(clean_user_input)

        # Phase 5: Decision Flow Implementation
        try:
            with st.spinner("Processing question..."):
                # Step 1: Primary FAQ Semantic Search
                match_result = faq_engine.get_best_match(clean_user_input)

            if match_result["status"] == "success":
                # High/Medium confidence -> FAQ Answer
                faq = match_result["matched_faq"]
                assistant_entry = {
                    "role": "assistant",
                    "content": match_result["answer"],
                    "score": match_result["similarity_score"],
                    "confidence": match_result["confidence_percent"],
                    "confidence_level": match_result.get("confidence_level", "High confidence"),
                    "response_type": "FAQ-based response",
                    "matched_question": faq["question"],
                    "matched_category": faq["category"],
                    "top_candidates": match_result.get("top_candidates", []),
                    "margin": match_result.get("margin", 0.0)
                }
            else:
                # Low confidence -> Attempt Gemini Fallback if enabled
                ai_reply = None
                if gemini_enabled and gemini_handler.is_configured():
                    with st.spinner("Consulting Gemini AI fallback..."):
                        ai_reply = gemini_handler.generate_fallback_response(clean_user_input)

                if ai_reply:
                    assistant_entry = {
                        "role": "assistant",
                        "content": ai_reply,
                        "score": match_result["similarity_score"],
                        "confidence": match_result["confidence_percent"],
                        "confidence_level": "Low confidence",
                        "response_type": "AI-generated response",
                        "matched_question": None,
                        "matched_category": None,
                        "top_candidates": match_result.get("top_candidates", []),
                        "margin": match_result.get("margin", 0.0)
                    }
                else:
                    # Phase 9: Normal FAQ low-confidence fallback
                    fallback_msg = "I'm sorry, I couldn't find a sufficiently relevant FAQ for your question right now. Please try rephrasing your question or contact customer support."
                    assistant_entry = {
                        "role": "assistant",
                        "content": fallback_msg,
                        "score": match_result["similarity_score"],
                        "confidence": match_result["confidence_percent"],
                        "confidence_level": "Low confidence",
                        "response_type": "FAQ-based response",
                        "matched_question": None,
                        "matched_category": None,
                        "top_candidates": match_result.get("top_candidates", []),
                        "margin": match_result.get("margin", 0.0)
                    }
        except Exception:
            # Phase 9: Exception handling prevents app crash
            fallback_msg = "I'm sorry, I couldn't find a sufficiently relevant FAQ for your question right now. Please try rephrasing your question or contact customer support."
            assistant_entry = {
                "role": "assistant",
                "content": fallback_msg,
                "score": 0.0,
                "confidence": "0.0%",
                "confidence_level": "Low confidence",
                "response_type": "FAQ-based response",
                "matched_question": None,
                "matched_category": None,
                "top_candidates": []
            }

        st.session_state.messages.append(assistant_entry)
        with st.chat_message("assistant"):
            st.markdown(assistant_entry["content"])
            render_match_details(assistant_entry)

        st.rerun()

# -----------------------------------------------------------------------------
# PAGE 2: SYSTEM EVALUATION BENCHMARK
# -----------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="header-container">
        <div class="brand-title">📊 System Evaluation & Benchmark</div>
        <div class="brand-subtitle">Empirical retrieval accuracy of Sentence Transformers + Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This page objectively evaluates the retrieval performance of the **`all-MiniLM-L6-v2`** Sentence Transformer engine.
    It runs a benchmark dataset of **25 test cases** (20 paraphrased e-commerce questions across 18 categories + 5 out-of-domain unrelated questions)
    and computes matching metrics without using an LLM.
    """)

    EVALUATION_DATASET = [
        {"query": "Where can I check my past purchases?", "expected_cat": "Orders", "expected_id": 1, "is_unrelated": False},
        {"query": "I want to cancel my purchase before shipping", "expected_cat": "Order Cancellation", "expected_id": 4, "is_unrelated": False},
        {"query": "I ordered the wrong size, can I swap it?", "expected_cat": "Order Modification", "expected_id": 8, "is_unrelated": False},
        {"query": "What are your fast shipping choices?", "expected_cat": "Shipping", "expected_id": 10, "is_unrelated": False},
        {"query": "What is my expected delivery date?", "expected_cat": "Delivery", "expected_id": 14, "is_unrelated": False},
        {"query": "Where is my parcel right now?", "expected_cat": "Order Tracking", "expected_id": 18, "is_unrelated": False},
        {"query": "Can I return an item I already opened?", "expected_cat": "Returns", "expected_id": 25, "is_unrelated": False},
        {"query": "How can I get my money back?", "expected_cat": "Refunds", "expected_id": 26, "is_unrelated": False},
        {"query": "My payment failed at checkout", "expected_cat": "Payments", "expected_id": 31, "is_unrelated": False},
        {"query": "Where do I enter my promo code?", "expected_cat": "Coupons and Discounts", "expected_id": 34, "is_unrelated": False},
        {"query": "How to reset lost login password?", "expected_cat": "Account and Login", "expected_id": 39, "is_unrelated": False},
        {"query": "Is this apparel true to size?", "expected_cat": "Product Information", "expected_id": 42, "is_unrelated": False},
        {"query": "When will this sold out item be restocked?", "expected_cat": "Product Availability", "expected_id": 46, "is_unrelated": False},
        {"query": "I need a tax invoice PDF for my purchase", "expected_cat": "Invoices and Receipts", "expected_id": 49, "is_unrelated": False},
        {"query": "How to email customer support team?", "expected_cat": "Customer Support", "expected_id": 52, "is_unrelated": False},
        {"query": "Website crashing and page won't load", "expected_cat": "Technical Issues", "expected_id": 55, "is_unrelated": False},
        {"query": "Will you match a lower price from another store?", "expected_cat": "Store Policies", "expected_id": 58, "is_unrelated": False},
        {"query": "Where are your physical retail stores?", "expected_cat": "General Questions", "expected_id": 61, "is_unrelated": False},
        {"query": "How do I qualify for free delivery?", "expected_cat": "Shipping", "expected_id": 11, "is_unrelated": False},
        {"query": "I was charged twice for my order", "expected_cat": "Payments", "expected_id": 32, "is_unrelated": False},
        
        {"query": "What is the capital of India?", "expected_cat": "Unrelated (Reject)", "expected_id": None, "is_unrelated": True},
        {"query": "Tell me a funny joke.", "expected_cat": "Unrelated (Reject)", "expected_id": None, "is_unrelated": True},
        {"query": "How do I cook chicken biryani?", "expected_cat": "Unrelated (Reject)", "expected_id": None, "is_unrelated": True},
        {"query": "Write Python code for me.", "expected_cat": "Unrelated (Reject)", "expected_id": None, "is_unrelated": True},
        {"query": "What is quantum computing?", "expected_cat": "Unrelated (Reject)", "expected_id": None, "is_unrelated": True},
    ]

    st.subheader("▶️ Live Evaluation Benchmark")
    if st.button("🚀 Run Full System Evaluation", type="primary"):
        results_rows = []
        correct_matches = 0
        correct_rejections = 0
        incorrect_matches = 0

        with st.spinner("Executing retrieval benchmark on 25 test cases..."):
            for idx, test_case in enumerate(EVALUATION_DATASET, start=1):
                query = test_case["query"]
                exp_cat = test_case["expected_cat"]
                exp_id = test_case["expected_id"]
                is_unrelated = test_case["is_unrelated"]

                match_res = faq_engine.get_best_match(query)
                score = match_res["similarity_score"]
                status = match_res["status"]
                conf_level = match_res.get("confidence_level", "Low confidence")

                if status == "success":
                    pred_faq = match_res["matched_faq"]
                    pred_cat = pred_faq["category"]
                    pred_id = pred_faq["id"]
                    pred_str = f"[{pred_cat}] FAQ #{pred_id}"

                    if not is_unrelated and (pred_id == exp_id or pred_cat == exp_cat):
                        is_correct = True
                        correct_matches += 1
                        result_icon = "✅ PASS"
                    else:
                        is_correct = False
                        incorrect_matches += 1
                        result_icon = "❌ FAIL"
                else:
                    pred_str = "None (Rejected)"
                    if is_unrelated:
                        is_correct = True
                        correct_rejections += 1
                        result_icon = "✅ REJECTED (PASS)"
                    else:
                        is_correct = False
                        incorrect_matches += 1
                        result_icon = "❌ FAIL"

                results_rows.append({
                    "#": idx,
                    "Test Question": query,
                    "Expected Category": exp_cat,
                    "Predicted Match": pred_str,
                    "Similarity Score": f"{score:.4f}",
                    "Confidence Level": conf_level,
                    "Benchmark Result": result_icon
                })

        total_tests = len(EVALUATION_DATASET)
        total_correct = correct_matches + correct_rejections
        accuracy_pct = round((total_correct / total_tests) * 100, 1)

        st.markdown("### 📈 Evaluation Performance Summary")
        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Total Test Cases", f"{total_tests}")
        m2.metric("Correct Matched FAQs", f"{correct_matches}")
        m3.metric("Correctly Rejected", f"{correct_rejections}")
        m4.metric("Incorrect Matches", f"{incorrect_matches}")
        m5.metric("Overall Accuracy", f"{accuracy_pct}%")

        st.markdown("---")
        st.markdown("### 📋 Detailed Test Case Evaluation Results")
        
        df_results = pd.DataFrame(results_rows)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

    else:
        st.info("Click the **'🚀 Run Full System Evaluation'** button above to execute the live 25-question retrieval benchmark!")

# Footer
st.markdown("""
<div class="footer-container">
    Powered by semantic search • Sentence Transformers • Cosine Similarity
</div>
""", unsafe_allow_html=True)
