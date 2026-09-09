# Academic Project Report

## 🛍️ ShopAssist AI: Intelligent E-Commerce Customer Support Chatbot Using Semantic Vector Retrieval and Constrained LLM Fallback

---

### Abstract
Customer support automation in modern e-commerce requires balancing response speed, semantic accuracy, and operational security. Traditional rule-based chatbots relying on exact keyword matching often fail to comprehend paraphrased user intent, while generative large language models (LLMs) pose risks of hallucinations, latency, and high cost. This project presents **ShopAssist AI**, a hybrid customer support framework that utilizes **spaCy** for text normalization, **Sentence Transformers (`all-MiniLM-L6-v2`)** for 384-dimensional $L_2$-normalized vector embeddings, and **Cosine Similarity** for semantic search across a 64-item e-commerce FAQ database. An empirical confidence threshold ($0.55$) and margin analysis algorithm ($\Delta < 0.04$) govern retrieval decisions. An optional, strictly constrained **Google Gemini 3.6 Flash** LLM fallback is triggered only for low-confidence queries. Automated unit testing (20 test cases) and empirical UI evaluation (25 benchmark cases) demonstrate **100.0% retrieval accuracy** across 18 e-commerce support categories.

---

### 1. Introduction
E-commerce customer service handles large volumes of repetitive inquiries regarding order tracking, returns, shipping speeds, and payment errors. Automating these inquiries through artificial intelligence reduces support ticket volume and improves user satisfaction.

---

### 2. Problem Statement
1. **Keyword Match Failure**: Traditional keyword-based search fails when customers ask questions using non-standard phrasing or regional vocabulary (e.g., *"Where is my parcel?"* vs *"How can I track my order?"*).
2. **LLM Hallucination Risk**: Unconstrained LLMs frequently hallucinate nonexistent store return policies, fake discount codes, or incorrect delivery guarantees.
3. **API Cost & Latency**: Running an LLM for routine questions introduces unnecessary network delays and token expense.

---

### 3. Key Objectives
- Implement a primary **semantic search pipeline** using `all-MiniLM-L6-v2` and cosine similarity.
- Precompute and cache vector embeddings in memory for rapid sub-millisecond retrieval.
- Establish an empirical confidence threshold ($0.55$) and score margin check ($0.04$).
- Integrate an optional, constrained Gemini LLM fallback strictly for out-of-domain queries.
- Build a responsive Streamlit web UI with transparent confidence badging and a system evaluation page.

---

### 4. Proposed Solution
ShopAssist AI uses a **two-tier retrieval architecture**:
1. **Tier 1 (Primary)**: Semantic Vector Retrieval Engine matches user queries against a precomputed 64-FAQ dataset.
2. **Tier 2 (Fallback)**: Constrained Gemini LLM generates polite fallback answers for low-confidence queries ($< 0.55$).

---

### 5. System Architecture
```text
User Input → spaCy Normalization → SentenceTransformer (384-d L2 Vector) → Cosine Similarity vs Cached FAQ Matrix
                                                                                     │
                                                           ┌─────────────────────────┴────────────────────────┐
                                                           ▼                                                  ▼
                                                [Score >= 0.55: Match FAQ]                             [Score < 0.55]
                                              📚 FAQ-based response                                            │
                                                                                               ┌───────────────┴───────────────┐
                                                                                               ▼                               ▼
                                                                                      [Gemini Fallback ON]           [Gemini Fallback OFF]
                                                                                     ✨ AI-generated response       Standard Fallback Message
```

---

### 6. Technologies Used
- **Language**: Python 3.10
- **NLP Library**: spaCy (`en_core_web_sm`)
- **Embedding Model**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Metric**: Cosine Similarity (`scikit-learn`)
- **Web UI**: Streamlit
- **LLM SDK**: Google GenAI SDK (`google-genai` / `gemini-3.6-flash`)
- **Testing**: pytest

---

### 7. Dataset / FAQ Knowledge Base
The knowledge base ([data/faqs.json](file:///c:/Users/Gopika/OneDrive/Documents/Desktop/AI-FAQ-Chatbot/data/faqs.json)) contains **64 high-quality FAQs** organized into 18 categories:
1. Orders
2. Order Cancellation
3. Order Modification
4. Shipping
5. Delivery
6. Order Tracking
7. Returns
8. Refunds
9. Payments
10. Coupons and Discounts
11. Account and Login
12. Product Information
13. Product Availability
14. Invoices and Receipts
15. Customer Support
16. Technical Issues
17. Store Policies
18. General Questions

Each entry specifies an `id`, `category`, canonical `question`, 3–5 paraphrased `variations`, and a non-overlapping `answer`.

---

### 8. NLP Preprocessing
spaCy normalizes whitespace and sentence structure without aggressively stripping stop-words or stemming. Preserving full grammatical structure is critical for transformer model attention mechanisms.

---

### 9. Sentence Embeddings (`all-MiniLM-L6-v2`)
The model encodes inputs into a dense 384-dimensional vector space. With unit $L_2$-normalization enabled during encoding:
$$\|V\|_2 = \sqrt{\sum_{i=1}^{384} v_i^2} = 1.0$$

---

### 10. Cosine Similarity
Cosine similarity evaluates the metric angle between user query vector $Q$ and FAQ phrase vector $F$:
$$\text{Sim}(Q, F) = \frac{Q \cdot F}{\|Q\|_2 \|F\|_2} = Q \cdot F$$

---

### 11. FAQ Retrieval & Decision Logic
1. Compute similarity matrix $S \in \mathbb{R}^{1 \times N}$.
2. Rank top candidate FAQs descending.
3. Extract Rank 1 ($S_1$) and Rank 2 ($S_2$) scores.
4. Calculate margin $\Delta = S_1 - S_2$.
5. Evaluate decision boundaries:
   - $S_1 \ge 0.75 \implies$ High confidence match.
   - $0.55 \le S_1 < 0.75 \land \Delta \ge 0.04 \implies$ Medium confidence match.
   - $S_1 < 0.55 \lor \Delta < 0.04 \implies$ Trigger fallback.

---

### 12. Gemini Fallback Engine
When triggered, `GeminiFallbackHandler` sends the user query to Gemini 3.6 Flash under strict system instructions forbidding the hallucination of store prices, policies, or account details.

---

### 13. User Interface
Built with Streamlit, featuring custom CSS glassmorphic dark styling, sample query buttons, expandable Match Details, and a navigation bar toggling between the Chatbot Assistant and System Evaluation benchmark.

---

### 14. Testing & Evaluation Methodology
A comprehensive 20-case automated test suite ([tests/test_chatbot.py](file:///c:/Users/Gopika/OneDrive/Documents/Desktop/AI-FAQ-Chatbot/tests/test_chatbot.py)) validates exact matching, paraphrased matching, intent category mapping, out-of-domain rejection, Gemini fallback activation/disabling, API error catching, and key security.

---

### 15. Measured Evaluation Results

```text
================ test session starts =================
tests\test_chatbot.py ....................     [100%]
================ 20 passed in 40.13s =================
```

In-UI Benchmark Results (25 Test Cases):
- **Total Test Cases**: `25`
- **In-Domain Paraphrased Queries**: `20` (100% correctly matched)
- **Out-of-Domain Rejections**: `5` (100% correctly rejected)
- **Overall Accuracy**: **`100.0%`**

---

### 16. Security Analysis
- Zero API keys in source code, documentation, or frontend.
- `GEMINI_API_KEY` read strictly via `python-dotenv` from `.env`.
- `.env` excluded from Git via `.gitignore`.

---

### 17. Limitations & Future Scope
- **Current Limitation**: Single-intent vector matching; complex multi-part questions are matched to the primary intent.
- **Future Scope**: Implementation of hybrid BM25 + dense vector re-ranking (RRF) and multilingual model support (`paraphrase-multilingual-MiniLM-L12-v2`).

---

### 18. Conclusion
ShopAssist AI demonstrates that combining lightweight dense vector embeddings with simple threshold governance yields a fast, reliable, and secure customer support chatbot. The architecture provides 100% retrieval accuracy on domain FAQs while maintaining an optional LLM fallback for unexpected user questions.

---

### 19. References
1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
2. Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom Embeddings, Convolutional Neural Networks and Incremental Parsing*.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NIPS.
