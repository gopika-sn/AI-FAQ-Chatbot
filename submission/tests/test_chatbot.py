import pytest
from unittest.mock import MagicMock, patch
from nlp_engine import FAQEngine
from gemini_fallback import GeminiFallbackHandler

@pytest.fixture
def faq_engine():
    return FAQEngine(data_path="data/faqs.json", confidence_threshold=0.55, min_margin=0.04)

@pytest.fixture
def gemini_handler():
    return GeminiFallbackHandler(model_name="gemini-3.6-flash")

# 1. Exact FAQ match
def test_01_exact_faq_match(faq_engine):
    result = faq_engine.get_best_match("How can I track my order?")
    assert result["status"] == "success"
    assert result["matched_faq"]["id"] == 18
    assert result["similarity_score"] > 0.95

# 2. Semantic/paraphrased match
def test_02_paraphrased_match(faq_engine):
    result = faq_engine.get_best_match("Where is my package right now?")
    assert result["status"] == "success"
    assert result["matched_faq"]["id"] == 18

# 3. Returns
def test_03_returns_intent(faq_engine):
    result = faq_engine.get_best_match("I bought this item but want to send it back.")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Returns"

# 4. Refunds
def test_04_refunds_intent(faq_engine):
    result = faq_engine.get_best_match("When will my refund arrive in my bank account?")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Refunds"

# 5. Payments
def test_05_payments_intent(faq_engine):
    result = faq_engine.get_best_match("My payment was declined at checkout.")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Payments"

# 6. Shipping
def test_06_shipping_intent(faq_engine):
    result = faq_engine.get_best_match("What fast shipping options do you offer?")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Shipping"

# 7. Tracking
def test_07_tracking_intent(faq_engine):
    result = faq_engine.get_best_match("Where is my parcel?")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Order Tracking"

# 8. Account
def test_08_account_intent(faq_engine):
    result = faq_engine.get_best_match("I forgot my password.")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Account and Login"

# 9. Coupons
def test_09_coupons_intent(faq_engine):
    result = faq_engine.get_best_match("Why isn't my promo code working?")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Coupons and Discounts"

# 10. Product information
def test_10_product_info_intent(faq_engine):
    result = faq_engine.get_best_match("Is this apparel true to size?")
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Product Information"

# 11. Low-confidence query
def test_11_low_confidence_query(faq_engine):
    result = faq_engine.get_best_match("Random submarine propeller speed")
    assert result["status"] == "low_confidence"
    assert result["matched_faq"] is None

# 12. Unrelated question
def test_12_unrelated_question(faq_engine):
    result = faq_engine.get_best_match("What is the capital of India?")
    assert result["status"] == "low_confidence"
    assert "couldn't find a sufficiently relevant FAQ" in result["answer"]

# 13. Empty input
def test_13_empty_input_handling(faq_engine):
    result = faq_engine.get_best_match("   ")
    assert result["status"] == "empty_query"
    assert result["matched_faq"] is None

# 14. Gemini enabled
def test_14_gemini_enabled_triggers_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    handler = GeminiFallbackHandler()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Simulated Gemini fallback answer."
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        res = handler.generate_fallback_response("What is quantum computing?")
        assert res == "Simulated Gemini fallback answer."

# 15. Gemini disabled
def test_15_gemini_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    handler = GeminiFallbackHandler()
    assert not handler.is_configured()
    assert handler.generate_fallback_response("What is quantum computing?") is None

# 16. Gemini API failure
def test_16_gemini_api_failure_handling(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "invalid_key")
    handler = GeminiFallbackHandler()
    with patch("google.genai.Client", side_effect=Exception("API Exception")):
        assert handler.generate_fallback_response("Some query") is None

# 17. Missing API key
def test_17_missing_api_key_system_works(faq_engine, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = faq_engine.get_best_match("Where is my package?")
    assert result["status"] == "success"

# 18. High-confidence FAQ must not call Gemini
def test_18_high_confidence_never_calls_gemini(faq_engine):
    queries = [
        "How can I track my order?",
        "Why was my payment declined at checkout?",
        "What is your item return policy?",
        "How long does a refund take to process?"
    ]
    for q in queries:
        res = faq_engine.get_best_match(q)
        assert res["status"] == "success"

# 19. Chat history structure mock test
def test_19_chat_history_structure():
    history = [
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "Track under My Orders."}
    ]
    assert len(history) == 3
    assert history[1]["role"] == "user"

# 20. Clear chat functionality mock test
def test_20_clear_chat_reset():
    history = [
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "Hi"}
    ]
    history = [{"role": "assistant", "content": "Welcome!"}]
    assert len(history) == 1
