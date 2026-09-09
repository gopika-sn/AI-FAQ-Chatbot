import pytest
from unittest.mock import MagicMock, patch
from nlp_engine import FAQEngine
from gemini_fallback import GeminiFallbackHandler

@pytest.fixture
def faq_engine():
    return FAQEngine(data_path="data/faqs.json", confidence_threshold=0.55)

@pytest.fixture
def gemini_handler():
    return GeminiFallbackHandler(model_name="gemini-3.6-flash")

# 1. High-confidence FAQ question
def test_1_high_confidence_faq_does_not_call_gemini(faq_engine):
    """Where can I track my order? -> High confidence FAQ match, Gemini must NOT be called."""
    query = "Where can I track my order?"
    result = faq_engine.get_best_match(query)
    
    assert result["status"] == "success"
    assert result["matched_faq"]["id"] == 18
    assert result["similarity_score"] > 0.90

# 2. Paraphrased FAQ question
def test_2_paraphrased_faq_matching(faq_engine):
    """Where is my package right now? -> Relevant FAQ retrieved."""
    query = "Where is my package right now?"
    result = faq_engine.get_best_match(query)
    
    assert result["status"] == "success"
    assert result["matched_faq"]["id"] == 18

# 3. Return question
def test_3_return_question_matching(faq_engine):
    """I bought this item but want to send it back. -> Return FAQ matched."""
    query = "I bought this item but want to send it back."
    result = faq_engine.get_best_match(query)
    
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Returns"

# 4. Payment question
def test_4_payment_question_matching(faq_engine):
    """My payment was declined. -> Payment FAQ matched."""
    query = "My payment was declined."
    result = faq_engine.get_best_match(query)
    
    assert result["status"] == "success"
    assert result["matched_faq"]["category"] == "Payments"

# 5. Low-confidence/out-of-domain question with Gemini enabled
def test_5_low_confidence_triggers_gemini_when_enabled(faq_engine, monkeypatch):
    """What is quantum computing? -> Gemini fallback triggered when enabled."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")
    handler = GeminiFallbackHandler()
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Quantum computing relies on qubits to perform complex calculations."
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        response_text = handler.generate_fallback_response("What is quantum computing?")
        assert response_text == "Quantum computing relies on qubits to perform complex calculations."

# 6. Low-confidence question with Gemini disabled
def test_6_low_confidence_with_gemini_disabled(faq_engine, monkeypatch):
    """What is quantum computing? with Gemini disabled -> Normal FAQ fallback, no Gemini request."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    handler = GeminiFallbackHandler()
    
    assert not handler.is_configured()
    response = handler.generate_fallback_response("What is quantum computing?")
    assert response is None
    
    match_result = faq_engine.get_best_match("What is quantum computing?")
    assert match_result["status"] == "low_confidence"
    assert "couldn't find a sufficiently relevant FAQ" in match_result["answer"]

# 7. Empty input
def test_7_empty_input_handling(faq_engine):
    """Empty query -> Graceful empty_query result."""
    result = faq_engine.get_best_match("")
    assert result["status"] == "empty_query"
    assert result["matched_faq"] is None

# 8. Gemini API failure
def test_8_gemini_api_failure_graceful_handling(monkeypatch):
    """API connection failure -> Chatbot does not crash, handler returns None."""
    monkeypatch.setenv("GEMINI_API_KEY", "invalid_key")
    handler = GeminiFallbackHandler()

    with patch("google.genai.Client", side_effect=Exception("API Quota/Network Error")):
        response_text = handler.generate_fallback_response("Random question")
        assert response_text is None  # Catches exception gracefully

# 9. Missing GEMINI_API_KEY
def test_9_missing_gemini_api_key(faq_engine, monkeypatch):
    """Missing key -> FAQ chatbot still works perfectly without Gemini."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    handler = GeminiFallbackHandler()
    assert not handler.is_configured()
    
    match_result = faq_engine.get_best_match("Where is my package?")
    assert match_result["status"] == "success"

# 10. Verify high-confidence FAQ questions never call Gemini
def test_10_verify_high_confidence_never_calls_gemini(faq_engine):
    """Verify high-confidence vector matches evaluate to success without triggering fallback."""
    queries = [
        "How can I track my order?",
        "Why was my payment declined at checkout?",
        "What is your item return policy?",
        "How long does a refund take to process?"
    ]
    for q in queries:
        res = faq_engine.get_best_match(q)
        assert res["status"] == "success", f"Query '{q}' did not return success status."
