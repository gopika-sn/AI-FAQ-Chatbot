import pytest
from nlp_engine import FAQEngine

@pytest.fixture
def faq_engine():
    return FAQEngine(data_path="data/faqs.json", confidence_threshold=0.55, min_margin=0.04)

def test_exact_matching(faq_engine):
    """Test exact FAQ question matching."""
    query = "How can I track my order?"
    result = faq_engine.get_best_match(query)
    assert result["status"] == "success"
    assert result["matched_faq"]["id"] == 18
    assert result["similarity_score"] > 0.95
    assert result["confidence_level"] == "High confidence"

def test_paraphrase_matching_requirement_15(faq_engine):
    """
    Test all 15 paraphrased questions explicitly specified in Requirement 15:
    1. 'Where is my package?'
    2. 'Can I see where my order is?'
    3. 'I want to know the location of my parcel.'
    4. 'My payment didn't work.'
    5. 'Why was my payment declined?'
    6. 'I couldn't complete the payment.'
    7. 'I don't want the product anymore.'
    8. 'How can I send this item back?'
    9. 'I need to return something I bought.'
    10. 'When will my refund arrive?'
    11. 'How long does it take to get my money back?'
    12. 'Can I change the address after placing an order?'
    13. 'I entered the wrong delivery address.'
    14. 'Why isn't my coupon working?'
    15. 'My promo code doesn't apply.'
    """
    test_cases = [
        ("Where is my package?", 18, "Order Tracking"),
        ("Can I see where my order is?", 18, "Order Tracking"),
        ("I want to know the location of my parcel.", 18, "Order Tracking"),
        ("My payment didn't work.", 31, "Payments"),
        ("Why was my payment declined?", 31, "Payments"),
        ("I couldn't complete the payment.", 31, "Payments"),
        ("I don't want the product anymore.", 22, "Returns"),
        ("How can I send this item back?", 22, "Returns"),
        ("I need to return something I bought.", 22, "Returns"),
        ("When will my refund arrive?", 27, "Refunds"),
        ("How long does it take to get my money back?", 27, "Refunds"),
        ("Can I change the address after placing an order?", 16, "Delivery"),
        ("I entered the wrong delivery address.", 16, "Delivery"),
        ("Why isn't my coupon working?", 35, "Coupons and Discounts"),
        ("My promo code doesn't apply.", 35, "Coupons and Discounts"),
    ]

    for query, expected_id, expected_category in test_cases:
        result = faq_engine.get_best_match(query)
        assert result["status"] == "success", f"Query '{query}' failed with status {result['status']}"
        assert result["matched_faq"]["id"] == expected_id, f"Query '{query}' matched ID {result['matched_faq']['id']} ({result['matched_faq']['question']}) instead of {expected_id}"
        assert result["matched_faq"]["category"] == expected_category
        assert result["similarity_score"] >= 0.55

def test_unrelated_questions_requirement_16(faq_engine):
    """
    Test unrelated questions from Requirement 16:
    1. 'What is the capital of India?'
    2. 'Tell me a joke.'
    3. 'Write Python code for me.'
    4. 'What is quantum computing?'
    """
    unrelated_queries = [
        "What is the capital of India?",
        "Tell me a joke.",
        "Write Python code for me.",
        "What is quantum computing?",
    ]

    for query in unrelated_queries:
        result = faq_engine.get_best_match(query)
        assert result["status"] == "low_confidence", f"Unrelated query '{query}' was falsely matched with status {result['status']} (score: {result['similarity_score']})"
        assert result["matched_faq"] is None
        assert result["confidence_level"] == "Low confidence"
        assert "couldn't find a sufficiently relevant FAQ" in result["answer"]

def test_empty_input(faq_engine):
    """Test empty string and whitespace input."""
    result = faq_engine.get_best_match("")
    assert result["status"] == "empty_query"
    assert result["matched_faq"] is None
    assert result["confidence_level"] == "Low confidence"

def test_low_confidence_queries(faq_engine):
    """Test queries with custom high threshold triggering low confidence fallback."""
    result = faq_engine.get_best_match("Random submarine propeller speed", threshold=0.70)
    assert result["status"] == "low_confidence"
    assert result["matched_faq"] is None
    assert result["confidence_level"] == "Low confidence"

def test_ambiguous_queries(faq_engine):
    """Test top 3 candidates and margin calculations."""
    result = faq_engine.get_best_match("Where is my package?")
    assert len(result["top_candidates"]) == 3
    assert result["top_candidates"][0]["score"] >= result["top_candidates"][1]["score"]
    expected_margin = result["top_candidates"][0]["score"] - result["top_candidates"][1]["score"]
    assert result["margin"] == pytest.approx(expected_margin, abs=1e-3)
