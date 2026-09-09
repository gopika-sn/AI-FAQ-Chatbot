import json
import os
import numpy as np

# Set HuggingFace offline mode for cached model to avoid network lookup delays
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

class FAQEngine:
    """
    Advanced Semantic FAQ Retrieval Engine.
    
    Core Pipeline Architecture:
    1. spaCy text inspection & cleaning (preserving natural sentence structure).
    2. Sentence Transformers ('all-MiniLM-L6-v2') generating 384-d L2-normalized embeddings.
    3. Precomputes and caches vector matrix for canonical questions AND variations.
    4. Computes exact Cosine Similarity between user query vector and cached embeddings.
    5. Retrieves Top 3 candidate FAQs internally.
    6. Applies configurable thresholding, ambiguity checking (margin analysis), and confidence levels.
    """
    def __init__(
        self,
        data_path="data/faqs.json",
        model_name="all-MiniLM-L6-v2",
        confidence_threshold=0.55,
        min_margin=0.04
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_margin = min_margin

        self.faqs = []
        self.nlp = None
        self.model = None
        self.faq_embeddings = None  # Precomputed L2-normalized vector matrix
        self.faq_mapping = []       # Maps vector index -> parent FAQ dict

        self._init_spacy()
        self._init_sentence_transformer()
        self.load_faqs(self.data_path)

    def _init_spacy(self):
        """Initialize spaCy for optional structural text analysis."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def preprocess_text(self, text: str) -> str:
        """
        Clean text while strictly preserving natural sentence structure for Sentence Transformers.
        Avoids aggressive stop-word removal or stemming.
        """
        if not text:
            return ""
        return " ".join(text.strip().split())

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer model with offline fallback."""
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)

    def load_faqs(self, data_path: str):
        """
        Load FAQ dataset and precompute unit-normalized vector embeddings 
        for all canonical questions AND natural paraphrased variations.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"FAQ dataset not found at path: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        sentences_to_encode = []
        self.faq_mapping = []

        for faq in self.faqs:
            # 1. Primary Question
            q_clean = self.preprocess_text(faq["question"])
            sentences_to_encode.append(q_clean)
            self.faq_mapping.append(faq)

            # 2. Paraphrased Variations
            variations = faq.get("variations", [])
            for var in variations:
                var_clean = self.preprocess_text(var)
                if var_clean:
                    sentences_to_encode.append(var_clean)
                    self.faq_mapping.append(faq)

        # Compute L2-normalized embeddings for all phrases
        self.faq_embeddings = self.model.encode(
            sentences_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def get_best_match(
        self,
        user_query: str,
        threshold: float = None,
        min_margin: float = None
    ) -> dict:
        """
        Find the top candidate FAQs for a user query using cosine similarity, top-3 ranking,
        and margin analysis.
        
        Args:
            user_query (str): Input query string.
            threshold (float, optional): Custom confidence threshold override.
            min_margin (float, optional): Custom margin threshold override.
            
        Returns:
            dict containing:
                - matched_faq (dict or None): Best matched FAQ object
                - answer (str): Direct answer string or fallback message
                - similarity_score (float): Top similarity score
                - confidence_percent (str): Formatted confidence string
                - confidence_level (str): 'High confidence', 'Medium confidence', or 'Low confidence'
                - status (str): 'success', 'low_confidence', 'ambiguous', or 'empty_query'
                - top_candidates (list): Top 3 candidate FAQs with scores
                - margin (float): Score difference between rank 1 and rank 2
        """
        if threshold is None:
            threshold = self.confidence_threshold
        if min_margin is None:
            min_margin = self.min_margin

        clean_query = self.preprocess_text(user_query)

        if not clean_query:
            return {
                "matched_faq": None,
                "answer": "Please enter a valid question.",
                "similarity_score": 0.0,
                "confidence_percent": "0.0%",
                "confidence_level": "Low confidence",
                "status": "empty_query",
                "top_candidates": [],
                "margin": 0.0
            }

        # Generate L2-normalized embedding for user query
        query_embedding = self.model.encode(
            [clean_query],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Compute exact cosine similarity scores against all indexed phrases
        raw_similarities = cosine_similarity(query_embedding, self.faq_embeddings)[0]

        # Aggregate max similarity score per unique FAQ item
        faq_max_scores = {}
        for idx, score in enumerate(raw_similarities):
            faq_obj = self.faq_mapping[idx]
            faq_id = faq_obj["id"]
            if faq_id not in faq_max_scores or score > faq_max_scores[faq_id]["score"]:
                faq_max_scores[faq_id] = {
                    "faq": faq_obj,
                    "score": float(score)
                }

        # Sort candidate FAQs by similarity score descending
        sorted_candidates = sorted(
            faq_max_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Extract Top 3 Candidates
        top_3 = sorted_candidates[:3]
        top_candidates_list = [
            {
                "id": c["faq"]["id"],
                "question": c["faq"]["question"],
                "category": c["faq"]["category"],
                "answer": c["faq"]["answer"],
                "score": round(c["score"], 4),
                "confidence_percent": f"{round(c['score'] * 100, 1)}%"
            }
            for c in top_3
        ]

        rank1 = top_3[0] if len(top_3) > 0 else None
        rank2 = top_3[1] if len(top_3) > 1 else None

        score1 = rank1["score"] if rank1 else 0.0
        score2 = rank2["score"] if rank2 else 0.0
        margin = round(score1 - score2, 4)

        confidence_pct = f"{round(score1 * 100, 1)}%"

        # Confidence Level Indicator (Requirement 14)
        if score1 >= 0.75:
            confidence_level = "High confidence"
        elif score1 >= threshold:
            confidence_level = "Medium confidence"
        else:
            confidence_level = "Low confidence"

        # Check 1: Insufficient confidence threshold (Requirement 11)
        if score1 < threshold:
            return {
                "matched_faq": None,
                "answer": "I'm sorry, I couldn't find a sufficiently relevant FAQ for your question. Please try rephrasing it or contact customer support.",
                "similarity_score": round(score1, 4),
                "confidence_percent": confidence_pct,
                "confidence_level": "Low confidence",
                "status": "low_confidence",
                "top_candidates": top_candidates_list,
                "margin": margin
            }

        # Check 2: Ambiguity check (Requirement 10)
        if rank2 and margin < min_margin and score1 < 0.75:
            return {
                "matched_faq": None,
                "answer": (
                    f"Your question seems related to multiple topics ('{rank1['faq']['question']}' or '{rank2['faq']['question']}'). "
                    "Could you please specify your question in more detail?"
                ),
                "similarity_score": round(score1, 4),
                "confidence_percent": confidence_pct,
                "confidence_level": "Medium confidence",
                "status": "ambiguous",
                "top_candidates": top_candidates_list,
                "margin": margin
            }

        # Successful confident match
        return {
            "matched_faq": rank1["faq"],
            "answer": rank1["faq"]["answer"],
            "similarity_score": round(score1, 4),
            "confidence_percent": confidence_pct,
            "confidence_level": confidence_level,
            "status": "success",
            "top_candidates": top_candidates_list,
            "margin": margin
        }
