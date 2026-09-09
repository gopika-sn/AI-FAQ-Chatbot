import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiFallbackHandler:
    """
    Optional Gemini LLM Fallback Handler using the official Google GenAI SDK (`google-genai`).
    
    Rules:
    - Never hardcodes an API key; reads from GEMINI_API_KEY environment variable.
    - Used ONLY when FAQ retrieval returns low confidence (< 0.55 similarity).
    - Constrains Gemini output to avoid fabricating store policies or specific facts.
    - Handles errors gracefully without crashing the application.
    """
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name

    @property
    def api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    def is_configured(self) -> bool:
        """Return True if google-genai package is installed and GEMINI_API_KEY is non-empty."""
        return GENAI_AVAILABLE and bool(self.api_key)

    def generate_fallback_response(self, user_query: str) -> Optional[str]:
        """
        Generate constrained AI response for unhandled customer query.
        Returns None if Gemini is unconfigured, disabled, or encounters an API error.
        """
        if not self.is_configured():
            return None

        try:
            client = genai.Client(api_key=self.api_key)

            system_instruction = (
                "You are the fallback assistant for an e-commerce FAQ chatbot.\n"
                "The primary source of store-specific information is the FAQ knowledge base.\n"
                "You do not know private or store-specific policies unless they are explicitly provided.\n\n"
                "Never invent:\n"
                "- prices\n"
                "- refund policies\n"
                "- delivery promises\n"
                "- product availability\n"
                "- order information\n"
                "- account information\n"
                "- store policies\n\n"
                "If the user's question requires store-specific information that is not present in the FAQ knowledge base, "
                "clearly state that you do not have that information and suggest contacting customer support.\n\n"
                "For general informational questions, provide a concise helpful answer.\n\n"
                "Keep responses concise and suitable for a customer-support chatbot."
            )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                max_output_tokens=300
            )

            # Try primary model, fallback if model deprecated
            try_models = [self.model_name, "gemini-2.5-flash-lite", "gemma-4-26b-a4b-it"]
            for m in try_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=user_query,
                        config=config
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception:
                    continue

            return None
        except Exception:
            # Handle API errors gracefully without crashing the application
            return None
