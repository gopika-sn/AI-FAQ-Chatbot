import os
from dotenv import load_dotenv
from google import genai

def test_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=api_key)
    
    test_list = ["gemini-3.6-flash", "gemini-2.5-flash-lite", "gemma-4-26b-a4b-it", "gemini-flash-latest"]
    
    for m in test_list:
        try:
            print(f"Testing model: {m}...")
            res = client.models.generate_content(model=m, contents="Say 'OK'")
            if res and res.text:
                print(f"  SUCCESS for {m}: {res.text.strip()}")
                return m
        except Exception as e:
            print(f"  Failed for {m}: {e}")
            
    return None

if __name__ == "__main__":
    test_models()
