import os
import requests
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def live_web_search(query: str) -> str:
    """Use Groq Cloud with Mistral to get live answers from the web."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-saba-24b",
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"{query}"}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        result = response.json()
        print("🔍 API raw response:", result)  # 👈 Print for debugging
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error from Groq API: {e}\n\nFull response: {response.text}"
