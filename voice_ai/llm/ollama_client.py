import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

def query_llm(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }, timeout=10)

        return response.json()["message"]["content"]

    except Exception as e:
        print("⚠️ Ollama failed:", e)
        return None