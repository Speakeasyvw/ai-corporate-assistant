import requests

url = "http://localhost:8000/chat"
payload = {
    "question": "What does GDPR say about personal data protection?",
    "session_id": "test-gdpr-2"
}

response = requests.post(url, json=payload)
print(response.json()["answer"])