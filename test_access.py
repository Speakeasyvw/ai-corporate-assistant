import requests

BASE_URL = "http://localhost:8000"

# Test 1 — sin API key, debe dar 401
r = requests.post(f"{BASE_URL}/chat", json={"question": "What is GDPR?"})
print("Sin key:", r.status_code, r.json())

# Test 2 — key válida con acceso permitido
r = requests.post(f"{BASE_URL}/chat",
    headers={"X-API-Key": "legal-key-003"},
    json={"question": "What is GDPR?", "document_filter": "compliance"})
print("Legal team + compliance:", r.status_code, r.json()["answer"][:80])

# Test 3 — key válida pero categoría no permitida, debe dar 403
r = requests.post(f"{BASE_URL}/chat",
    headers={"X-API-Key": "legal-key-003"},
    json={"question": "What was NEC revenue?", "document_filter": "financial"})
print("Legal team + financial:", r.status_code, r.json())

# Test 4 — admin puede todo
r = requests.post(f"{BASE_URL}/chat",
    headers={"X-API-Key": "admin-key-001"},
    json={"question": "What was NEC revenue?", "document_filter": "financial"})
print("Admin + financial:", r.status_code, r.json()["answer"][:80])