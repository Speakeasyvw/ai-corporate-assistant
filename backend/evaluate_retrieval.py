import json
from backend.retrieval import search_docs

with open("backend/evaluation_dataset.json") as f:
    dataset = json.load(f)

correct = 0

for item in dataset:
    question = item["question"]
    expected_doc = item["expected_document"]

    results = search_docs(question)

    top_doc = results[0].metadata["source_file"]

    if top_doc == expected_doc:
        correct += 1
        print("✔", question)
    else:
        print("✘", question, "->", top_doc)

accuracy = correct / len(dataset)

print("\nRetrieval accuracy:", accuracy)