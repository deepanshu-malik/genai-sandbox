"""
Embeddings Basics
=================

Concept: Embeddings convert text into vectors (arrays of numbers) that capture semantic meaning.
Similar texts have similar vectors. This enables semantic search.

Key Learning Points:
- OpenAI's text-embedding-3-small: 1536 dimensions, $0.02 per 1M tokens
- Embeddings capture meaning, not just keywords
- Use cosine similarity to find similar texts
- Foundation for RAG, semantic search, clustering, recommendations
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate similarity between two vectors (0 to 1)"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: Find similar documents
documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning uses algorithms to learn from data",
    "Deep learning is a subset of machine learning",
]

query = "What is ML?"

print("Embedding documents...\n")
doc_embeddings = [get_embedding(doc) for doc in documents]
query_embedding = get_embedding(query)

print(f"Query: '{query}'\n")
print("Similarity scores:")

results = []
for i, doc in enumerate(documents):
    similarity = cosine_similarity(query_embedding, doc_embeddings[i])
    results.append((doc, similarity))

# Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

for doc, score in results:
    print(f"  {score:.3f} - {doc}")

print(f"\n💡 Most relevant: '{results[0][0]}'")
print(f"\nEmbedding dimensions: {len(query_embedding)}")
print(f"Cost: ~$0.00002 for {len(documents)+1} embeddings")
