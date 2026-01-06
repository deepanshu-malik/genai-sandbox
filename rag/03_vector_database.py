"""
Vector Databases with ChromaDB
===============================

Problem: In-memory embedding storage doesn't scale and isn't persistent.
Solution: Vector databases provide efficient storage, search, and persistence.

Key Learning Points:
- Chroma stores embeddings + metadata + documents together
- Automatic persistence (save/load your index)
- Fast similarity search using HNSW algorithm
- Metadata filtering: "Find similar docs from source X published after Y"
- No external APIs needed - runs locally

Production alternatives: Pinecone, Weaviate, Qdrant, FAISS
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def initialize_chroma(persist_directory: str = "./databases/chroma") -> chromadb.Client:
    """
    Initialize ChromaDB with persistence

    persist_directory: Where to save the database on disk
    Returns: ChromaDB client
    """
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client


def create_or_get_collection(chroma_client: chromadb.Client, name: str = "documents"):
    """
    Create or retrieve a collection (like a table in SQL)

    Collections store: documents, embeddings, metadata
    """
    # Delete if exists (for demo purposes)
    try:
        chroma_client.delete_collection(name=name)
        print(f"🗑️  Deleted existing collection '{name}'")
    except:
        pass

    # Create new collection
    collection = chroma_client.create_collection(
        name=name,
        metadata={"description": "RAG learning collection"}
    )
    print(f"✅ Created collection '{name}'")
    return collection


def add_documents(collection, documents: List[Dict]):
    """
    Add documents with embeddings to collection

    Each document needs:
    - id: unique identifier
    - document: the actual text
    - embedding: vector representation (optional - Chroma can auto-generate)
    - metadata: additional fields for filtering
    """
    print(f"\n📝 Adding {len(documents)} documents...")

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    for doc in documents:
        ids.append(doc["id"])
        texts.append(doc["text"])
        embeddings.append(get_embedding(doc["text"]))
        metadatas.append(doc["metadata"])

    # Batch add to collection
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"✅ Added {len(documents)} documents to collection")
    print(f"💰 Cost: ~${len(documents) * 0.00002:.6f}")


def search_similar(collection, query: str, n_results: int = 3, metadata_filter: Dict = None):
    """
    Search for similar documents

    Args:
        query: The search query
        n_results: How many results to return
        metadata_filter: Filter by metadata (e.g., {"source": "python_docs"})

    Returns: Similar documents with similarity scores
    """
    print(f"\n🔍 Searching for: '{query}'")
    if metadata_filter:
        print(f"   With filter: {metadata_filter}")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=metadata_filter,  # Optional metadata filter
        include=["documents", "metadatas", "distances"]
    )

    return results


def print_results(results):
    """Pretty print search results"""
    print("\n📊 RESULTS:")
    print("-" * 80)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # Convert distance to similarity score (0-1, higher is better)
        similarity = 1 / (1 + dist)

        print(f"\n{i+1}. Similarity: {similarity:.3f} | Source: {meta['source']} | Category: {meta['category']}")
        print(f"   {doc[:200]}..." if len(doc) > 200 else f"   {doc}")


# Example: Building a knowledge base
if __name__ == "__main__":
    print("=" * 80)
    print("VECTOR DATABASE DEMO")
    print("=" * 80)

    # Initialize ChromaDB
    chroma_client = initialize_chroma(persist_directory="./databases/chroma")

    # Create collection
    collection = create_or_get_collection(chroma_client, name="tech_docs")

    # Sample documents with metadata
    sample_docs = [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
            "metadata": {"source": "python_docs", "category": "programming", "year": 2024}
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. It's used in recommendation systems, image recognition, and natural language processing.",
            "metadata": {"source": "ml_guide", "category": "ai", "year": 2024}
        },
        {
            "id": "doc3",
            "text": "React is a JavaScript library for building user interfaces. It uses a component-based architecture and a virtual DOM for efficient updates.",
            "metadata": {"source": "react_docs", "category": "programming", "year": 2024}
        },
        {
            "id": "doc4",
            "text": "Vector databases store high-dimensional vectors (embeddings) and enable fast similarity search. They're essential for RAG systems, semantic search, and recommendation engines.",
            "metadata": {"source": "db_guide", "category": "ai", "year": 2024}
        },
        {
            "id": "doc5",
            "text": "FastAPI is a modern Python web framework for building APIs. It's fast, easy to use, and includes automatic API documentation with Swagger UI.",
            "metadata": {"source": "python_docs", "category": "programming", "year": 2024}
        },
        {
            "id": "doc6",
            "text": "Transformers revolutionized NLP by using attention mechanisms to process sequential data. They power models like GPT, BERT, and Claude.",
            "metadata": {"source": "ml_guide", "category": "ai", "year": 2023}
        },
    ]

    # Add documents
    add_documents(collection, sample_docs)

    # Test 1: Basic semantic search
    print("\n" + "=" * 80)
    print("TEST 1: Basic Semantic Search")
    print("=" * 80)
    results = search_similar(collection, "How do I build web APIs with Python?", n_results=3)
    print_results(results)

    # Test 2: Search with metadata filter (only AI category)
    print("\n" + "=" * 80)
    print("TEST 2: Search with Metadata Filter (category='ai')")
    print("=" * 80)
    results = search_similar(
        collection,
        "What are neural networks?",
        n_results=3,
        metadata_filter={"category": "ai"}
    )
    print_results(results)

    # Test 3: Filter by multiple metadata fields (using $and operator)
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Metadata Filters (source='python_docs' AND category='programming')")
    print("=" * 80)
    results = search_similar(
        collection,
        "web development",
        n_results=3,
        metadata_filter={"$and": [{"source": "python_docs"}, {"category": "programming"}]}
    )
    print_results(results)

    # Show collection stats
    print("\n" + "=" * 80)
    print("COLLECTION STATS")
    print("=" * 80)
    print(f"Total documents: {collection.count()}")
    print(f"Persisted to: ./databases/chroma")

    # Demonstrate persistence
    print("\n" + "=" * 80)
    print("PERSISTENCE DEMO")
    print("=" * 80)
    print("✅ All data is automatically saved to disk at ./databases/chroma")
    print("📌 You can restart your Python script and reload the same collection")
    print("📌 Try this:")
    print("   1. Run this script")
    print("   2. Stop it")
    print("   3. Comment out 'add_documents()' line")
    print("   4. Run again - data will still be there!")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
✅ Vector DB vs In-Memory:
   - In-memory: Fast but not persistent, limited scale
   - Vector DB: Persistent, scalable, production-ready

✅ Metadata Filtering:
   - Combine semantic search with traditional filters
   - Example: "Find similar docs from source X published after 2023"
   - Critical for real applications

✅ ChromaDB Benefits:
   - Local (no external APIs, no costs beyond embeddings)
   - Auto-persistence (survives restarts)
   - Simple API (great for learning and prototyping)
   - Can scale to production with server mode

✅ Production Considerations:
   - Chroma: Great for prototypes, small-medium apps
   - Pinecone: Managed service, scales to billions
   - Weaviate: Open-source, feature-rich
   - FAISS: Facebook's library, fastest for pure vector search

🎯 Next Step: Combine this with LLM to build complete RAG pipeline!
""")
