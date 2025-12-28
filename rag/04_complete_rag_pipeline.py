"""
Complete RAG Pipeline
=====================

RAG = Retrieval Augmented Generation
The killer pattern that makes LLMs useful for your own data.

How it works:
1. User asks a question
2. Find relevant documents (retrieval)
3. Include them in the LLM prompt (augmentation)
4. LLM generates answer using that context (generation)
5. Cite sources for transparency

Key Learning Points:
- RAG gives LLMs access to your knowledge base
- Much cheaper than fine-tuning for most use cases
- Provides citations (LLM alone can't do this)
- Can update knowledge without retraining
- Production pattern for most GenAI apps
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from typing import List, Dict
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def chunk_document(text: str, sentences_per_chunk: int = 3, overlap_sentences: int = 1) -> List[str]:
    """Simple sentence-based chunking"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = ' '.join(chunk_sentences)
        chunks.append(chunk_text)
        i += sentences_per_chunk - overlap_sentences

    return chunks


class SimpleRAG:
    """
    A simple but production-ready RAG system

    Features:
    - Document ingestion with chunking
    - Semantic search retrieval
    - Context-aware LLM generation
    - Source citations
    """

    def __init__(self, collection_name: str = "rag_docs", persist_directory: str = "./databases/chroma"):
        """Initialize RAG system with ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Delete existing collection (for demo)
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "RAG knowledge base"}
        )

        print(f"✅ Initialized RAG system with collection '{collection_name}'")

    def ingest_documents(self, documents: List[Dict[str, str]]):
        """
        Ingest documents into the knowledge base

        Args:
            documents: List of dicts with 'title', 'content', and optional 'category'

        Process:
        1. Chunk each document
        2. Create embeddings
        3. Store in vector DB with metadata
        """
        print(f"\n📚 Ingesting {len(documents)} documents...")

        all_ids = []
        all_chunks = []
        all_embeddings = []
        all_metadata = []

        chunk_counter = 0

        for doc in documents:
            title = doc["title"]
            content = doc["content"]
            category = doc.get("category", "general")

            # Chunk the document
            chunks = chunk_document(content, sentences_per_chunk=3, overlap_sentences=1)

            print(f"  📄 {title}: {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{title.lower().replace(' ', '_')}_chunk_{i}"
                embedding = get_embedding(chunk)

                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    "title": title,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

                chunk_counter += 1

        # Batch add to collection
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadata
        )

        print(f"✅ Ingested {chunk_counter} total chunks")
        print(f"💰 Embedding cost: ~${chunk_counter * 0.00002:.6f}")

    def retrieve(self, query: str, n_results: int = 3, category_filter: str = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: User's question
            n_results: How many chunks to retrieve
            category_filter: Optional category to filter by

        Returns:
            List of relevant chunks with metadata
        """
        query_embedding = get_embedding(query)

        metadata_filter = {"category": category_filter} if category_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        retrieved_chunks = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 / (1 + dist)
            retrieved_chunks.append({
                "text": doc,
                "title": meta["title"],
                "category": meta["category"],
                "similarity": similarity
            })

        return retrieved_chunks

    def generate_answer(self, query: str, retrieved_chunks: List[Dict], model: str = "gpt-4o-mini") -> Dict:
        """
        Generate answer using retrieved context

        Args:
            query: User's question
            retrieved_chunks: Chunks from retrieval step
            model: Which OpenAI model to use

        Returns:
            Dict with answer, sources, and metadata
        """
        # Build context from retrieved chunks
        context_parts = []
        sources = set()

        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}\n")
            sources.add(chunk['title'])

        context = "\n".join(context_parts)

        # Construct prompt (critical for RAG quality!)
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information in the context above
- If the context doesn't contain enough information, say so
- Cite your sources by mentioning the document titles
- Be concise but complete

QUESTION: {query}

ANSWER:"""

        # Call LLM
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more factual responses
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return {
            "answer": answer,
            "sources": list(sources),
            "tokens_used": tokens_used,
            "chunks_retrieved": len(retrieved_chunks),
            "model": model
        }

    def query(self, question: str, n_results: int = 3, category_filter: str = None, model: str = "gpt-4o-mini") -> Dict:
        """
        Complete RAG pipeline: retrieve + generate

        This is the main method you'd expose in a production API
        """
        print(f"\n❓ Question: {question}")

        # Step 1: Retrieve relevant chunks
        print(f"🔍 Retrieving {n_results} relevant chunks...")
        retrieved_chunks = self.retrieve(question, n_results=n_results, category_filter=category_filter)

        print("\n📊 Retrieved chunks:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. [{chunk['title']}] (similarity: {chunk['similarity']:.3f})")
            print(f"     {chunk['text'][:100]}...")

        # Step 2: Generate answer with context
        print(f"\n🤖 Generating answer with {model}...")
        result = self.generate_answer(question, retrieved_chunks, model=model)

        return result


# Example: Building a company knowledge base
if __name__ == "__main__":
    print("=" * 100)
    print("COMPLETE RAG PIPELINE DEMO")
    print("=" * 100)

    # Initialize RAG system
    rag = SimpleRAG(collection_name="company_kb", persist_directory="./databases/chroma")

    # Sample documents (imagine these are your company docs, wikis, etc.)
    documents = [
        {
            "title": "Python Best Practices",
            "content": """
                Python code should follow PEP 8 style guidelines. Use meaningful variable names and add docstrings to functions.
                Type hints improve code readability and enable better IDE support. They should be used for function parameters and return values.
                For error handling, use specific exception types rather than bare except clauses. Always clean up resources with context managers.
                Write unit tests for critical functions using pytest or unittest. Aim for at least 80% code coverage.
                Use virtual environments to isolate project dependencies. Never commit secrets or API keys to version control.
            """,
            "category": "programming"
        },
        {
            "title": "FastAPI Development Guide",
            "content": """
                FastAPI is a modern Python web framework for building APIs. It provides automatic API documentation via Swagger UI.
                Use Pydantic models for request and response validation. FastAPI automatically validates data and returns clear error messages.
                For async operations, use async def instead of def. This enables concurrent request handling and better performance.
                Dependency injection in FastAPI is done through the Depends function. Use it for database connections, authentication, and shared logic.
                For production deployment, use Uvicorn with multiple workers. Enable CORS middleware if your API is accessed from browsers.
            """,
            "category": "programming"
        },
        {
            "title": "Machine Learning Fundamentals",
            "content": """
                Machine learning models learn patterns from data. Supervised learning requires labeled examples, while unsupervised learning finds patterns in unlabeled data.
                Feature engineering is crucial for model performance. Normalize numerical features and encode categorical variables properly.
                Always split data into training, validation, and test sets. Use cross-validation to ensure robust model evaluation.
                Common algorithms include linear regression, decision trees, random forests, and neural networks. Choose based on your data and problem type.
                Regularization techniques like L1 and L2 help prevent overfitting. Monitor both training and validation metrics during development.
            """,
            "category": "ai"
        },
        {
            "title": "RAG Systems Architecture",
            "content": """
                RAG combines retrieval and generation for better LLM outputs. It enables LLMs to access external knowledge bases.
                The retrieval step uses embeddings and vector databases. ChromaDB, Pinecone, and Weaviate are popular choices.
                Chunk size affects retrieval quality. Typical sizes are 256-512 tokens with 10-20% overlap between chunks.
                Prompt engineering is critical in RAG. Include clear instructions and retrieved context in a structured format.
                For production, implement caching, error handling, and monitoring. Track metrics like retrieval relevance and answer quality.
            """,
            "category": "ai"
        },
        {
            "title": "API Security Best Practices",
            "content": """
                All API endpoints should use HTTPS to encrypt data in transit. Never send sensitive data over HTTP.
                Implement authentication using JWT tokens or OAuth 2.0. Validate tokens on every request to protected endpoints.
                Use rate limiting to prevent abuse. Implement per-user and per-IP limits with exponential backoff.
                Validate and sanitize all user inputs to prevent injection attacks. Use parameterized queries for database operations.
                Log security events and monitor for suspicious activity. Implement alerting for failed authentication attempts and unusual patterns.
            """,
            "category": "security"
        }
    ]

    # Ingest documents
    rag.ingest_documents(documents)

    # Example queries
    print("\n" + "=" * 100)
    print("EXAMPLE 1: General question")
    print("=" * 100)

    result = rag.query(
        "How should I handle async operations in FastAPI?",
        n_results=3
    )

    print("\n" + "-" * 100)
    print("💡 ANSWER:")
    print("-" * 100)
    print(result["answer"])
    print("\n📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Example with category filter
    print("\n\n" + "=" * 100)
    print("EXAMPLE 2: With category filter (only 'ai' docs)")
    print("=" * 100)

    result = rag.query(
        "What is the recommended chunk size for RAG systems?",
        n_results=3,
        category_filter="ai"
    )

    print("\n" + "-" * 100)
    print("💡 ANSWER:")
    print("-" * 100)
    print(result["answer"])
    print("\n📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Example showing RAG handles "I don't know"
    print("\n\n" + "=" * 100)
    print("EXAMPLE 3: Question not in knowledge base")
    print("=" * 100)

    result = rag.query(
        "How do I deploy a React application to AWS?",
        n_results=3
    )

    print("\n" + "-" * 100)
    print("💡 ANSWER:")
    print("-" * 100)
    print(result["answer"])
    print("\n📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Key takeaways
    print("\n\n" + "=" * 100)
    print("KEY TAKEAWAYS")
    print("=" * 100)
    print("""
✅ RAG Pipeline:
   1. Ingest: Chunk docs → Embed → Store in vector DB
   2. Retrieve: User question → Find similar chunks
   3. Augment: Add chunks to LLM prompt
   4. Generate: LLM answers using context
   5. Cite: Return sources for transparency

✅ Why RAG > Fine-tuning (for most cases):
   - Cheaper: No expensive training
   - Updatable: Add/remove docs anytime
   - Transparent: Can cite sources
   - Flexible: Works with any LLM

✅ Critical Components:
   - Chunking strategy: Affects retrieval quality
   - Embedding model: Determines semantic understanding
   - Number of chunks: More context vs more noise tradeoff
   - Prompt engineering: How you present context to LLM

✅ Production Considerations:
   - Cache embeddings (expensive to regenerate)
   - Monitor retrieval relevance (are right chunks returned?)
   - Track costs (embeddings + LLM calls)
   - Implement feedback loop (improve based on bad answers)
   - Add re-ranking for better results

✅ Common Improvements:
   - Hybrid search: Combine semantic + keyword search
   - Re-ranking: Re-score chunks with better model
   - Query expansion: Rephrase query for better retrieval
   - Citation extraction: Parse exact quotes from sources
   - Streaming: Stream LLM response for better UX

🎯 Next: Build a real project with file upload and Q&A!
""")
