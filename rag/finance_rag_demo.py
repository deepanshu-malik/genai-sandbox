"""
Finance Domain RAG Demo
=======================
Complete RAG pipeline demonstration using finance domain documents:
- Loan Origination System (LOS)
- Loan Management System (LMS)
- Credit Reports
- Underwriting Guidelines
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from pathlib import Path
import re
from typing import List, Dict

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


class FinanceRAG:
    """RAG system for Finance domain documents"""

    def __init__(self, collection_name: str = "finance_kb", persist_directory: str = "./databases/finance_rag"):
        """Initialize RAG system with ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Delete existing collection (for demo)
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Finance domain knowledge base"}
        )

        print(f"✅ Initialized Finance RAG system")

    def ingest_documents(self, doc_directory: str):
        """Ingest finance documents from directory"""
        doc_path = Path(doc_directory)
        doc_files = list(doc_path.glob("*.txt"))

        print(f"\n📚 Ingesting {len(doc_files)} finance documents...")

        all_ids = []
        all_chunks = []
        all_embeddings = []
        all_metadata = []

        chunk_counter = 0

        for doc_file in doc_files:
            title = doc_file.stem.replace('_', ' ').title()
            content = doc_file.read_text()

            # Chunk the document
            chunks = chunk_document(content, sentences_per_chunk=4, overlap_sentences=1)

            print(f"  📄 {title}: {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_file.stem}_chunk_{i}"
                embedding = get_embedding(chunk)

                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    "title": title,
                    "category": "finance",
                    "document_type": doc_file.stem,
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

    def query(self, question: str, n_results: int = 3) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        print(f"\n{'='*100}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*100}")

        # Step 1: Retrieve relevant chunks
        print(f"\n🔍 Retrieving top {n_results} relevant chunks...")

        query_embedding = get_embedding(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        retrieved_chunks = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 / (1 + dist)
            retrieved_chunks.append({
                "text": doc,
                "title": meta["title"],
                "document_type": meta["document_type"],
                "similarity": similarity
            })

        print("\n📊 RETRIEVED CHUNKS:")
        print("-" * 100)
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"{i}. [{chunk['title']}] (Similarity: {chunk['similarity']:.3f})")
            print(f"   {chunk['text'][:150]}...")
            print()

        # Step 2: Generate answer with context
        print("🤖 GENERATING ANSWER...\n")

        # Build context from retrieved chunks
        context_parts = []
        sources = set()

        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}\n")
            sources.add(chunk['title'])

        context = "\n".join(context_parts)

        # Construct prompt
        prompt = f"""You are a finance domain expert assistant specializing in lending systems, credit analysis, and loan processing.

CONTEXT FROM KNOWLEDGE BASE:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information in the context above
- If the context doesn't contain enough information, say so clearly
- Cite your sources by mentioning the document titles
- Be precise with financial terms and regulatory requirements
- Use industry terminology appropriately

QUESTION: {question}

ANSWER:"""

        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature for factual finance answers
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return {
            "answer": answer,
            "sources": list(sources),
            "tokens_used": tokens_used,
            "chunks_retrieved": len(retrieved_chunks)
        }


if __name__ == "__main__":
    print("=" * 100)
    print("FINANCE DOMAIN RAG SYSTEM DEMO")
    print("=" * 100)

    # Initialize RAG system
    rag = FinanceRAG()

    # Ingest finance documents
    rag.ingest_documents("./sample_docs/finance")

    # Example queries relevant to finance domain

    # Query 1: LOS functionality
    result = rag.query(
        "How does a Loan Origination System handle income verification?",
        n_results=3
    )

    print("=" * 100)
    print("💡 ANSWER:")
    print("=" * 100)
    print(result["answer"])
    print()
    print("📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Query 2: Credit report analysis
    result = rag.query(
        "What is debt-to-credit ratio and how does it affect credit scores?",
        n_results=3
    )

    print("\n" + "=" * 100)
    print("💡 ANSWER:")
    print("=" * 100)
    print(result["answer"])
    print()
    print("📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Query 3: LMS functionality
    result = rag.query(
        "How does a Loan Management System handle delinquent loans?",
        n_results=3
    )

    print("\n" + "=" * 100)
    print("💡 ANSWER:")
    print("=" * 100)
    print(result["answer"])
    print()
    print("📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Query 4: Underwriting
    result = rag.query(
        "What are the standard debt-to-income ratio requirements for conventional mortgages?",
        n_results=3
    )

    print("\n" + "=" * 100)
    print("💡 ANSWER:")
    print("=" * 100)
    print(result["answer"])
    print()
    print("📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    # Query 5: Cross-document question
    result = rag.query(
        "What's the relationship between credit scores from credit reports and loan approval in underwriting?",
        n_results=4
    )

    print("\n" + "=" * 100)
    print("💡 ANSWER:")
    print("=" * 100)
    print(result["answer"])
    print()
    print("📚 Sources:", ", ".join(result["sources"]))
    print(f"📊 Tokens used: {result['tokens_used']} | Chunks retrieved: {result['chunks_retrieved']}")

    print("\n" + "=" * 100)
    print("🎯 KEY OBSERVATIONS")
    print("=" * 100)
    print("""
✅ Finance-Specific RAG Features Demonstrated:
   - Domain-specific document ingestion (LOS, LMS, Credit Reports, Underwriting)
   - Precise retrieval of regulatory and technical information
   - Low temperature (0.2) for factual financial answers
   - Cross-document synthesis (last query used multiple sources)

✅ RAG Benefits for Finance Domain:
   - Provides accurate, source-backed answers to compliance questions
   - Can query across multiple systems (LOS, LMS, etc.)
   - Citations ensure auditability for regulatory requirements
   - Easy to update as regulations change (just update documents)

✅ Production Considerations for Finance RAG:
   - Add metadata filtering by regulation type, effective date
   - Implement version control for regulatory documents
   - Include confidence scores and uncertainty handling
   - Add audit logging for all queries (compliance tracking)
   - Integrate with document approval workflows
""")
