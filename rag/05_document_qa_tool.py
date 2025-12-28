"""
Document Q&A Tool - Production-Ready RAG CLI
===========================================

A practical command-line tool for querying your documents using RAG.

Features:
- Load documents from a directory
- Automatic chunking and indexing
- Interactive Q&A with citations
- Cost tracking
- Persistent knowledge base

Usage:
    # Build knowledge base from documents
    python 05_document_qa_tool.py build ./my_docs

    # Ask questions interactively
    python 05_document_qa_tool.py query

    # One-off question
    python 05_document_qa_tool.py query "What is the refund policy?"

    # Reset knowledge base
    python 05_document_qa_tool.py reset
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from typing import List, Dict
import re
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
PERSIST_DIR = "./databases/document_qa"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 3  # sentences per chunk
CHUNK_OVERLAP = 1  # sentence overlap


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def chunk_text(text: str, sentences_per_chunk: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk text by sentences with overlap"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = ' '.join(chunk_sentences)
        if chunk_text.strip():
            chunks.append(chunk_text)
        i += sentences_per_chunk - overlap

    return chunks


def read_document(file_path: Path) -> str:
    """Read document content (supports .txt, .md)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


class DocumentQA:
    """Production-ready document Q&A system"""

    def __init__(self, persist_directory: str = PERSIST_DIR):
        self.persist_directory = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Try to load existing collection
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Loaded existing knowledge base with {self.collection.count()} chunks")
        except:
            self.collection = None
            print("⚠️  No knowledge base found. Run 'build' first.")

    def build_knowledge_base(self, docs_directory: str):
        """Build knowledge base from documents in a directory"""
        docs_path = Path(docs_directory)

        if not docs_path.exists() or not docs_path.is_dir():
            print(f"❌ Directory not found: {docs_directory}")
            return

        # Delete existing collection
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            print("🗑️  Cleared existing knowledge base")
        except:
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"created": datetime.now().isoformat()}
        )

        # Find all text files
        supported_extensions = ['.txt', '.md']
        doc_files = [f for f in docs_path.rglob('*') if f.suffix.lower() in supported_extensions]

        if not doc_files:
            print(f"❌ No .txt or .md files found in {docs_directory}")
            return

        print(f"\n📚 Found {len(doc_files)} documents")
        print("-" * 80)

        total_chunks = 0
        all_ids = []
        all_texts = []
        all_embeddings = []
        all_metadata = []

        for doc_file in doc_files:
            print(f"📄 Processing: {doc_file.name}")

            # Read document
            content = read_document(doc_file)

            # Chunk document
            chunks = chunk_text(content)

            print(f"   → {len(chunks)} chunks created")

            # Create embeddings and metadata
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_file.stem}_chunk_{i}"
                embedding = get_embedding(chunk)

                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    "filename": doc_file.name,
                    "filepath": str(doc_file),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

                total_chunks += 1

        # Batch add to collection
        if all_ids:
            self.collection.add(
                ids=all_ids,
                documents=all_texts,
                embeddings=all_embeddings,
                metadatas=all_metadata
            )

            print("-" * 80)
            print(f"✅ Knowledge base built successfully!")
            print(f"   📊 Total chunks: {total_chunks}")
            print(f"   💰 Embedding cost: ~${total_chunks * 0.00002:.6f}")
            print(f"   💾 Saved to: {self.persist_directory}")

    def query(self, question: str, n_results: int = 3) -> Dict:
        """Query the knowledge base"""
        if not self.collection:
            return {
                "error": "No knowledge base found. Run 'build' first.",
                "answer": None
            }

        # Retrieve relevant chunks
        query_embedding = get_embedding(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return {
                "error": "No relevant documents found.",
                "answer": None
            }

        # Build context
        context_parts = []
        sources = {}

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            source_num = len(sources) + 1 if meta["filename"] not in sources else sources[meta["filename"]]
            if meta["filename"] not in sources:
                sources[meta["filename"]] = source_num

            context_parts.append(f"[Source {source_num}: {meta['filename']}]\n{doc}\n")

        context = "\n".join(context_parts)

        # Generate answer
        prompt = f"""You are a helpful assistant that answers questions based on the provided documents.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information in the context
- If the context doesn't contain the answer, say "I don't have enough information to answer that."
- Cite sources by mentioning the filenames
- Be concise and accurate

QUESTION: {question}

ANSWER:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Calculate costs
        embedding_cost = (n_results + 1) * 0.00002  # Query + retrieved chunks
        llm_cost = tokens_used * (0.00000015 + 0.0000006) / 2  # Approximate gpt-4o-mini cost

        return {
            "answer": answer,
            "sources": list(sources.keys()),
            "tokens_used": tokens_used,
            "chunks_retrieved": n_results,
            "cost": embedding_cost + llm_cost
        }

    def interactive_mode(self):
        """Interactive Q&A mode"""
        if not self.collection:
            print("❌ No knowledge base found. Run 'build' first.")
            return

        print("\n" + "=" * 80)
        print("📖 INTERACTIVE Q&A MODE")
        print("=" * 80)
        print(f"Knowledge base: {self.collection.count()} chunks")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 80 + "\n")

        total_cost = 0.0

        while True:
            try:
                question = input("❓ Your question: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print(f"\n💰 Total session cost: ${total_cost:.6f}")
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                print()
                result = self.query(question)

                if "error" in result:
                    print(f"❌ {result['error']}")
                    continue

                print("💡 ANSWER:")
                print("-" * 80)
                print(result["answer"])
                print("-" * 80)
                print(f"📚 Sources: {', '.join(result['sources'])}")
                print(f"💰 Cost: ${result['cost']:.6f} | Tokens: {result['tokens_used']}")
                print()

                total_cost += result['cost']

            except KeyboardInterrupt:
                print(f"\n\n💰 Total session cost: ${total_cost:.6f}")
                print("👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def reset(self):
        """Reset the knowledge base"""
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            print("✅ Knowledge base reset successfully")
        except:
            print("⚠️  No knowledge base to reset")


def print_usage():
    """Print usage instructions"""
    print("""
Document Q&A Tool - Usage
=========================

Commands:
  build <directory>     Build knowledge base from documents in directory
  query [question]      Interactive Q&A mode (or one-off question)
  reset                 Reset knowledge base
  help                  Show this help message

Examples:
  python 05_document_qa_tool.py build ./docs
  python 05_document_qa_tool.py query
  python 05_document_qa_tool.py query "What is the pricing?"
  python 05_document_qa_tool.py reset

Supported file formats: .txt, .md
""")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()
    qa = DocumentQA()

    if command == "build":
        if len(sys.argv) < 3:
            print("❌ Usage: python 05_document_qa_tool.py build <directory>")
            return

        docs_dir = sys.argv[2]
        qa.build_knowledge_base(docs_dir)

    elif command == "query":
        if len(sys.argv) > 2:
            # One-off query
            question = " ".join(sys.argv[2:])
            result = qa.query(question)

            if "error" in result:
                print(f"❌ {result['error']}")
                return

            print("\n💡 ANSWER:")
            print("-" * 80)
            print(result["answer"])
            print("-" * 80)
            print(f"📚 Sources: {', '.join(result['sources'])}")
            print(f"💰 Cost: ${result['cost']:.6f} | Tokens: {result['tokens_used']}\n")
        else:
            # Interactive mode
            qa.interactive_mode()

    elif command == "reset":
        qa.reset()

    elif command == "help":
        print_usage()

    else:
        print(f"❌ Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
