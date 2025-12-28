"""
Document Chunking Strategies
=============================

Problem: Large documents don't fit in embedding models or LLM context windows.
Solution: Split into chunks while preserving semantic meaning.

Key Learning Points:
- Simple vs semantic chunking strategies
- Overlap prevents loss of context at boundaries
- Chunk size affects retrieval quality vs context
- Metadata helps track sources for citations

Common chunk sizes: 256-512 tokens (~200-400 words)
"""

from typing import List, Dict
import re


def simple_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, any]]:
    """
    Simple character-based chunking with overlap

    Pros: Fast, consistent chunk sizes
    Cons: May split mid-sentence or mid-word
    Use case: Quick prototyping, uniform content
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "start_char": start,
            "end_char": end,
            "method": "simple"
        })

        start = end - overlap  # Overlap for context

    return chunks


def sentence_chunking(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[Dict[str, any]]:
    """
    Sentence-aware chunking with overlap

    Pros: Preserves sentence boundaries, more semantic
    Cons: Variable chunk sizes
    Use case: Most production RAG systems (best balance)
    """
    # Split into sentences (simple regex - production would use spacy/nltk)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    i = 0

    while i < len(sentences):
        # Take sentences_per_chunk sentences
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk_text = ' '.join(chunk_sentences)

        chunks.append({
            "text": chunk_text,
            "sentence_start": i,
            "sentence_end": i + len(chunk_sentences),
            "method": "sentence"
        })

        # Move forward by sentences_per_chunk - overlap
        i += sentences_per_chunk - overlap_sentences

    return chunks


def paragraph_chunking(text: str, paragraphs_per_chunk: int = 2) -> List[Dict[str, any]]:
    """
    Paragraph-aware chunking (for well-structured docs)

    Pros: Preserves semantic units (paragraphs)
    Cons: Highly variable chunk sizes
    Use case: Books, articles, documentation with clear structure
    """
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    i = 0

    while i < len(paragraphs):
        chunk_paragraphs = paragraphs[i:i + paragraphs_per_chunk]
        chunk_text = '\n\n'.join(chunk_paragraphs)

        chunks.append({
            "text": chunk_text,
            "paragraph_start": i,
            "paragraph_end": i + len(chunk_paragraphs),
            "method": "paragraph"
        })

        i += paragraphs_per_chunk

    return chunks


def add_metadata(chunks: List[Dict[str, any]], source_file: str, doc_title: str = None) -> List[Dict[str, any]]:
    """
    Add metadata for source tracking and filtering

    Why: Enables citation, filtering by source/date/type
    Production systems add: date, author, category, etc.
    """
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
        chunk["source"] = source_file
        chunk["title"] = doc_title or source_file
        chunk["chunk_index"] = f"{source_file}_{i}"

    return chunks


# Example: Compare chunking strategies
if __name__ == "__main__":
    sample_doc = """
Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.
The goal is to enable computers to learn automatically without human intervention.

Deep learning is a specialized form of machine learning. It uses neural networks with multiple layers.
These networks can learn hierarchical representations of data. Deep learning has revolutionized computer vision and natural language processing.

Supervised learning requires labeled training data. The algorithm learns to map inputs to outputs.
Common applications include classification and regression. This is the most common type of machine learning in practice.

Unsupervised learning works with unlabeled data. It discovers hidden patterns and structures.
Clustering and dimensionality reduction are key techniques. This approach is useful for exploratory data analysis.

Reinforcement learning learns through trial and error. An agent interacts with an environment.
It receives rewards or penalties for actions. This approach powers game-playing AI and robotics.
"""

    print("=" * 60)
    print("DOCUMENT CHUNKING COMPARISON")
    print("=" * 60)

    # Simple chunking
    print("\n1. SIMPLE CHUNKING (char-based, 200 chars, 20 overlap)")
    print("-" * 60)
    simple_chunks = simple_chunking(sample_doc, chunk_size=200, overlap=20)
    for i, chunk in enumerate(simple_chunks[:3]):  # Show first 3
        print(f"\nChunk {i}: {len(chunk['text'])} chars")
        print(f"  {chunk['text'][:100]}...")
    print(f"\nTotal chunks: {len(simple_chunks)}")

    # Sentence chunking (RECOMMENDED for most cases)
    print("\n\n2. SENTENCE CHUNKING (3 sentences/chunk, 1 overlap)")
    print("-" * 60)
    sentence_chunks = sentence_chunking(sample_doc, sentences_per_chunk=3, overlap_sentences=1)
    for i, chunk in enumerate(sentence_chunks[:3]):
        print(f"\nChunk {i}: {len(chunk['text'])} chars")
        print(f"  {chunk['text']}")
    print(f"\nTotal chunks: {len(sentence_chunks)}")

    # Paragraph chunking
    print("\n\n3. PARAGRAPH CHUNKING (1 paragraph/chunk)")
    print("-" * 60)
    paragraph_chunks = paragraph_chunking(sample_doc, paragraphs_per_chunk=1)
    for i, chunk in enumerate(paragraph_chunks[:3]):
        print(f"\nChunk {i}: {len(chunk['text'])} chars")
        print(f"  {chunk['text'][:100]}...")
    print(f"\nTotal chunks: {len(paragraph_chunks)}")

    # Add metadata
    print("\n\n4. WITH METADATA (for source tracking)")
    print("-" * 60)
    chunks_with_metadata = add_metadata(
        sentence_chunks,
        source_file="ml_basics.txt",
        doc_title="Introduction to Machine Learning"
    )
    print(f"\nExample chunk with metadata:")
    print(chunks_with_metadata[0])

    # Production recommendations
    print("\n\n" + "=" * 60)
    print("PRODUCTION RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. START WITH: Sentence chunking (3-5 sentences, 1 overlap)
   - Good balance of semantic units and size
   - Works for most content types

2. CHUNK SIZE: Aim for 200-500 words (~256-512 tokens)
   - Small enough: Precise retrieval
   - Large enough: Sufficient context

3. OVERLAP: 10-20% of chunk size
   - Prevents loss of context at boundaries
   - Helps with questions spanning chunks

4. METADATA: Always include source, chunk_id, doc_title
   - Enables citations in answers
   - Allows filtering by source/date/category

5. FOR PRODUCTION: Use specialized libraries
   - LangChain's text splitters (semantic-aware)
   - LlamaIndex's node parsers
   - Custom splitters for domain-specific content
""")
