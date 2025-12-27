"""
Context Window Management
=========================

Concept: Every LLM has a token limit (context window). For gpt-4o-mini: 128k tokens.
When processing long documents, you need strategies to stay within limits.

Key Learning Points:
- Strategy 1 (Truncate): Simple but loses tail information
- Strategy 2 (Chunk): Process pieces separately, keeps all info but loses coherence
- Strategy 3 (Map-Reduce): Best of both - summarize chunks then combine
- Token estimation: ~4 characters = 1 token (rough approximation)
- Use map-reduce for production document summarization
"""

import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulate a long document
LONG_DOC = """
Machine learning is a subset of artificial intelligence that focuses on building systems 
that learn from data. """ * 100  # Repeat to make it long

def count_tokens_estimate(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4

async def strategy_1_truncate(text: str, max_tokens: int = 1000):
    """Strategy 1: Simple truncation - loses tail information"""
    truncated = text[:max_tokens * 4]  # Rough char limit
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this text in 2 sentences."},
            {"role": "user", "content": truncated}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

async def strategy_2_chunk(text: str, chunk_size: int = 1000):
    """Strategy 2: Process chunks separately"""
    chunks = [text[i:i+chunk_size*4] for i in range(0, len(text), chunk_size*4)]
    
    tasks = []
    for i, chunk in enumerate(chunks[:3]):  # Limit to 3 for demo
        task = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Summarize chunk {i+1}:"},
                {"role": "user", "content": chunk}
            ],
            max_tokens=50
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in results]

async def strategy_3_map_reduce(text: str, chunk_size: int = 1000):
    """Strategy 3: Summarize chunks, then combine summaries"""
    # Map: Summarize each chunk
    chunk_summaries = await strategy_2_chunk(text, chunk_size)
    
    # Reduce: Combine summaries
    combined = "\n".join(chunk_summaries)
    final = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Combine these summaries into one coherent summary:"},
            {"role": "user", "content": combined}
        ],
        max_tokens=100
    )
    return final.choices[0].message.content

async def main():
    print(f"Document length: ~{count_tokens_estimate(LONG_DOC)} tokens\n")
    
    print("Strategy 1 - Truncate:")
    result1 = await strategy_1_truncate(LONG_DOC)
    print(f"{result1}\n")
    
    await asyncio.sleep(20)  # Wait for rate limit reset
    
    print("Strategy 2 - Chunk (first 3 chunks):")
    result2 = await strategy_2_chunk(LONG_DOC)
    for i, summary in enumerate(result2):
        print(f"  Chunk {i+1}: {summary}")
    print()
    
    await asyncio.sleep(20)  # Wait for rate limit reset
    
    print("Strategy 3 - Map-Reduce:")
    result3 = await strategy_3_map_reduce(LONG_DOC)
    print(f"{result3}")

if __name__ == "__main__":
    asyncio.run(main())
