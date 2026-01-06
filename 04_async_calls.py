"""
Async API Calls
===============

Concept: Use asyncio to make concurrent LLM API calls instead of sequential ones.
Dramatically reduces total processing time for batch operations.

Key Learning Points:
- AsyncOpenAI client for async operations
- asyncio.gather() runs multiple calls concurrently
- 5-10x faster for batch processing (5 calls in ~2s vs ~10s sequential)
- Essential for production systems handling multiple requests
- Use when: processing batches, handling concurrent users, parallel tool calls
"""

import asyncio
import time
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Async single call
async def classify_text(text: str) -> dict:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify sentiment as positive, negative, or neutral. Reply with one word only."},
            {"role": "user", "content": text}
        ],
        max_tokens=10
    )
    return {
        "text": text,
        "sentiment": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    }

# Process multiple texts concurrently
async def classify_batch(texts: list[str]) -> list[dict]:
    tasks = [classify_text(text) for text in texts]
    return await asyncio.gather(*tasks)

# Compare sync vs async performance
async def main():
    texts = [
        "I love this product!",
        "Terrible experience, very disappointed.",
        "It's okay, nothing special.",
        "Best purchase ever!",
        "Waste of money.",
    ]
    
    # Async (concurrent)
    start = time.time()
    results = await classify_batch(texts)
    async_time = time.time() - start
    
    print("Results:")
    for r in results:
        print(f"  '{r['text'][:30]}...' → {r['sentiment']} ({r['tokens']} tokens)")
    
    total_tokens = sum(r['tokens'] for r in results)
    cost = (total_tokens / 1000) * 0.00015
    
    print(f"\nAsync time: {async_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Cost: ${cost:.6f}")
    print(f"\n💡 Sequential would take ~{len(texts)}x longer")

if __name__ == "__main__":
    asyncio.run(main())
