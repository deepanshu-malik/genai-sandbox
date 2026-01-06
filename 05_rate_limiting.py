"""
Rate Limiting with Semaphores
==============================

Concept: Control concurrent API calls to avoid hitting rate limits (RPM/TPM).
Use asyncio.Semaphore to limit how many requests run simultaneously.

Key Learning Points:
- OpenAI has rate limits: Free tier = 3 RPM, Paid tier 1 = 500 RPM
- Semaphore controls max concurrent requests (like a bouncer at a club)
- Set MAX_CONCURRENT based on your tier (free: 2, paid: 50+)
- Without rate limiting, you get 429 errors and waste time on retries
- Check limits at: https://platform.openai.com/account/rate-limits
"""

import asyncio
import time
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Semaphore limits concurrent requests
# Free tier: 3 RPM limit - keep MAX_CONCURRENT = 2 to stay safe
# Paid tier 1: 500 RPM - can use MAX_CONCURRENT = 50+
MAX_CONCURRENT = 2

async def classify_with_limit(text: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:  # Only MAX_CONCURRENT calls run at once
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify sentiment: positive, negative, or neutral."},
                {"role": "user", "content": text}
            ],
            max_tokens=10
        )
        return {
            "text": text[:30],
            "sentiment": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }

async def main():
    # Simulate 6 requests (2 batches of 3 for free tier)
    texts = [f"Sample review number {i}" for i in range(6)]
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    start = time.time()
    tasks = [classify_with_limit(text, semaphore) for text in texts]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    total_tokens = sum(r['tokens'] for r in results)
    cost = (total_tokens / 1000) * 0.00015
    
    print(f"Processed {len(results)} requests in {elapsed:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Cost: ${cost:.6f}")
    print(f"Throughput: {len(results)/elapsed:.1f} requests/sec")
    print(f"\n💡 Semaphore kept max {MAX_CONCURRENT} concurrent requests")

if __name__ == "__main__":
    asyncio.run(main())
