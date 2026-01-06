"""
Batch Document Analyzer
========================

Project: Analyze multiple documents concurrently and generate a report.
Real-world use case: Customer feedback analysis, support ticket processing.

Features:
- Concurrent document processing with rate limiting
- Sentiment analysis, key topics extraction, summarization
- Cost tracking and performance metrics
- JSON report generation
"""

import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_CONCURRENT = 2  # Free tier limit

async def analyze_document(file_path: Path, semaphore: asyncio.Semaphore) -> dict:
    """Analyze a single document: sentiment, topics, summary"""
    async with semaphore:
        content = file_path.read_text()
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Analyze this text and respond in JSON format:
{
  "sentiment": "positive/negative/neutral",
  "key_topics": ["topic1", "topic2"],
  "summary": "one sentence summary"
}"""},
                {"role": "user", "content": content}
            ],
            max_tokens=150
        )
        
        result = json.loads(response.choices[0].message.content)
        result["file"] = file_path.name
        result["tokens"] = response.usage.total_tokens
        return result

async def main():
    docs_dir = Path(__file__).parent / "sample_docs"
    files = list(docs_dir.glob("*.txt"))
    
    print(f"Analyzing {len(files)} documents...\n")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    start = time.time()
    
    tasks = [analyze_document(f, semaphore) for f in files]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    total_tokens = sum(r["tokens"] for r in results)
    cost = (total_tokens / 1000) * 0.00015
    
    # Generate report
    report = {
        "summary": {
            "total_documents": len(results),
            "processing_time": f"{elapsed:.2f}s",
            "total_tokens": total_tokens,
            "total_cost": f"${cost:.6f}"
        },
        "documents": results
    }
    
    # Save report
    report_path = Path(__file__).parent / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    # Print results
    print("Analysis Complete!\n")
    for doc in results:
        print(f"📄 {doc['file']}")
        print(f"   Sentiment: {doc['sentiment']}")
        print(f"   Topics: {', '.join(doc['key_topics'])}")
        print(f"   Summary: {doc['summary']}")
        print(f"   Tokens: {doc['tokens']}\n")
    
    print(f"Total: {len(results)} docs in {elapsed:.2f}s | {total_tokens} tokens | ${cost:.6f}")
    print(f"\n✅ Report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
