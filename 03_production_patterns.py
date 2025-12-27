"""
Production Patterns
===================

Concept: Production-ready LLM code needs error handling, retries, and token limits.
APIs can fail, rate limits hit, and costs can spiral without proper controls.

Key Learning Points:
- Exponential backoff for retries (wait 1s, 2s, 4s, 8s...)
- Handle specific errors: RateLimitError, APIError, APIConnectionError
- Set max_tokens to control costs and prevent runaway generation
- Always wrap API calls in try-except blocks
- Log errors for debugging and monitoring
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 1. Error Handling - APIs fail, always handle it
def safe_llm_call(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=30  # Don't wait forever
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# 2. Retry Logic - Transient failures happen
def llm_call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

# 3. Token Limits - Prevent expensive mistakes
def llm_call_with_limits(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100  # Cap output length
    )
    return response.choices[0].message.content

# Test them
print("1. Safe call:", safe_llm_call("What is Python?"))
print("\n2. With retry:", llm_call_with_retry("Name 3 AWS services"))
print("\n3. Token limited:", llm_call_with_limits("Explain machine learning"))
