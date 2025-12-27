"""
First LLM API Call
==================

Concept: Make your first call to OpenAI's API and track costs.
Understanding token usage and pricing is critical for production apps.

Key Learning Points:
- OpenAI client setup with API key from environment
- Basic chat completion with system + user messages
- Token counting: prompt_tokens + completion_tokens = total_tokens
- Cost calculation: gpt-4o-mini costs ~$0.15 per 1M tokens
- Always track tokens and costs in production
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Your first LLM call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2? Answer in one sentence."}]
)

# Extract the response
answer = response.choices[0].message.content
tokens_in = response.usage.prompt_tokens
tokens_out = response.usage.completion_tokens
cost = (tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000

print(f"Answer: {answer}")
print(f"\nTokens: {tokens_in} in, {tokens_out} out")
print(f"Cost: ${cost:.6f}")
