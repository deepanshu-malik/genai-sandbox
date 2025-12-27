"""
Streaming Responses
===================

Concept: Stream LLM responses token-by-token for better user experience.
Instead of waiting for the full response, show output as it's generated.

Key Learning Points:
- Set stream=True in API call
- Iterate over response chunks to get tokens as they arrive
- Improves perceived latency (users see progress immediately)
- Token counts come at the end (in final chunk with finish_reason)
- Use for: chatbots, long-form content generation, interactive apps
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Streaming response (like ChatGPT UI)
print("Streaming response: ", end="", flush=True)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\nNote: Streaming gives better UX but you don't get token counts until the end")
