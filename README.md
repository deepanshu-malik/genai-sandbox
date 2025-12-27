# GenAI Learning Journey

## What You've Built So Far

### Session 1 - LLM API Basics
- `01_first_call.py` - Your first LLM API call with cost tracking
- `02_streaming.py` - Streaming responses for better UX
- `03_production_patterns.py` - Error handling, retries, token limits
- `04_async_calls.py` - Concurrent API calls for better performance
- `05_rate_limiting.py` - Semaphore-based rate limiting for batch operations

## Setup
```bash
# Install dependencies
pip install openai python-dotenv

# Create .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Quick Start
```bash
cd ~/Learnings
source venv/bin/activate
python <script_name>.py
```

**Note:** All scripts use `python-dotenv` to load `OPENAI_API_KEY` from `.env` file.

## Key Takeaways
- gpt-4o-mini costs ~$0.000015 per simple call
- Always track tokens and costs
- Production code needs: error handling, retries, token limits
- Streaming improves UX but delays token counts
- Async calls enable concurrent processing (5x-10x faster for batch operations)

## Common Issues

### Rate Limit Error (429)
```
openai.RateLimitError: Error code: 429 - Rate limit reached for gpt-4o-mini
```

**Cause:** Exceeded requests per minute (RPM) or tokens per minute (TPM) limits.

**Solution:**
- Free tier: 3 RPM limit → Use `MAX_CONCURRENT = 2` in semaphore
- Paid tier 1: 500 RPM → Can use `MAX_CONCURRENT = 50+`
- Check your limits: https://platform.openai.com/account/rate-limits
- Add payment method to increase limits: https://platform.openai.com/account/billing

## Next Session
- Add OpenAI payment method to continue
- Phase 2: RAG - Embeddings and semantic search

## Phase 2 Preview: RAG
Once you add billing, you'll learn:
- Embeddings and semantic search
- Vector databases (Chroma, Pinecone)
- Document chunking strategies
- Building a document Q&A system
