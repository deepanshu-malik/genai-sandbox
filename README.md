# GenAI Learning Journey

## What You've Built So Far

### Session 1 - LLM API Basics
- `01_first_call.py` - Your first LLM API call with cost tracking
- `02_streaming.py` - Streaming responses for better UX
- `03_production_patterns.py` - Error handling, retries, token limits
- `04_async_calls.py` - Concurrent API calls for better performance
- `05_rate_limiting.py` - Semaphore-based rate limiting for batch operations

## Project Structure

```
~/Learnings/
├── 01_first_call.py          # Session 1: LLM API basics
├── 02_streaming.py
├── 03_production_patterns.py
├── 04_async_calls.py
├── 05_rate_limiting.py
├── rag/                       # Session 2: RAG systems
│   ├── 01_embeddings.py
│   ├── 02_chunking_strategies.py
│   ├── 03_vector_database.py
│   ├── 04_complete_rag_pipeline.py
│   ├── 05_document_qa_tool.py
│   ├── finance_rag_demo.py
│   ├── embeddings_demo.ipynb
│   ├── finance_rag_notebook.ipynb
│   ├── databases/             # Vector databases (gitignored)
│   │   ├── chroma/
│   │   ├── document_qa/
│   │   └── finance_rag/
│   ├── sample_docs/           # Sample documents for testing
│   │   └── finance/
│   └── README.md             # Detailed RAG documentation
├── MEMORY.md                  # Session history and learning notes
├── .env                       # API keys (gitignored)
├── .gitignore
└── requirements.txt
```

**Note:** All databases are stored in `rag/databases/` and excluded from git via `.gitignore` to keep the repository clean.

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

### Session 2 - RAG (Retrieval Augmented Generation) ✅ COMPLETED
Complete RAG learning path from fundamentals to production:
- `rag/01_embeddings.py` - Text embeddings and cosine similarity
- `rag/02_chunking_strategies.py` - Document chunking strategies (simple, sentence, paragraph)
- `rag/03_vector_database.py` - ChromaDB for persistent embedding storage with metadata filtering
- `rag/04_complete_rag_pipeline.py` - Full RAG: retrieval + augmented prompts + generation
- `rag/05_document_qa_tool.py` - **Production-ready CLI tool for document Q&A!**

**🎯 What you can do now:**
- Build RAG systems from scratch
- Choose optimal chunking strategies
- Work with vector databases (ChromaDB, Pinecone, etc.)
- Deploy production Q&A systems with citations
- Optimize for cost and performance

**See `rag/README.md` for complete documentation!**

## Next Sessions
Ideas for continuing your GenAI journey:
- **LangChain**: Framework for building LLM applications
- **LlamaIndex**: Specialized framework for RAG and data connectors
- **Agents**: Autonomous systems that use tools and make decisions
- **Fine-tuning**: Customize models for your specific use case
- **Production deployment**: FastAPI + streaming + monitoring
