# Batch Document Analyzer

Analyze multiple text documents concurrently using LLMs. Extracts sentiment, key topics, and summaries.

## Features
- Concurrent processing with rate limiting
- Sentiment analysis (positive/negative/neutral)
- Key topics extraction
- One-sentence summaries
- Cost tracking and performance metrics
- JSON report generation

## Usage
```bash
cd project_batch_analyzer
source ../venv/bin/activate
python analyzer.py
```

## Output
- Console: Pretty-printed analysis results
- File: `analysis_report.json` with structured data

## Configuration
- `MAX_CONCURRENT = 2` - Adjust based on your OpenAI rate limits
- Free tier: Keep at 2
- Paid tier: Increase to 50+

## Real-World Use Cases
- Customer feedback analysis
- Support ticket categorization
- Content moderation
- Document classification
- Review aggregation
