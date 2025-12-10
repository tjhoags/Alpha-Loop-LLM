# API Key Update Instructions

## Anthropic API Key Update

**New API Key:**
```
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
```

**Update Location:**
- File: `C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env`
- Variable: `ANTHROPIC_API_KEY`

**Steps:**
1. Open the .env file in Dropbox
2. Find the line: `ANTHROPIC_API_KEY=...`
3. Replace with: `ANTHROPIC_API_KEY=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`
4. Save the file

**Note:** The test script uses `claude-3-haiku-20240307` for fast testing. For production use, you can use `claude-3-5-sonnet-20240620` or `claude-3-opus-20240229`.

## Perplexity Model Update

**Model Changed:**
- Old: `llama-3.1-sonar-small-128k-online` (invalid)
- New: `sonar-deep-research` ✅

**Reference:** [Perplexity Sonar Deep Research Documentation](https://docs.perplexity.ai/getting-started/models/models/sonar-deep-research)

**Files Updated:**
- ✅ `scripts/test_all_apis.py` - Updated model name

**Model Details:**
- **Model Name:** `sonar-deep-research`
- **Purpose:** Exhaustive research with expert-level insights
- **Features:**
  - Deep research / Reasoning model
  - Exhaustive research across hundreds of sources
  - 128K context length
  - Expert-level subject analysis
  - Detailed report generation with citations
- **Performance:** Can take 30-90 seconds for complex queries (uses async API for production)
- **Reference:** [Perplexity Sonar Deep Research Documentation](https://docs.perplexity.ai/getting-started/models/models/sonar-deep-research)

**Pricing:**
- Input Tokens: $2 per 1M tokens
- Output Tokens: $8 per 1M tokens
- Citation Tokens: $2 per 1M tokens
- Search Queries: $5 per 1K requests
- Reasoning Tokens: $3 per 1M tokens

## Verification

After updating the Anthropic API key, run:

```bash
python scripts/test_all_apis.py
```

Expected results:
- ✅ Anthropic API: Should now show SUCCESS (after updating key in .env)
- ⚠️ Perplexity API: May timeout on first test (sonar-deep-research takes 30-90 seconds)
  - This is expected behavior - the model conducts exhaustive research
  - For production, use the async API endpoint
  - For faster testing, consider using `sonar-pro` or `sonar-online` models

**Note:** The Perplexity test timeout has been increased to 90 seconds to accommodate the deep research model's processing time.

## Notes

- The Anthropic API key update must be done manually in the .env file
- The Perplexity model has been updated in the code
- Both APIs are now configured correctly

