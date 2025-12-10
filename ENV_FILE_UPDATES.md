================================================================================
ENVIRONMENT FILE (.env) UPDATES REQUIRED
================================================================================
Author: Tom Hogan
Date: December 2025

This document lists all the environment variables that need to be added or
updated in your .env file for Azure-GitHub integration and logging.

Location: C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env

================================================================================
REQUIRED ADDITIONS
================================================================================

Add these lines to your .env file:

# ============================================================================
# GITHUB INTEGRATION (For Azure Logging/CI/CD)
# ============================================================================
GITHUB_TOKEN=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
GITHUB_REPO=tjhoags/alpha-loop-llm
GITHUB_OWNER=tjhoags

# ============================================================================
# PERPLEXITY API (Updated Key)
# ============================================================================
PERPLEXITY_API_KEY=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

================================================================================
WHAT THESE DO
================================================================================

GITHUB_TOKEN:
-------------
- Allows Azure services to authenticate with GitHub
- Used for CI/CD pipelines, logging, and deployments
- Required for Azure Functions/Apps to access GitHub API

GITHUB_REPO:
------------
- Identifies which repository Azure should interact with
- Format: owner/repository-name
- Default: tjhoags/alpha-loop-llm

GITHUB_OWNER:
-------------
- GitHub username/organization name
- Used for API calls and repository identification
- Default: tjhoags

PERPLEXITY_API_KEY:
-------------------
- Updated Perplexity API key for AI research
- Used by research agents for web-connected queries

================================================================================
VERIFICATION
================================================================================

After updating your .env file, verify the settings are loaded:

1. Test in Python:
   python -c "from src.config.settings import get_settings; s = get_settings(); print('GitHub Token:', s.github_token[:10] + '...' if s.github_token else 'NOT SET'); print('GitHub Repo:', s.github_repo); print('Perplexity Key:', s.perplexity_api_key[:10] + '...' if s.perplexity_api_key else 'NOT SET')"

2. Test API connections:
   python scripts/test_api_connections.py

3. Test Perplexity specifically:
   python scripts/test_all_apis.py

================================================================================
SECURITY NOTES
================================================================================

- Never commit the .env file to Git
- Keep the file secure in Dropbox
- Rotate tokens every 90 days
- If token is compromised, revoke immediately on GitHub

================================================================================
END OF UPDATES
================================================================================

