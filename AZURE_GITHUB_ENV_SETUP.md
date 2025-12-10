================================================================================
AZURE-GITHUB INTEGRATION ENVIRONMENT SETUP
================================================================================
Author: Tom Hogan
Date: December 2025

This guide explains what environment variables need to be added to your .env
file to enable Azure services to access GitHub for logging, CI/CD, and other
integrations.

================================================================================
WHY AZURE NEEDS GITHUB ACCESS
================================================================================

Azure services may need GitHub access for:
1. CI/CD Pipelines: GitHub Actions integration
2. Logging: Sending logs to GitHub (issues, commits, releases)
3. Deployment: Automated deployments from GitHub
4. Monitoring: Creating GitHub issues for alerts
5. Backup: Syncing data to GitHub repositories

================================================================================
REQUIRED ENVIRONMENT VARIABLES
================================================================================

Add these to your .env file located at:
C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env

================================================================================
GITHUB TOKEN (REQUIRED)
================================================================================

GITHUB_TOKEN=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

This is your Personal Access Token (PAT) that allows Azure services to
authenticate with GitHub.

USAGE:
- Azure Functions/Apps can use this to access GitHub API
- CI/CD pipelines can use this for deployments
- Logging services can create issues or commits

SECURITY:
- Never commit this token to Git
- Keep it secure in the .env file
- Rotate periodically (every 90 days recommended)

================================================================================
GITHUB REPOSITORY INFORMATION (OPTIONAL BUT RECOMMENDED)
================================================================================

GITHUB_REPO=tjhoags/alpha-loop-llm
GITHUB_OWNER=tjhoags

These help identify which repository Azure should interact with.

================================================================================
AZURE-SPECIFIC GITHUB INTEGRATION VARIABLES
================================================================================

If using Azure DevOps or Azure Functions with GitHub integration:

AZURE_GITHUB_CONNECTION_NAME=alpha-loop-github
AZURE_GITHUB_WEBHOOK_SECRET=<your-webhook-secret-if-needed>

These are typically configured in Azure Portal, but can be referenced
in environment variables if needed.

================================================================================
EXAMPLE .ENV FILE ADDITIONS
================================================================================

Add these lines to your .env file:

# GitHub Integration for Azure Logging/CI/CD
GITHUB_TOKEN=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
GITHUB_REPO=tjhoags/alpha-loop-llm
GITHUB_OWNER=tjhoags

================================================================================
VERIFYING THE SETUP
================================================================================

After adding these variables, verify they're loaded:

1. Test in Python:
   python -c "from src.config.settings import get_settings; s = get_settings(); print(f'GitHub Token: {s.github_token[:10]}...'); print(f'GitHub Repo: {s.github_repo}')"

2. Check environment variables are loaded:
   python scripts/test_api_connections.py

3. If using Azure Functions/Apps:
   - Restart the Azure service
   - Check Azure Portal > Configuration > Application settings
   - Verify variables are available

================================================================================
AZURE PORTAL CONFIGURATION
================================================================================

If deploying to Azure App Service or Azure Functions:

1. Go to Azure Portal
2. Navigate to your App Service/Function App
3. Go to Configuration > Application settings
4. Add these as Application Settings:
   - GITHUB_TOKEN: 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
   - GITHUB_REPO: tjhoags/alpha-loop-llm
   - GITHUB_OWNER: tjhoags

5. Mark GITHUB_TOKEN as "Hidden" (secure setting)
6. Save and restart the app

================================================================================
GITHUB ACTIONS INTEGRATION
================================================================================

If using GitHub Actions for CI/CD, the token can be used as a secret:

1. Go to GitHub repository > Settings > Secrets and variables > Actions
2. Add new repository secret:
   - Name: GITHUB_TOKEN
   - Value: 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

3. Use in workflow files:
   ```yaml
   env:
     GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
   ```

================================================================================
AZURE LOGGING TO GITHUB
================================================================================

If you want Azure to send logs to GitHub (e.g., create issues for errors):

The application code can use the GITHUB_TOKEN to:
1. Create GitHub issues for critical errors
2. Commit log summaries to a logs branch
3. Create releases with deployment logs
4. Update status checks

Example usage in code:
```python
from src.config.settings import get_settings
import requests

settings = get_settings()
headers = {
    "Authorization": f"Bearer {settings.github_token}",
    "Accept": "application/vnd.github.v3+json"
}

# Create issue for critical error
response = requests.post(
    f"https://api.github.com/repos/{settings.github_repo}/issues",
    headers=headers,
    json={"title": "Critical Error", "body": "Error details..."}
)
```

================================================================================
SECURITY BEST PRACTICES
================================================================================

1. NEVER commit the .env file to Git
   - Ensure .env is in .gitignore
   - Use .env.example for template (without real values)

2. Rotate tokens regularly
   - Every 90 days recommended
   - Revoke old tokens immediately

3. Use least privilege
   - Only grant necessary scopes to the PAT
   - Limit repository access if possible

4. Monitor token usage
   - Check GitHub > Settings > Developer settings > Tokens
   - Review access logs regularly

5. Use Azure Key Vault (Recommended for production)
   - Store secrets in Azure Key Vault
   - Reference from Azure services
   - More secure than environment variables

================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: Azure can't access GitHub
---------------------------------
- Verify GITHUB_TOKEN is set correctly
- Check token hasn't expired
- Verify token has necessary scopes (repo, workflow, etc.)
- Check Azure service has access to environment variables

ISSUE: GitHub Actions failing
------------------------------
- Verify secret is set in GitHub repository
- Check token permissions
- Review workflow logs for specific errors

ISSUE: Logging to GitHub not working
-------------------------------------
- Verify GITHUB_TOKEN is loaded in application
- Check token has write permissions
- Verify repository name is correct
- Review API rate limits

================================================================================
QUICK REFERENCE
================================================================================

Add to .env file:
-----------------
GITHUB_TOKEN=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
GITHUB_REPO=tjhoags/alpha-loop-llm
GITHUB_OWNER=tjhoags

Verify in code:
---------------
from src.config.settings import get_settings
settings = get_settings()
print(settings.github_token)  # Should show token (first 10 chars)
print(settings.github_repo)  # Should show: tjhoags/alpha-loop-llm

================================================================================
END OF GUIDE
================================================================================

