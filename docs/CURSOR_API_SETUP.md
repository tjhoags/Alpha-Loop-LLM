# Cursor Cloud Agents API Setup

## Overview

This guide helps you manage your Cursor Cloud Agents API key named **"ALC 2"**.

The Cursor Cloud Agents API allows you to programmatically launch and manage cloud agents that work on your repositories.

**API Documentation:** https://cursor.com/docs/cloud-agent/api/endpoints

## Getting Your API Key

1. **Go to Cursor Settings**
   - Visit: https://cursor.com/settings
   - Navigate to **API Keys** section

2. **Find or Create "ALC 2" Key**
   - Look for existing key named "ALC 2"
   - Or create a new API key with name "ALC 2"

3. **Copy the API Key**
   - ⚠️ **Important:** Copy immediately - you won't see it again!
   - Store securely in password manager or encrypted file

## Verifying Your API Key

### Option 1: Use the Verification Script

```bash
python scripts/verify_cursor_api_key.py
```

This script will:
- Check for `CURSOR_API_KEY` environment variable
- Verify the key using `/v0/me` endpoint
- Test agent listing functionality
- Show your API key details

### Option 2: Manual Verification

```bash
curl --request GET \
  --url https://api.cursor.com/v0/me \
  -u YOUR_API_KEY:
```

**Expected Response:**
```json
{
  "apiKeyName": "ALC 2",
  "createdAt": "2024-01-15T10:30:00Z",
  "userEmail": "your-email@example.com"
}
```

## Setting Up Environment Variable

### Windows PowerShell
```powershell
# Temporary (current session)
$env:CURSOR_API_KEY = "your_api_key_here"

# Permanent (User-level)
[System.Environment]::SetEnvironmentVariable("CURSOR_API_KEY", "your_api_key_here", "User")
```

### Windows CMD
```cmd
set CURSOR_API_KEY=your_api_key_here
```

### Linux/Mac
```bash
export CURSOR_API_KEY=your_api_key_here

# Add to ~/.bashrc or ~/.zshrc for permanent:
echo 'export CURSOR_API_KEY=your_api_key_here' >> ~/.bashrc
```

### Add to .env File

Add to your `.env` file (or Dropbox synced env file):

```bash
CURSOR_API_KEY=your_api_key_here
```

**Note:** The `.env` file is gitignored and should never be committed.

## API Endpoints Reference

### Verify API Key
```bash
GET /v0/me
```

### List Agents
```bash
GET /v0/agents?limit=20
```

### Launch Agent
```bash
POST /v0/agents
```

### Get Agent Status
```bash
GET /v0/agents/{id}
```

### Stop Agent
```bash
POST /v0/agents/{id}/stop
```

### Delete Agent
```bash
DELETE /v0/agents/{id}
```

## Authentication

The Cursor Cloud Agents API uses **Basic Authentication**:

```python
import requests

api_key = "your_api_key_here"
response = requests.get(
    "https://api.cursor.com/v0/me",
    auth=(api_key, "")  # username=api_key, password=""
)
```

## Rate Limits

- **List Repositories:** Very strict limits
  - 1 request per user per minute
  - 30 requests per user per hour
  - Can take tens of seconds to respond

- **Other endpoints:** Standard rate limits apply

See: https://cursor.com/docs/api for full rate limit details.

## Troubleshooting

### API Key Not Working

1. **Verify key exists**
   - Check https://cursor.com/settings
   - Ensure key name is "ALC 2"

2. **Check expiration**
   - API keys may expire
   - Create new key if expired

3. **Verify permissions**
   - Ensure key has Cloud Agents API access
   - Check account subscription status

4. **Test with curl**
   ```bash
   curl -u YOUR_API_KEY: https://api.cursor.com/v0/me
   ```

### Environment Variable Not Loading

1. **Check variable name**
   - Use `CURSOR_API_KEY` or `CURSOR_CLOUD_AGENTS_API_KEY`

2. **Restart terminal**
   - Environment variables require terminal restart

3. **Check .env file**
   - Ensure `.env` file is in project root
   - Verify file is loaded by your application

### Authentication Errors

- **401 Unauthorized:** Invalid API key
- **403 Forbidden:** API key lacks permissions
- **429 Too Many Requests:** Rate limit exceeded

## Integration Example

```python
import os
import requests

# Get API key from environment
api_key = os.getenv("CURSOR_API_KEY")

if not api_key:
    raise ValueError("CURSOR_API_KEY environment variable not set")

# Verify key
response = requests.get(
    "https://api.cursor.com/v0/me",
    auth=(api_key, "")
)

if response.status_code == 200:
    print(f"✅ Connected as: {response.json()['userEmail']}")
else:
    print(f"❌ Authentication failed: {response.status_code}")
```

## Security Best Practices

1. **Never commit API keys**
   - Add to `.gitignore`
   - Use environment variables
   - Store in secure password manager

2. **Rotate keys regularly**
   - Create new keys periodically
   - Revoke old keys when not needed

3. **Use separate keys**
   - Different keys for different environments
   - "ALC 2" for development/testing

4. **Monitor usage**
   - Check API usage in Cursor Dashboard
   - Watch for unexpected activity

## Related Documentation

- [Cursor Cloud Agents API Docs](https://cursor.com/docs/cloud-agent/api/endpoints)
- [Cursor API Overview](https://cursor.com/docs/api)
- [OpenAPI Specification](https://cursor.com/docs-static/cloud-agents-openapi.yaml)

---

**Last Updated:** 2025-12-10  
**API Key Name:** ALC 2

