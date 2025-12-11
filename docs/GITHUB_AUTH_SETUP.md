# GitHub Authentication & Private Repository Setup

## Overview

The `tjhoags/alc-newco` repository is **private**. This guide covers:
1. Creating a Personal Access Token (PAT)
2. Configuring Git for authentication
3. Managing collaborator access

---

## Step 1: Generate Personal Access Token (PAT)

### Why PAT?
- Private repos require authentication for all git operations
- PATs are more secure than password authentication
- Required for: git clone, push, pull, API access

### Create PAT on GitHub

1. Go to [GitHub Token Settings](https://github.com/settings/tokens)
2. Click **Generate new token** → **Generate new token (classic)**
3. Configure:
   - **Note**: `alc-newco-dev`
   - **Expiration**: 90 days (recommended)
   - **Scopes**:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (if using GitHub Actions)
4. Click **Generate token**
5. **COPY IMMEDIATELY** - You won't see it again!

### Store PAT Securely
- ✅ Password manager
- ✅ Encrypted file in Dropbox
- ✅ Environment variable
- ❌ **NEVER** in code or committed files

---

## Step 2: Configure Git Credentials

### Windows (PowerShell)

```powershell
# Enable Windows Credential Manager
git config --global credential.helper wincred

# OR use the newer manager-core
git config --global credential.helper manager-core

# Next push/pull will prompt once:
# Username: your-github-username
# Password: YOUR_PAT_TOKEN
```

### Mac (Terminal)

```bash
# Use macOS Keychain
git config --global credential.helper osxkeychain

# Next push/pull will prompt once for credentials
```

### Linux

```bash
# Store credentials in encrypted file
git config --global credential.helper store

# Or cache for 1 hour
git config --global credential.helper 'cache --timeout=3600'
```

---

## Step 3: Test Authentication

```bash
# Test fetch
git fetch origin

# Success: Shows branch info or "Already up to date"
# Failure: "Authentication failed" - check PAT
```

---

## Step 4: Environment Variable (Optional)

For scripts that need GitHub access:

### Windows (PowerShell)
```powershell
# Temporary (current session)
$env:GITHUB_TOKEN = "ghp_your_token_here"

# Permanent
[System.Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_your_token_here", "User")
```

### Mac/Linux
```bash
# Add to ~/.bashrc or ~/.zshrc
export GITHUB_TOKEN="ghp_your_token_here"
```

---

## Collaborator Management

### Adding Collaborators

1. Go to [Repository Access Settings](https://github.com/tjhoags/alc-newco/settings/access)
2. Click **Add people**
3. Enter GitHub username or email
4. Select permission:
   - **Read**: View only
   - **Write**: Push to branches
   - **Admin**: Full control

### Collaborator PAT Setup

Each collaborator needs their own PAT:
1. Accept invitation email
2. Generate personal PAT (Step 1)
3. Configure git credentials (Step 2)
4. Clone: `git clone https://github.com/tjhoags/alc-newco.git`

---

## PAT Rotation (Every 90 Days)

### Clear Old Credentials

**Windows:**
```powershell
cmdkey /delete:git:https://github.com
```

**Mac:**
```bash
git credential-osxkeychain erase
# Type: host=github.com
# Type: protocol=https
# Press Enter twice
```

### Generate New PAT
1. Create new PAT on GitHub
2. Delete old PAT
3. Run `git fetch` to enter new credentials

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Repository not found" | Confirm collaborator access; enter PAT |
| "Authentication failed" | PAT expired; generate new one |
| "Permission denied" | Check permission level (Read vs Write) |
| "Invalid credentials" | Clear cached credentials; re-enter PAT |

---

## Security Best Practices

1. **Rotate PATs** every 90 days
2. **Minimum scopes** - only what's needed
3. **Never commit** PATs to code
4. **Use credential manager** over plaintext storage
5. **Revoke immediately** if compromised

---

*© 2025 Alpha Loop Capital, LLC*

