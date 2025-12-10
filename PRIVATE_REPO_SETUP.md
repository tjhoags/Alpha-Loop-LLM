# Private Repository Setup Guide

## STATUS: REPOSITORY IS NOW PRIVATE

The `tjhoags/alpha-loop-llm` repository has been made private. This guide explains how to configure access.

---

## Step 1: Personal Access Token (PAT)

### Why You Need a PAT
- Private repos require authentication for all git operations
- PATs replace password authentication
- More secure than using your GitHub password

### Generate PAT on GitHub

1. Go to https://github.com/settings/tokens
2. Click **Generate new token** -> **Generate new token (classic)**
3. Settings:
   - **Note**: `alpha-loop-llm-access` (descriptive name)
   - **Expiration**: 90 days (or "No expiration" if preferred)
   - **Select scopes**:
     - [x] `repo` (Full control of private repositories)
     - [x] `workflow` (if using GitHub Actions)
4. Click **Generate token**
5. **COPY IMMEDIATELY** - You won't see it again!

### Store Your PAT Safely
Save it in a secure location:
- Dropbox: `ALC Tech Agents/credentials/github_pat.txt`
- Password manager
- **NEVER in code** - Never commit PATs to git!

---

## Step 2: Configure Git to Use PAT

### Option A: Store in Git Credential Manager (Recommended)

**Windows:**
```powershell
# Enable credential manager
git config --global credential.helper manager

# Next time you push/pull, enter:
# Username: your-github-username
# Password: YOUR_PAT (paste the token, not your password)
```

**Mac:**
```bash
# Use macOS keychain
git config --global credential.helper osxkeychain

# Next time you push/pull, enter:
# Username: your-github-username
# Password: YOUR_PAT
```

### Option B: Store in Git Config (Less Secure)
```bash
# Store credentials for this repository
git config credential.helper store
# Then push/pull once and enter credentials
```

---

## Step 3: Test Authentication

```bash
# Try to fetch from the private repo
git fetch origin

# If successful, you'll see:
# "Already up to date" or branch info

# If failed, you'll see:
# "Authentication failed" - check your PAT
```

---

## Step 4: Add Collaborators to Private Repo

### Adding Chris (or others)

1. Go to https://github.com/tjhoags/alpha-loop-llm/settings/access
2. Click **Add people**
3. Enter `chris@alphaloopcapital.com` (or their GitHub username)
4. Select permission level:
   - **Read**: Can view code only
   - **Write**: Can push to branches (recommended for Chris)
   - **Admin**: Full control (Tom only)
5. Click **Add to repository**

Chris will receive an email invitation to accept.

---

## Step 5: Update Remote URLs (if needed)

If you cloned when public, URLs should still work. But if issues arise:

```bash
# Check current remote
git remote -v

# Should show:
# origin  git@github.com:tjhoags/alpha-loop-llm.git (...)
# or
# origin  https://github.com/tjhoags/alpha-loop-llm.git (...)

# If using HTTPS, it will prompt for PAT when pushing
```

---

## Cursor IDE Integration

Cursor uses your system git credentials, so once configured above, it will work automatically.

### If Cursor Prompts for Credentials
1. Click the Git icon in the sidebar
2. If prompted, enter GitHub username and PAT
3. Check "Remember credentials"

---

## Providing PAT to AI Assistants (Cursor/Claude)

### When/Why
If you want AI assistants in Cursor to perform git operations on your behalf (rare), they may need access to your PAT.

### How to Provide Safely
1. **DO NOT** paste your PAT directly in chat
2. Instead, set it as an environment variable:

```powershell
# Windows - temporary for session
$env:GITHUB_PAT = "your_pat_here"

# Windows - permanent
[System.Environment]::SetEnvironmentVariable("GITHUB_PAT", "your_pat_here", "User")
```

```bash
# Mac/Linux - temporary for session
export GITHUB_PAT="your_pat_here"

# Mac/Linux - permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export GITHUB_PAT="your_pat_here"' >> ~/.zshrc
```

3. AI can then use `$env:GITHUB_PAT` without seeing the actual token

### Security Notes
- AI assistants should NOT need your PAT for normal operation
- Only provide if specifically needed for automation
- Rotate PAT regularly (every 90 days recommended)

---

## Revoking/Rotating PATs

### To Revoke a PAT
1. Go to https://github.com/settings/tokens
2. Find the token
3. Click **Delete**

### To Rotate (Replace)
1. Generate new PAT (Step 1 above)
2. Update stored credentials:
   ```bash
   # Clear cached credentials
   # Windows:
   cmdkey /delete:git:https://github.com
   
   # Mac:
   git credential-osxkeychain erase
   # Then type: host=github.com, protocol=https, [Enter], [Enter]
   ```
3. Next push/pull will prompt for new credentials

---

## Troubleshooting

### "Authentication failed" Error
1. Check PAT hasn't expired
2. Check PAT has `repo` scope
3. Try clearing cached credentials and re-entering

### "Repository not found" Error
1. Confirm you're a collaborator on the repo
2. Check repo URL is correct
3. Confirm repo is actually private (not deleted)

### "Permission denied" Error
1. Check your permission level (Write vs Read)
2. Confirm you're pushing to an allowed branch
3. For Chris: only push to `chris/development` branch

---

## Summary Checklist

- [x] Repository set to private on GitHub
- [ ] PAT generated with `repo` scope
- [ ] PAT stored securely (not in code!)
- [ ] Git credential helper configured
- [ ] Authentication tested with `git fetch`
- [ ] Collaborators added (Chris)
- [ ] Team members have their own PATs

