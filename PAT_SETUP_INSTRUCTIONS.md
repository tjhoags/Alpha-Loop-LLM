================================================================================
PERSONAL ACCESS TOKEN (PAT) SETUP FOR PRIVATE REPOSITORY
================================================================================
Author: Tom Hogan
Date: December 2025

This guide explains how to set up a Personal Access Token (PAT) for GitHub
when the repository is made private, and what other functionality needs to
be configured.

================================================================================
WHY PAT IS NEEDED
================================================================================

When a GitHub repository is private, you cannot use your regular GitHub
password for authentication. Instead, you must use a Personal Access Token (PAT).

The PAT acts as a password replacement for:
- Git operations (clone, push, pull, fetch)
- GitHub API access
- CI/CD workflows
- Third-party integrations

================================================================================
STEP 1: CREATE A PERSONAL ACCESS TOKEN
================================================================================

1. Log in to GitHub.com with your account

2. Go to Settings:
   - Click your profile picture (top right)
   - Click "Settings"

3. Navigate to Developer settings:
   - Scroll down in left sidebar
   - Click "Developer settings"

4. Go to Personal access tokens:
   - Click "Tokens (classic)"
   - Or use "Fine-grained tokens" (newer, more secure)

5. Generate new token:
   - Click "Generate new token"
   - Choose "Generate new token (classic)" for maximum compatibility

6. Configure token:
   - Note: Give it a descriptive name
     Example: "ALC Development - Alpha Loop LLM"
   
   - Expiration: Set based on your needs
     - 30 days: For temporary access
     - 90 days: For regular development
     - No expiration: For long-term use (less secure)
   
   - Select scopes (permissions):
     REQUIRED:
     - repo (Full control of private repositories)
       - This includes: repo:status, repo_deployment, public_repo, repo:invite, security_events
     
     RECOMMENDED:
     - workflow (Update GitHub Action workflows)
       - Needed if using GitHub Actions for CI/CD
     
     OPTIONAL:
     - read:org (Read org membership)
       - If repository is in an organization
     - admin:repo_hook (Full control of repository hooks)
       - If you need to manage webhooks

7. Generate token:
   - Click "Generate token" at bottom
   - IMPORTANT: Copy the token immediately
   - You will NOT be able to see it again after closing the page
   - Save it securely (password manager recommended)

================================================================================
STEP 2: STORE YOUR PAT SECURELY
================================================================================

DO NOT:
- Commit the PAT to Git
- Share it in emails or chat
- Store it in plain text files
- Include it in code

DO:
- Store in a password manager (1Password, LastPass, Bitwarden)
- Use environment variables (see Step 3)
- Use Git credential helper (see Step 4)
- Rotate tokens regularly (every 90 days recommended)

================================================================================
STEP 3: USE PAT WITH GIT OPERATIONS
================================================================================

METHOD 1: Git Credential Helper (Recommended)
----------------------------------------------

This stores your PAT securely so you don't have to enter it every time.

WINDOWS:
--------
1. Open PowerShell

2. Configure credential helper:
   git config --global credential.helper wincred

3. On first use, Git will prompt:
   Username: [your GitHub username]
   Password: [paste your PAT here, NOT your GitHub password]

4. Windows Credential Manager will store it securely

MACBOOK:
--------
1. Open Terminal

2. Configure credential helper:
   git config --global credential.helper osxkeychain

3. On first use, Git will prompt:
   Username: [your GitHub username]
   Password: [paste your PAT here, NOT your GitHub password]

4. macOS Keychain will store it securely

METHOD 2: Environment Variable
-------------------------------

WINDOWS (PowerShell):
----------------------
$env:GITHUB_TOKEN = "your_pat_here"
git clone https://github.com/tjhoags/alpha-loop-llm.git

WINDOWS (Command Prompt):
--------------------------
set GITHUB_TOKEN=your_pat_here
git clone https://github.com/tjhoags/alpha-loop-llm.git

MACBOOK:
--------
export GITHUB_TOKEN=your_pat_here
git clone https://github.com/tjhoags/alpha-loop-llm.git

METHOD 3: Include in URL (Not Recommended)
--------------------------------------------
git clone https://YOUR_USERNAME:YOUR_PAT@github.com/tjhoags/alpha-loop-llm.git

This method is less secure as the PAT may appear in logs.

================================================================================
STEP 4: VERIFY PAT WORKS
================================================================================

1. Test clone (if not already cloned):
   git clone https://github.com/tjhoags/alpha-loop-llm.git

2. Test push:
   git push origin main

3. Test pull:
   git pull origin main

If all operations work without password prompts, your PAT is configured correctly.

================================================================================
STEP 5: UPDATE EXISTING CLONES
================================================================================

If you already have the repository cloned before it became private:

1. Update remote URL:
   git remote set-url origin https://github.com/tjhoags/alpha-loop-llm.git

2. Test connection:
   git fetch origin

3. If prompted, enter:
   Username: [your GitHub username]
   Password: [your PAT]

================================================================================
STEP 6: CI/CD AND AUTOMATION
================================================================================

If you're using GitHub Actions or other CI/CD:

1. Add PAT as GitHub Secret:
   - Go to repository > Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: GITHUB_TOKEN (or custom name)
   - Value: [paste your PAT]
   - Click "Add secret"

2. Use in workflows:
   ```yaml
   - name: Checkout code
     uses: actions/checkout@v3
     with:
       token: ${{ secrets.GITHUB_TOKEN }}
   ```

3. For third-party integrations:
   - Check integration documentation
   - Usually requires adding PAT in integration settings

================================================================================
STEP 7: ROTATE TOKENS REGULARLY
================================================================================

Best practice: Rotate tokens every 90 days

1. Generate new token (follow Step 1)

2. Update credential helper:
   - Delete old token from credential manager
   - Use new token on next Git operation
   - Git will prompt and store new token

3. Revoke old token:
   - Go to GitHub > Settings > Developer settings > Tokens
   - Find old token
   - Click "Revoke"

4. Update any CI/CD secrets with new token

================================================================================
STEP 8: TROUBLESHOOTING
================================================================================

ISSUE: "Authentication failed" or "Invalid credentials"
-------------------------------------------------------
- Verify PAT is correct (no extra spaces)
- Check token hasn't expired
- Ensure "repo" scope is selected
- Try regenerating token

ISSUE: "Permission denied" when pushing
---------------------------------------
- Verify you have write access to repository
- Check token has "repo" scope (not just "public_repo")
- Contact repository owner to verify access

ISSUE: "Repository not found"
------------------------------
- Verify repository URL is correct
- Check repository is private and you have access
- Verify PAT has "repo" scope

ISSUE: Credential helper not working
------------------------------------
WINDOWS:
- Try: git config --global credential.helper manager-core
- Or: git config --global credential.helper wincred

MACBOOK:
- Try: git config --global credential.helper osxkeychain
- Check Keychain Access app for stored credentials

================================================================================
STEP 9: ADDITIONAL FUNCTIONALITY TO CONFIGURE
================================================================================

After making repository private, verify these work:

1. GITHUB ACTIONS (CI/CD):
   - Update workflow files if needed
   - Add PAT as secret (see Step 6)
   - Test workflows run successfully

2. DEPENDENCIES AND PACKAGES:
   - If using private packages, update authentication
   - Update package.json, requirements.txt, etc. if needed

3. WEBHOOKS:
   - Update webhook URLs if changed
   - Verify webhook secrets are configured

4. INTEGRATIONS:
   - Slack notifications
   - Issue trackers
   - Deployment tools
   - All may need PAT or updated authentication

5. DOCUMENTATION:
   - Update README.md with new clone instructions
   - Update setup guides with PAT instructions
   - Document any access requirements

6. BRANCH PROTECTION:
   - Configure branch protection rules
   - Set up required reviewers
   - Configure status checks

7. COLLABORATORS:
   - Add Chris and other team members
   - Ensure they have appropriate access levels
   - Share PAT setup instructions with them

================================================================================
STEP 10: SECURITY CHECKLIST
================================================================================

Before making repository private:

[ ] All collaborators have GitHub accounts
[ ] All collaborators know how to create PATs
[ ] PAT creation instructions documented
[ ] CI/CD workflows updated (if applicable)
[ ] Webhooks configured (if applicable)
[ ] Branch protection rules set (if desired)
[ ] Access levels assigned appropriately
[ ] Old tokens revoked (if rotating)
[ ] Documentation updated with PAT instructions

After making repository private:

[ ] Test clone works with PAT
[ ] Test push works with PAT
[ ] Test pull works with PAT
[ ] CI/CD workflows still work
[ ] All integrations still work
[ ] Team members can access repository

================================================================================
QUICK REFERENCE
================================================================================

CREATE PAT:
1. GitHub.com > Settings > Developer settings > Tokens (classic)
2. Generate new token (classic)
3. Select "repo" scope
4. Copy token immediately

CONFIGURE GIT:
Windows: git config --global credential.helper wincred
Mac: git config --global credential.helper osxkeychain

USE PAT:
When Git prompts for password, paste PAT (not GitHub password)

VERIFY:
git fetch origin
(Should work without prompting if configured correctly)

================================================================================
END OF PAT SETUP GUIDE
================================================================================

