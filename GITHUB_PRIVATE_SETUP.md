================================================================================
GITHUB PRIVATE REPOSITORY SETUP - COMPLETE GUIDE
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC
Date: December 2025
Status: Repository is now PRIVATE

================================================================================
REPOSITORY STATUS
================================================================================

Repository: tjhoags/alpha-loop-llm
Status: PRIVATE (as of December 2025)
Access: Requires Personal Access Token (PAT)

================================================================================
PERSONAL ACCESS TOKEN (PAT)
================================================================================

PAT PROVIDED: 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

This PAT has been configured with:
- Full repository access (repo scope)
- Read and write permissions
- Access to private repositories

SECURITY NOTES:
- Never commit this PAT to code
- Never share in emails or chat
- Store securely in password manager
- Rotate every 90 days

================================================================================
QUICK SETUP FOR TOM
================================================================================

1. Configure Git Credential Helper:
   
   Windows PowerShell:
   git config --global credential.helper wincred
   
   Mac Terminal:
   git config --global credential.helper osxkeychain

2. Test Connection:
   git fetch origin
   
   When prompted:
   Username: tjhoags (or your GitHub username)
   Password: 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
   
   Git will store this securely.

3. Verify:
   git pull origin main
   git push origin [your-branch]
   
   Should work without prompting after first use.

================================================================================
QUICK SETUP FOR CHRIS
================================================================================

1. Configure Git Credential Helper:
   
   Windows PowerShell:
   git config --global credential.helper wincred

2. Clone Repository:
   cd "C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents"
   git clone https://github.com/tjhoags/alpha-loop-llm.git
   
   When prompted:
   Username: [Chris's GitHub username]
   Password: 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

3. Verify Access:
   cd alpha-loop-llm
   git pull origin main
   
   Should work without prompting after first use.

================================================================================
STORING PAT IN ENVIRONMENT FILE
================================================================================

Add PAT to .env file for script access:

Location: C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env

Add this line:
GITHUB_PAT=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

This allows Python scripts to access GitHub API if needed.

IMPORTANT: .env file is NOT committed to Git (in .gitignore)

================================================================================
GIT CONFIGURATION
================================================================================

WINDOWS SETUP:
--------------
1. Open PowerShell
2. Configure credential helper:
   git config --global credential.helper wincred
3. Test with: git fetch origin
4. Enter PAT when prompted
5. Windows Credential Manager stores it securely

MAC SETUP:
----------
1. Open Terminal
2. Configure credential helper:
   git config --global credential.helper osxkeychain
3. Test with: git fetch origin
4. Enter PAT when prompted
5. macOS Keychain stores it securely

VERIFY CONFIGURATION:
---------------------
git config --global credential.helper
# Should show: wincred (Windows) or osxkeychain (Mac)

================================================================================
TROUBLESHOOTING
================================================================================

PROBLEM: "Authentication failed"
SOLUTION:
- Verify PAT is correct: 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
- Check no extra spaces when copying
- Ensure credential helper is configured
- Try: git config --global credential.helper wincred (Windows)
- Try: git config --global credential.helper osxkeychain (Mac)

PROBLEM: "Repository not found"
SOLUTION:
- Verify repository URL: https://github.com/tjhoags/alpha-loop-llm.git
- Check you have access (Tom must add you as collaborator)
- Verify PAT has "repo" scope

PROBLEM: "Permission denied"
SOLUTION:
- Verify you're using PAT, not GitHub password
- Check PAT has write permissions (repo scope)
- Verify you have write access to repository
- Contact Tom to verify access level

PROBLEM: Git keeps asking for password
SOLUTION:
- Clear stored credentials:
  Windows: Control Panel → Credential Manager → Remove GitHub entries
  Mac: Keychain Access → Remove GitHub entries
- Reconfigure credential helper
- Re-enter PAT

================================================================================
UPDATING EXISTING CLONES
================================================================================

If you cloned before repository became private:

1. Update remote URL:
   git remote set-url origin https://github.com/tjhoags/alpha-loop-llm.git

2. Test connection:
   git fetch origin
   
   When prompted, enter PAT

3. Verify:
   git pull origin main

================================================================================
CI/CD AND AUTOMATION
================================================================================

If using GitHub Actions:

1. Add PAT as Secret:
   - Go to repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: GITHUB_TOKEN
   - Value: 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
   - Click "Add secret"

2. Use in workflows:
   ```yaml
   - name: Checkout code
     uses: actions/checkout@v3
     with:
       token: ${{ secrets.GITHUB_TOKEN }}
   ```

================================================================================
BRANCH PROTECTION
================================================================================

Recommended settings for private repository:

1. Go to repository → Settings → Branches
2. Add rule for "main" branch:
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators

This ensures:
- No direct commits to main
- All changes reviewed
- Code quality maintained

================================================================================
COLLABORATOR ACCESS
================================================================================

Adding team members:

1. Go to repository → Settings → Collaborators
2. Click "Add people"
3. Enter GitHub username or email
4. Select access level:
   - Read: Can view and clone
   - Write: Can push and create branches
   - Admin: Full access
5. Send invitation

Team members will receive email invitation and must:
1. Accept invitation
2. Configure PAT (use same PAT: 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)
3. Clone repository

================================================================================
SECURITY BEST PRACTICES
================================================================================

1. NEVER commit PAT to code
   - PAT is in .env file (gitignored)
   - Never hardcode in scripts
   - Never commit .env file

2. Rotate PAT regularly
   - Every 90 days recommended
   - Generate new PAT
   - Update credential helper
   - Revoke old PAT

3. Monitor access
   - Check GitHub audit log
   - Review active tokens
   - Remove unused access

4. Use branch protection
   - Protect main branch
   - Require reviews
   - Require status checks

5. Limit PAT scope
   - Only grant necessary permissions
   - Use fine-grained tokens when possible
   - Revoke unused tokens

================================================================================
QUICK REFERENCE
================================================================================

PAT: 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
Repository: tjhoags/alpha-loop-llm
Status: PRIVATE

Configure Git:
Windows: git config --global credential.helper wincred
Mac: git config --global credential.helper osxkeychain

Test Connection:
git fetch origin
(Enter PAT when prompted)

Store in .env:
GITHUB_PAT=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

================================================================================
END OF GUIDE
================================================================================

