# Understanding Local, Worktree, and Cloud Environments

## Quick Reference Table

| Type | Location | Use When | Syncs To |
|------|----------|----------|----------|
| **Local** | Your machine only | Quick experiments, temporary changes | Nowhere (manual backup needed) |
| **Worktree** | `.cursor/worktrees/` | Active development, isolated branches | GitHub via git push |
| **Cloud** | Azure/GitHub | Production, shared data | Accessible from anywhere |

---

## 1. LOCAL

### What It Is
Files that exist only on your computer. Not synced anywhere automatically.

### Location
- Windows: `C:\Users\tom\...` (any folder outside Dropbox/OneDrive)
- Mac: `~/Desktop/...` or any local folder

### When to Use Local
- Testing code changes before committing
- Temporary scripts you'll delete
- Sensitive files you don't want synced
- Large data files (to avoid slow cloud sync)

### Example Use Case
```
# Creating a temporary test file
cd C:\temp
python test_my_idea.py
# Delete when done - it won't affect anything
```

### Risk
- NO BACKUP unless you manually copy
- If your machine dies, files are LOST
- Other team members can't see your work

---

## 2. WORKTREE

### What It Is
Git worktrees are separate working directories for the same repository.
Each worktree can have a different branch checked out.

### Location
- Windows: `C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\xxx`
- Mac: `~/.cursor/worktrees/Alpha-Loop-LLM-1/xxx`

### Why We Use Worktrees
1. **Isolated Development**: Work on multiple branches simultaneously
2. **Safe Experiments**: Try changes without affecting main code
3. **Parallel Work**: Tom can work on `main`, Chris on `chris/development`

### Creating a New Worktree
```bash
# From the main repository
cd ~/Alpha-Loop-LLM

# Create a new worktree for a feature
git worktree add ../Alpha-Loop-LLM-1-feature feature-branch

# Now you have two working directories:
# - ~/Alpha-Loop-LLM (main branch)
# - ~/Alpha-Loop-LLM-1-feature (feature branch)
```

### When to Use Worktree
- Active development work
- Testing features before merging
- Working on your own branch (e.g., chris/development)
- Code that you WILL commit to GitHub

### Workflow
```bash
# 1. Navigate to your worktree
cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1

# 2. Make changes to files

# 3. Stage and commit
git add .
git commit -m "Your message"

# 4. Push to GitHub (your branch only!)
git push origin chris/development
```

---

## 3. CLOUD

### What It Is
Services and storage that exist on the internet, accessible from anywhere.

### Our Cloud Services
| Service | Purpose | Access |
|---------|---------|--------|
| **GitHub** | Code repository | github.com/tjhoags/alpha-loop-llm (PRIVATE) |
| **Azure SQL** | Database | alc-sql-server.database.windows.net |
| **Dropbox** | Shared files, API keys | Alphaloopcapital Dropbox |

### GitHub (Code)
- Stores all our code versions
- Enables collaboration
- Maintains history of changes

```bash
# Pull latest code from cloud
git pull origin main

# Push your changes to cloud
git push origin chris/development
```

### Azure SQL (Database)
- Stores market data
- Stores model results
- Accessible from any machine

```python
# Automatically connects via settings
from src.config.settings import get_settings
settings = get_settings()
print(settings.sqlalchemy_url)  # Shows Azure connection
```

### Dropbox (Shared Files)
- API keys (API - Dec 2025.env)
- Trained models
- Shared datasets

```
Path: C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents
Contents:
  - API - Dec 2025.env (credentials)
  - models/ (trained ML models)
  - data/ (shared datasets)
  - backups/ (code backups)
```

---

## Decision Flowchart

```
Is this code you want to keep?
|
+-- NO --> Use LOCAL (temporary files, experiments)
|
+-- YES --> Is this code ready to share?
            |
            +-- NO --> Use WORKTREE (active development)
            |
            +-- YES --> Push to CLOUD (GitHub)
```

---

## Practical Examples

### Example 1: Quick Test
"I want to test a formula real quick"
```
# Use LOCAL - it's temporary
cd C:\temp
echo "print(1+1)" > test.py
python test.py
del test.py
```

### Example 2: New Feature Development
"I'm building a new report for investors"
```
# Use WORKTREE - active development
cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
git checkout chris/development

# Make your changes...
# Then commit when ready
git add .
git commit -m "Chris: Added investor report feature"
git push origin chris/development
```

### Example 3: Checking Production Data
"I need to see what's in the database"
```
# Use CLOUD - Azure SQL
python scripts/test_db_connection.py
# Or query directly
python -c "from src.database.connection import get_engine ..."
```

### Example 4: Sharing Models
"I trained a great model, team needs it"
```
# Copy to CLOUD (Dropbox)
Copy-Item models\my_model.pkl "C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\models"
# Now accessible to all team members via Dropbox
```

---

## Summary

| Task | Environment | Why |
|------|-------------|-----|
| Quick experiment | Local | Fast, disposable |
| Feature development | Worktree | Git-tracked, isolated |
| Code collaboration | GitHub (Cloud) | Team access, versioned |
| Data storage | Azure SQL (Cloud) | Centralized, reliable |
| File sharing | Dropbox (Cloud) | Easy access, synced |

---

## Chris's Typical Day

1. **Morning**: Pull latest from GitHub (Cloud -> Worktree)
   ```bash
   cd worktree-folder
   git pull origin main
   ```

2. **During Day**: Work in your worktree
   ```bash
   # Make changes, test locally
   python scripts/test_api_connections.py
   ```

3. **End of Day**: Push to GitHub (Worktree -> Cloud)
   ```bash
   git add .
   git commit -m "Chris: Today's work"
   git push origin chris/development
   ```

4. **Sharing**: Copy important files to Dropbox
   ```
   # Manual copy for team access
   ```

---

## Common Mistakes to Avoid

1. **DON'T** work directly in the main repository
   - Use worktrees instead

2. **DON'T** store API keys in git
   - Keep them in Dropbox, copy to local .env

3. **DON'T** push directly to main branch
   - Always use your branch (chris/development)

4. **DON'T** forget to commit before switching worktrees
   - Changes can be lost

