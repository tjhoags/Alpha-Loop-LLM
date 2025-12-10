# ALC-Algo GitHub Project Guide

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Repository:** PRIVATE - Alpha Loop Capital Only

---

## 🔒 Repository Configuration

### Privacy Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Visibility | **PRIVATE** | Proprietary trading system |
| Access | Tom Hogan only | Intellectual property protection |
| Forking | Disabled | No external copies |
| Wiki | Disabled | Docs in repo |

### Branch Strategy

```
main (protected)
├── develop (integration)
├── feature/* (new features)
├── release/* (version releases)
└── hotfix/* (emergency fixes)
```

---

## 📋 Recommended Branch Structure

### Main Branches

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production-ready code | Protected, requires PR |
| `develop` | Integration branch | Protected |

### Feature Branches (Recommended)

```
feature/training-infrastructure
feature/agent-improvements
feature/backtesting-engine
feature/azure-deployment
feature/paper-trading-monitor
```

### Current Work Branches

For Day 0 training setup:
```
feature/day0-training-setup     # Tonight's training work
feature/documentation-update    # Docs improvements
feature/code-optimization       # Code consolidation
```

---

## 🚀 Git Workflow

### Initial Setup

```powershell
# Navigate to project
cd C:\Users\tom\ALC-Algo

# Initialize if needed
git init

# Add remote (if not already)
git remote add origin https://github.com/AlphaLoopCapital/ALC-Algo.git

# Check status
git status
```

### Daily Workflow

```powershell
# Start of day - get latest
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-feature

# Work on code...

# Stage changes
git add .

# Commit with message
git commit -m "feat: description of change"

# Push branch
git push origin feature/my-feature

# Create PR in GitHub UI
```

### Commit Message Convention

```
type(scope): description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Testing
- chore: Maintenance

Examples:
- feat(agents): add SKILLS agent
- fix(risk): correct MoS calculation
- docs(training): update training guide
- refactor(core): consolidate agent base
```

---

## 📁 Files to Exclude (.gitignore)

```gitignore
# Secrets
config/secrets.py
master_alc_env
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/

# IDE
.idea/
.vscode/
*.swp

# Data
data/raw/*
data/processed/*
data/portfolio_history/*
!data/**/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Models
models/trained/*
!models/trained/.gitkeep

# OS
.DS_Store
Thumbs.db
```

---

## 🏷️ Release Tags

### Version Scheme

`v{major}.{minor}.{patch}`

- **Major**: Breaking changes
- **Minor**: New features
- **Patch**: Bug fixes

### Current Version

```
v0.1.0 - Day 0 (December 9, 2025)
```

### Creating a Release

```powershell
# Tag the release
git tag -a v0.1.0 -m "Day 0: Training infrastructure ready"

# Push tag
git push origin v0.1.0
```

---

## 🔐 Security Checklist

Before any commit:

```
☐ No API keys in code
☐ No passwords in code
☐ secrets.py is gitignored
☐ master_alc_env not committed
☐ No personal data exposed
☐ Trading strategies not in commit messages
```

---

## 📊 Recommended GitHub Actions (Future)

### CI Pipeline

```yaml
name: CI

on:
  push:
    branches: [ develop, main ]
  pull_request:
    branches: [ develop, main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

---

## 📝 Today's Git Tasks

### Push Day 0 Work

```powershell
# Check current status
git status

# Stage all changes
git add .

# Commit Day 0 work
git commit -m "feat(day0): complete training infrastructure

- Add 51+ agents
- Complete documentation
- Training guides ready
- ACA engine implemented
- Ready for training tonight"

# Push to remote
git push origin main

# Or push to develop first
git push origin develop
```

---

## 🌿 Branch Allocation Recommendations

| Branch | Content | Status |
|--------|---------|--------|
| `main` | Production code | Day 0 release |
| `develop` | Integration | Active development |
| `feature/training-day1` | Day 1 training code | Create tonight |
| `feature/azure-prod` | Azure production config | Pending |
| `feature/backtesting` | Backtest improvements | Pending |

---

## 📞 GitHub Support

For repository issues:
- Owner: Tom Hogan
- Email: Tom@alphaloopcapital.com

---

*GitHub Project Guide - ALC-Algo*  
*Tom Hogan | Alpha Loop Capital, LLC*  
*Repository: PRIVATE*
