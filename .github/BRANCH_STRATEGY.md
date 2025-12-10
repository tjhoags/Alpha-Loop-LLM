# ALC-Algo Branch Strategy

**Author:** Tom Hogan | Alpha Loop Capital, LLC

## Branch Structure

```
main                    # Production-ready code only
├── develop             # Integration branch for features
│   ├── feature/*       # New features
│   ├── bugfix/*        # Bug fixes
│   └── hotfix/*        # Urgent production fixes
├── release/*           # Release preparation branches
└── infra/*             # Infrastructure changes
```

## Branch Descriptions

### `main` (Protected)
- **Purpose:** Production-ready, stable code
- **Merge From:** `release/*` branches only, or `hotfix/*` for emergencies
- **CI/CD:** Automatically deploys to production
- **Rules:**
  - Requires pull request
  - Requires 1 approval (Tom Hogan or designated reviewer)
  - Requires all CI checks to pass
  - No force push

### `develop` (Integration)
- **Purpose:** Integration branch for ongoing development
- **Merge From:** `feature/*`, `bugfix/*` branches
- **CI/CD:** Automatically deploys to staging
- **Rules:**
  - All feature branches merge here first
  - Must pass CI before merge
  - Squash merges preferred

### `feature/*` (Feature Development)
- **Purpose:** New features and enhancements
- **Naming:** `feature/agent-name` or `feature/description`
- **Examples:**
  - `feature/ghost-agent-improvements`
  - `feature/backtest-engine-v2`
  - `feature/azure-ml-integration`

### `bugfix/*` (Bug Fixes)
- **Purpose:** Non-urgent bug fixes
- **Naming:** `bugfix/issue-number-description`
- **Examples:**
  - `bugfix/123-data-agent-timeout`
  - `bugfix/risk-calculation-error`

### `hotfix/*` (Emergency Fixes)
- **Purpose:** Critical production issues
- **Merge From:** `main`
- **Merge To:** `main` AND `develop`
- **Naming:** `hotfix/critical-description`

### `release/*` (Release Preparation)
- **Purpose:** Prepare for new version releases
- **Naming:** `release/v1.0.0`
- **Actions:**
  - Update version numbers
  - Final testing
  - Documentation updates

### `infra/*` (Infrastructure)
- **Purpose:** Infrastructure changes (Terraform, Docker, etc.)
- **Naming:** `infra/azure-update` or `infra/terraform-module`
- **Requires:** Additional review for production impact

## Workflow

### Feature Development
```bash
# Start from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-feature

# Work on feature
git add .
git commit -m "feat: description of feature"

# Push and create PR
git push origin feature/my-feature
# Create PR to develop on GitHub
```

### Release Process
```bash
# Create release branch from develop
git checkout develop
git checkout -b release/v1.0.0

# Update version, test, fix minor issues
# Merge to main
git checkout main
git merge release/v1.0.0 --no-ff
git tag v1.0.0
git push origin main --tags

# Merge back to develop
git checkout develop
git merge release/v1.0.0 --no-ff
git push origin develop
```

### Hotfix Process
```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/critical-fix

# Fix and test
git add .
git commit -m "hotfix: critical fix description"

# Merge to main
git checkout main
git merge hotfix/critical-fix --no-ff
git tag v1.0.1
git push origin main --tags

# Merge to develop
git checkout develop
git merge hotfix/critical-fix --no-ff
git push origin develop
```

## Commit Message Convention

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

### Examples
```
feat(ghost-agent): add autonomous decision-making mode
fix(data-agent): resolve timeout on large data fetch
docs(setup): add Azure deployment instructions
refactor(core): optimize agent base class memory usage
test(risk-agent): add unit tests for margin calculation
ci(github): add security scanning to pipeline
```

## Code Review Guidelines

1. **All PRs require review** before merging
2. **CI must pass** - no exceptions
3. **Self-review first** - check your own code before requesting review
4. **Keep PRs small** - easier to review, faster to merge
5. **Update tests** - for any code changes
6. **Update docs** - if behavior changes

## Branch Protection Rules

### `main`
- Require pull request before merging
- Require status checks (CI pipeline)
- Require conversation resolution
- Include administrators
- Do not allow force pushes

### `develop`
- Require status checks (CI pipeline)
- Allow squash merging

---

*Alpha Loop Capital, LLC - Built tough as hell, no compromises.*

