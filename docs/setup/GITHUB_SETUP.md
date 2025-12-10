# GitHub Setup Guide for ALC-Algo

**Institutional-Grade Version Control Setup**

This guide outlines the procedure to initialize the repository, secure it, and push it to GitHub. This is designed for the "Other Agent" or the user to follow to ensure a compliant setup.

**Author:** Tom Hogan | Alpha Loop Capital, LLC

---

## 1. Prerequisites

Ensure you have the following tools installed:
-   **Git**: [Download Git](https://git-scm.com/downloads)
-   **GitHub Account**: [Sign up](https://github.com)
-   **GitHub CLI (Optional but Recommended)**: [Download `gh`](https://cli.github.com/)

---

## 2. Repository Initialization (One-Time Setup)

You can use the provided automated scripts or manual commands.

### Option A: Automated Script

**Windows (PowerShell):**
```powershell
.\scripts\setup_github.ps1
```

**Mac/Linux (Bash):**
```bash
chmod +x scripts/setup_github.sh
./scripts/setup_github.sh
```

### Option B: Manual Setup

1.  **Initialize Git**:
    ```bash
    git init -b main
    ```

2.  **Verify .gitignore** (CRITICAL):
    Ensure `config/secrets.py` and `.env` are ignored.
    ```bash
    cat .gitignore
    ```

3.  **Add Files**:
    ```bash
    git add .
    ```

4.  **Initial Commit**:
    ```bash
    git commit -m "feat: Initial commit of ALC-Algo Institutional Platform"
    ```

5.  **Create Repository on GitHub**:
    -   Go to GitHub.com -> New Repository.
    -   Name: `ALC-Algo`.
    -   **Private**: YES (This is proprietary).

6.  **Link & Push**:
    ```bash
    git remote add origin https://github.com/AlphaLoopCapital/ALC-Algo.git
    git push -u origin main
    ```

---

## 3. Branching Strategy (Strict Adherence)

Refer to `BRANCHING_STRATEGY.md` for full details.

-   **`main`**: Production (Protected).
-   **`dev`**: Integration (Default for PRs).
-   **`feat/*`**: Feature branches.

**Setup Dev Branch:**
```bash
git checkout -b dev
git push -u origin dev
```

---

## 4. CI/CD Pipeline

The repository includes a GitHub Actions workflow in `.github/workflows/ci.yml`.

**What it does:**
1.  **Linting**: Flake8, Black, Isort.
2.  **Testing**: Runs `pytest` with 30% Margin of Safety checks.
3.  **Security**: runs `bandit` and `safety` scans.

**Status Check:**
-   Go to `Actions` tab in GitHub repository.
-   Ensure the `CI Pipeline` is green.

---

## 5. Security & Secrets

**NEVER COMMIT SECRETS.**

Use **GitHub Secrets** for CI/CD keys:
1.  Go to **Settings** -> **Secrets and variables** -> **Actions**.
2.  Add the following secrets:
    -   `ALPHA_VANTAGE_API_KEY`
    -   `OPENAI_API_KEY`
    -   `IBKR_ACCOUNT`
    -   `AZURE_CREDENTIALS` (JSON blob from Azure CLI)

---

**Troubleshooting**
-   **Permission Denied**: Check your SSH keys (`ssh-add -l`) or use Personal Access Token.
-   **Large Files**: Ensure `data/` is ignored. If you committed large files, use `git filter-repo` to remove them.

**Attribution**: Tom Hogan | Alpha Loop Capital, LLC

