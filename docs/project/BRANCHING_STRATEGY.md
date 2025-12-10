# ALC-Algo Branching Strategy

To maintain "Institutional-Grade" code quality, this repository follows a strict branching strategy. This ensures that unstable code never reaches production and that all changes are verified.

## Core Branches

| Branch | Role | Access | Rules |
|--------|------|--------|-------|
| `main` | **Production**. Stable, battle-tested code only. | **Read-Only** (Merge via PR) | Must pass all tests. 30% MoS verification required. |
| `dev` | **Integration**. Development version. | **Read-Only** (Merge via PR) | Working feature integration. Frequent updates. |

## Feature Branches

All development happens in feature branches. Name them using the following convention:

`type/agent-name/description`

### Types
- `feat`: New capability or agent
- `fix`: Bug fix or patch
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation updates
- `infra`: Azure/Terraform changes

### Examples
- `feat/strategy_agent/new_volatility_model`
- `fix/execution_agent/ibkr_connection_retry`
- `docs/setup/azure_guide_update`
- `infra/azure/terraform_keyvault`

## Workflow

1.  **Branch Off**: Always create your branch from `dev`.
    ```bash
    git checkout dev
    git pull origin dev
    git checkout -b feat/your_feature
    ```

2.  **Develop & Test**: Write code and run tests locally.
    ```bash
    pytest
    ```

3.  **Pull Request (PR)**: Push your branch and open a PR to `dev`.
    -   **Title**: Clear description of the change.
    -   **Description**: Link to requirements/issues.
    -   **Checklist**:
        -   [ ] Tests passed
        -   [ ] Linter passed
        -   [ ] 30% Margin of Safety preserved
        -   [ ] Attributed to Tom Hogan

4.  **Merge to Main**: Periodic releases from `dev` to `main` after full regression testing.

## Information Isolation

-   **Secrets**: NEVER commit `secrets.py` or `.env` files. Use Azure Key Vault or Environment Variables.
-   **Config**: Environment-specific configs (`dev.yaml`, `prod.yaml`) isolate behavior.
-   **Docs**: Documentation lives in `docs/` and is updated with code changes.

## Setup Script

To initialize this structure locally:

```bash
# Ensure you are on main
git checkout main

# Create dev if it doesn't exist
git checkout -b dev

# Push dev
git push -u origin dev
```

