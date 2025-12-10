"""
Step Functions Training Pipeline
================================
AWS Step Function definitions and Lambda handlers for live data training.

Files:
- live_training_pipeline.json: Full training pipeline with all tiers
- incremental_training.json: Quick incremental updates (scheduled)
- lambda_handlers.py: Lambda function implementations
"""

from pathlib import Path

STEP_FUNCTIONS_DIR = Path(__file__).parent

def get_step_function_definition(name: str) -> dict:
    """Load a step function definition by name."""
    import json
    path = STEP_FUNCTIONS_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    raise FileNotFoundError(f"Step function not found: {name}")
