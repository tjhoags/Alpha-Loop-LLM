"""MODEL PERFORMANCE DASHBOARD
============================
Shows you exactly where models started and where they ended up.

Run: python scripts/model_dashboard.py
"""
import sys

sys.path.insert(0, ".")

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Try to import optional dependencies
try:
    import pyodbc
    HAS_SQL = True
except ImportError:
    HAS_SQL = False


def load_model_grades() -> List[Dict]:
    """Load all trained models and their grades."""
    models_dir = Path("models")
    grades = []

    for model_file in models_dir.glob("*.pkl"):
        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            # Extract grade info
            grade = {
                "file": model_file.name,
                "symbol": model_file.stem.replace("_ensemble", "").replace("_model", ""),
                "created": datetime.fromtimestamp(model_file.stat().st_mtime),
                "size_mb": model_file.stat().st_size / 1024 / 1024,
            }

            # Try to get metrics from model
            if isinstance(model_data, dict):
                grade.update({
                    "auc": model_data.get("auc", model_data.get("metrics", {}).get("auc")),
                    "accuracy": model_data.get("accuracy", model_data.get("metrics", {}).get("accuracy")),
                    "sharpe": model_data.get("sharpe_ratio", model_data.get("metrics", {}).get("sharpe")),
                    "max_drawdown": model_data.get("max_drawdown", model_data.get("metrics", {}).get("max_drawdown")),
                    "win_rate": model_data.get("win_rate"),
                    "profit_factor": model_data.get("profit_factor"),
                })
            elif hasattr(model_data, "grade"):
                grade.update(model_data.grade)
            elif hasattr(model_data, "metrics"):
                grade.update(model_data.metrics)

            grades.append(grade)
        except Exception as e:
            grades.append({
                "file": model_file.name,
                "symbol": model_file.stem,
                "error": str(e),
            })

    return grades


def interpret_metric(name: str, value: float) -> str:
    """Interpret what a metric value means."""
    if value is None:
        return "N/A"

    interpretations = {
        "auc": [
            (0.50, "Random (no skill)"),
            (0.52, "Minimal edge"),
            (0.55, "Weak signal"),
            (0.58, "Decent signal"),
            (0.62, "Good signal"),
            (0.65, "Strong signal"),
            (0.70, "Excellent (rare)"),
            (1.00, "Perfect (suspicious)"),
        ],
        "accuracy": [
            (0.50, "Random (coin flip)"),
            (0.52, "Slight edge"),
            (0.54, "Weak edge"),
            (0.56, "Decent edge"),
            (0.58, "Good edge"),
            (0.60, "Strong edge"),
            (0.65, "Excellent"),
            (1.00, "Perfect (overfitting?)"),
        ],
        "sharpe": [
            (0.0, "No risk-adjusted return"),
            (0.5, "Below average"),
            (1.0, "Acceptable"),
            (1.5, "Good"),
            (2.0, "Very good"),
            (2.5, "Excellent"),
            (3.0, "Outstanding"),
            (999, "Elite"),
        ],
        "max_drawdown": [
            (0.01, "Excellent risk control"),
            (0.03, "Good risk control"),
            (0.05, "Acceptable"),
            (0.10, "Moderate risk"),
            (0.15, "High risk"),
            (0.20, "Very high risk"),
            (1.00, "Dangerous"),
        ],
    }

    if name not in interpretations:
        return f"{value:.4f}"

    for threshold, meaning in interpretations[name]:
        if value <= threshold:
            return f"{value:.4f} - {meaning}"

    return f"{value:.4f}"


def get_promotion_status(grade: Dict) -> str:
    """Determine if model should be promoted to production."""
    # Thresholds for production
    MIN_AUC = 0.52
    MIN_ACCURACY = 0.52
    MIN_SHARPE = 1.5
    MAX_DRAWDOWN = 0.05

    auc = grade.get("auc")
    accuracy = grade.get("accuracy")
    sharpe = grade.get("sharpe")
    drawdown = grade.get("max_drawdown")

    issues = []

    if auc is not None and auc < MIN_AUC:
        issues.append(f"AUC {auc:.3f} < {MIN_AUC}")
    if accuracy is not None and accuracy < MIN_ACCURACY:
        issues.append(f"Accuracy {accuracy:.3f} < {MIN_ACCURACY}")
    if sharpe is not None and sharpe < MIN_SHARPE:
        issues.append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE}")
    if drawdown is not None and drawdown > MAX_DRAWDOWN:
        issues.append(f"Drawdown {drawdown:.1%} > {MAX_DRAWDOWN:.0%}")

    if not issues:
        return "PROMOTED - Ready for live trading"
    else:
        return f"TRAINING - Issues: {', '.join(issues)}"


def print_dashboard():
    """Print the model performance dashboard."""
    print("\n" + "=" * 80)
    print("                    MODEL PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    grades = load_model_grades()

    if not grades:
        print("\n  No models found in models/ directory.")
        print('  Run training first: python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"')
        print("=" * 80)
        return

    # Summary stats
    total = len(grades)
    with_metrics = [g for g in grades if g.get("auc") is not None]
    promoted = [g for g in with_metrics if "PROMOTED" in get_promotion_status(g)]

    print("\n  SUMMARY")
    print("  -------")
    print(f"  Total Models:     {total}")
    print(f"  With Metrics:     {len(with_metrics)}")
    print(f"  Production Ready: {len(promoted)}")
    print(f"  Still Training:   {len(with_metrics) - len(promoted)}")

    # Metric interpretation guide
    print("\n  METRIC INTERPRETATION GUIDE")
    print("  ----------------------------")
    print("  AUC (Area Under Curve):")
    print("    0.50 = Random (no skill)")
    print("    0.55 = Weak signal")
    print("    0.60 = Good signal")
    print("    0.65+ = Strong signal (hedge fund quality)")
    print("")
    print("  Sharpe Ratio (Risk-Adjusted Return):")
    print("    0.0 = No edge")
    print("    1.0 = Acceptable")
    print("    1.5 = Good (our minimum)")
    print("    2.0+ = Excellent")
    print("    3.0+ = Outstanding (top funds)")
    print("")
    print("  Max Drawdown (Worst Loss):")
    print("    < 3% = Excellent risk control")
    print("    < 5% = Good (our maximum)")
    print("    > 10% = High risk")

    # Individual model grades
    print("\n  INDIVIDUAL MODEL GRADES")
    print("  -----------------------")

    for grade in sorted(grades, key=lambda x: x.get("auc") or 0, reverse=True):
        symbol = grade.get("symbol", "Unknown")
        print(f"\n  {symbol}")
        print(f"  {'-' * len(symbol)}")

        if "error" in grade:
            print(f"    Error: {grade['error']}")
            continue

        auc = grade.get("auc")
        accuracy = grade.get("accuracy")
        sharpe = grade.get("sharpe")
        drawdown = grade.get("max_drawdown")

        print(f"    AUC:          {interpret_metric('auc', auc)}")
        print(f"    Accuracy:     {interpret_metric('accuracy', accuracy)}")
        print(f"    Sharpe Ratio: {interpret_metric('sharpe', sharpe)}")
        print(f"    Max Drawdown: {interpret_metric('max_drawdown', drawdown)}")
        print(f"    Status:       {get_promotion_status(grade)}")

    print("\n" + "=" * 80)
    print("  HOW TO IMPROVE MODELS")
    print("=" * 80)
    print("""
  1. MORE DATA = Better generalization
     - Run hydration longer to get more historical data
     - Add more data sources (news, fundamentals)

  2. MORE TRAINING TIME = Better convergence
     - Let training run overnight (6-8+ hours)
     - Models improve as they see more iterations

  3. FEATURE ENGINEERING = Better signals
     - Add behavioral features (sentiment, fear/greed)
     - Add fundamental features (P/E, growth rates)

  4. HYPERPARAMETER TUNING = Better fit
     - Models auto-tune during training
     - Better data = better hyperparameters found

  5. ENSEMBLE DIVERSITY = More robust
     - We use XGBoost + LightGBM + CatBoost
     - Different algorithms catch different patterns
    """)
    print("=" * 80)


def save_grades_to_sql():
    """Save model grades to Azure SQL for tracking over time."""
    if not HAS_SQL:
        print("pyodbc not available - skipping SQL export")
        return

    try:
        conn = pyodbc.connect(
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=alc-sql-server.database.windows.net;"
            "Database=alc_market_data;"
            "UID=CloudSAb3fcbb35;"
            "PWD=ALCadmin27!",
        )
        cursor = conn.cursor()

        # Create grades table if not exists
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='model_grades' AND xtype='U')
            CREATE TABLE model_grades (
                id INT IDENTITY(1,1) PRIMARY KEY,
                symbol VARCHAR(20),
                auc FLOAT,
                accuracy FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                status VARCHAR(50),
                recorded_at DATETIME DEFAULT GETDATE()
            )
        """)
        conn.commit()

        # Insert grades
        grades = load_model_grades()
        for grade in grades:
            if grade.get("auc") is not None:
                cursor.execute("""
                    INSERT INTO model_grades (symbol, auc, accuracy, sharpe_ratio, max_drawdown, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    grade.get("symbol"),
                    grade.get("auc"),
                    grade.get("accuracy"),
                    grade.get("sharpe"),
                    grade.get("max_drawdown"),
                    "PROMOTED" if "PROMOTED" in get_promotion_status(grade) else "TRAINING",
                ))

        conn.commit()
        print(f"\n  Saved {len(grades)} model grades to Azure SQL")
        print("  Track progress over time with: SELECT * FROM model_grades ORDER BY recorded_at")
        conn.close()

    except Exception as e:
        print(f"\n  Could not save to SQL: {e}")


if __name__ == "__main__":
    print_dashboard()

    # Optionally save to SQL for tracking
    if "--save-sql" in sys.argv:
        save_grades_to_sql()


