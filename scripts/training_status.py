"""TRAINING STATUS DASHBOARD
============================
Comprehensive view of all training progress, model performance, and launch readiness.

Run: python scripts/training_status.py
"""
import sys
sys.path.insert(0, ".")

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import glob

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Model loading
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    import pickle

# Optional imports
try:
    import pyodbc
    HAS_SQL = True
except ImportError:
    HAS_SQL = False


def count_models_by_type() -> Dict[str, int]:
    """Count models by algorithm type."""
    counts = {"xgboost": 0, "lightgbm": 0, "catboost": 0, "other": 0}
    models_dir = Path("models")
    
    if models_dir.exists():
        for f in models_dir.glob("*.pkl"):
            name = f.stem.lower()
            if "xgboost" in name:
                counts["xgboost"] += 1
            elif "lightgbm" in name:
                counts["lightgbm"] += 1
            elif "catboost" in name:
                counts["catboost"] += 1
            else:
                counts["other"] += 1
    
    return counts


def get_model_metrics(model_file: Path) -> Dict:
    """Extract metrics from a model file."""
    try:
        # Try joblib first (most models), fall back to pickle
        if HAS_JOBLIB:
            try:
                model_data = joblib.load(model_file)
            except Exception:
                import pickle
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)
        else:
            import pickle
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)
        
        metrics = {}
        
        # Handle different model formats
        if isinstance(model_data, dict):
            # Format 1: {"model": ..., "metadata": {"metrics": {...}}} from massive_trainer
            if "metadata" in model_data:
                meta = model_data["metadata"]
                if isinstance(meta, dict):
                    # Nested metrics inside metadata
                    if "metrics" in meta:
                        nested = meta["metrics"]
                        if isinstance(nested, dict):
                            metrics = nested
                        elif hasattr(nested, "auc"):  # ModelMetrics dataclass
                            metrics = {
                                "auc": nested.auc,
                                "accuracy": nested.accuracy,
                                "sharpe": getattr(nested, "sharpe", None),
                                "max_drawdown": getattr(nested, "max_drawdown", None),
                            }
                    else:
                        # cv_auc/cv_acc at metadata level (older format)
                        metrics = {
                            "auc": meta.get("cv_auc"),
                            "accuracy": meta.get("cv_acc"),
                        }
            # Format 2: {"model": ..., "metrics": {...}} from small_mid_cap_models.py
            elif "metrics" in model_data:
                m = model_data["metrics"]
                if hasattr(m, "auc"):  # ModelMetrics dataclass
                    metrics = {
                        "auc": m.auc,
                        "accuracy": m.accuracy,
                        "sharpe": getattr(m, "sharpe", None),
                        "max_drawdown": getattr(m, "max_drawdown", None),
                    }
                elif isinstance(m, dict):
                    metrics = m
            # Format 3: Direct metrics dict
            else:
                metrics = model_data.get("metrics", model_data)
        elif hasattr(model_data, "metrics"):
            metrics = model_data.metrics
        
        return {
            "auc": metrics.get("auc", metrics.get("roc_auc", metrics.get("cv_auc"))),
            "accuracy": metrics.get("accuracy", metrics.get("cv_acc")),
            "sharpe": metrics.get("sharpe", metrics.get("sharpe_ratio")),
            "max_drawdown": metrics.get("max_drawdown"),
            "created": datetime.fromtimestamp(model_file.stat().st_mtime),
            "size_mb": model_file.stat().st_size / 1024 / 1024,
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_models() -> Dict:
    """Analyze all models and return summary statistics."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        return {"total": 0, "promoted": 0, "training": 0}
    
    model_files = list(models_dir.glob("*.pkl"))
    
    results = {
        "total": len(model_files),
        "promoted": 0,
        "training": 0,
        "with_metrics": 0,
        "unique_symbols": set(),
        "best_auc": 0,
        "best_sharpe": 0,
        "avg_auc": [],
        "recent_24h": 0,
        "by_type": count_models_by_type(),
    }
    
    # Promotion thresholds
    MIN_AUC = 0.52
    MIN_SHARPE = 1.5
    MAX_DRAWDOWN = 0.05
    
    now = datetime.now()
    cutoff = now - timedelta(hours=24)
    
    for f in model_files:
        metrics = get_model_metrics(f)
        
        if "error" in metrics:
            continue
        
        # Extract symbol
        symbol = f.stem.split("_")[0]
        results["unique_symbols"].add(symbol)
        
        # Check if recent
        if metrics.get("created") and metrics["created"] > cutoff:
            results["recent_24h"] += 1
        
        auc = metrics.get("auc")
        sharpe = metrics.get("sharpe")
        drawdown = metrics.get("max_drawdown")
        
        if auc is not None:
            results["with_metrics"] += 1
            results["avg_auc"].append(auc)
            
            if auc > results["best_auc"]:
                results["best_auc"] = auc
            
            if sharpe and sharpe > results["best_sharpe"]:
                results["best_sharpe"] = sharpe
            
            # Check promotion criteria
            promoted = (
                auc >= MIN_AUC and
                (sharpe is None or sharpe >= MIN_SHARPE) and
                (drawdown is None or drawdown <= MAX_DRAWDOWN)
            )
            
            if promoted:
                results["promoted"] += 1
            else:
                results["training"] += 1
    
    # Calculate averages
    if results["avg_auc"]:
        results["avg_auc"] = sum(results["avg_auc"]) / len(results["avg_auc"])
    else:
        results["avg_auc"] = 0
    
    results["unique_symbols"] = len(results["unique_symbols"])
    
    return results


def check_active_training() -> Dict:
    """Check if training is currently running."""
    result = {
        "small_mid_cap": {"running": False, "last_cycle": None},
        "overnight": {"running": False, "last_cycle": None},
        "agent_trainer": {"running": False, "last_cycle": None},
    }
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return result
    
    now = datetime.now()
    cutoff = now - timedelta(minutes=20)  # Within 20 minutes = likely still running
    
    # Check small/mid cap training logs
    small_mid_logs = sorted(logs_dir.glob("small_mid_cap_training_*.log"), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if small_mid_logs:
        latest = small_mid_logs[0]
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        result["small_mid_cap"]["last_cycle"] = mod_time
        result["small_mid_cap"]["running"] = mod_time > cutoff
    
    # Check overnight training log
    overnight_log = logs_dir / "overnight_training.log"
    if overnight_log.exists():
        mod_time = datetime.fromtimestamp(overnight_log.stat().st_mtime)
        result["overnight"]["last_cycle"] = mod_time
        result["overnight"]["running"] = mod_time > cutoff
    
    return result


def get_data_status() -> Dict:
    """Check status of market data."""
    result = {
        "azure_sql": {"connected": False, "rows": 0},
        "csv_backup": {"files": 0, "total_mb": 0},
    }
    
    # Check Azure SQL
    if HAS_SQL:
        try:
            conn = pyodbc.connect(
                "Driver={ODBC Driver 17 for SQL Server};"
                "Server=alc-sql-server.database.windows.net;"
                "Database=alc_market_data;"
                "UID=CloudSAb3fcbb35;"
                "PWD=ALCadmin27!",
                timeout=5
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_bars")
            rows = cursor.fetchone()[0]
            result["azure_sql"]["connected"] = True
            result["azure_sql"]["rows"] = rows
            conn.close()
        except Exception as e:
            result["azure_sql"]["error"] = str(e)[:50]
    
    # Check CSV backups
    csv_dir = Path("data/csv_backup")
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        result["csv_backup"]["files"] = len(csv_files)
        result["csv_backup"]["total_mb"] = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    
    return result


def print_status_dashboard():
    """Print the comprehensive training status dashboard."""
    print("\n" + "=" * 80)
    print("                    🎯 TRAINING STATUS DASHBOARD")
    print("=" * 80)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # MODEL STATUS
    print("\n  📊 MODEL TRAINING STATUS")
    print("  " + "-" * 40)
    
    models = analyze_models()
    
    print(f"  Total Models:         {models['total']}")
    print(f"  Unique Symbols:       {models['unique_symbols']}")
    print(f"  Trained in Last 24h:  {models['recent_24h']}")
    print()
    print(f"  By Algorithm:")
    print(f"    XGBoost:            {models['by_type']['xgboost']}")
    print(f"    LightGBM:           {models['by_type']['lightgbm']}")
    print(f"    CatBoost:           {models['by_type']['catboost']}")
    print(f"    Other:              {models['by_type']['other']}")
    print()
    print(f"  With Metrics:         {models['with_metrics']}")
    print(f"  Average AUC:          {models['avg_auc']:.4f}" if models['avg_auc'] else "  Average AUC:          N/A")
    print(f"  Best AUC:             {models['best_auc']:.4f}" if models['best_auc'] else "  Best AUC:             N/A")
    print(f"  Best Sharpe:          {models['best_sharpe']:.2f}" if models['best_sharpe'] else "  Best Sharpe:          N/A")
    
    # PROMOTION STATUS
    print("\n  🏆 PROMOTION STATUS")
    print("  " + "-" * 40)
    print(f"  Promoted (Live Ready): {models['promoted']}")
    print(f"  Still Training:        {models['training']}")
    
    if models['total'] > 0:
        pct = (models['promoted'] / models['total']) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  Progress:              [{bar}] {pct:.1f}%")
    
    # Promotion criteria reminder
    print("\n  Promotion Criteria:")
    print("    ✓ AUC >= 0.52 (better than random)")
    print("    ✓ Sharpe Ratio >= 1.5 (good risk-adjusted return)")
    print("    ✓ Max Drawdown <= 5% (acceptable risk)")
    
    # ACTIVE TRAINING
    print("\n  🔄 ACTIVE TRAINING PROCESSES")
    print("  " + "-" * 40)
    
    training = check_active_training()
    
    for name, info in training.items():
        status = "✅ RUNNING" if info["running"] else "⏸️ Stopped"
        last = info["last_cycle"].strftime("%H:%M:%S") if info["last_cycle"] else "Never"
        print(f"  {name:20} {status:12} Last: {last}")
    
    # DATA STATUS
    print("\n  💾 DATA STATUS")
    print("  " + "-" * 40)
    
    data = get_data_status()
    
    sql_status = "✅ Connected" if data["azure_sql"]["connected"] else "❌ Not connected"
    print(f"  Azure SQL:    {sql_status}")
    if data["azure_sql"]["connected"]:
        print(f"                {data['azure_sql']['rows']:,} price bars")
    elif "error" in data["azure_sql"]:
        print(f"                Error: {data['azure_sql']['error']}")
    
    print(f"  CSV Backup:   {data['csv_backup']['files']} files ({data['csv_backup']['total_mb']:.1f} MB)")
    
    # LAUNCH READINESS
    print("\n" + "=" * 80)
    print("  🚀 LAUNCH READINESS")
    print("=" * 80)
    
    ready_models = models['promoted'] > 0
    ready_data = data["azure_sql"]["connected"] and data["azure_sql"]["rows"] > 10000
    
    # Check IBKR
    ibkr_status = "Not checked"
    ibkr_ready = False
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", 7497))
        ibkr_ready = result == 0
        ibkr_status = "Connected (port 7497)" if ibkr_ready else "Not running - Start TWS/Gateway"
        sock.close()
    except:
        ibkr_status = "Check manually"
    
    checks = [
        ("Promoted Models", ready_models, f"{models['promoted']} models ready"),
        ("Market Data", ready_data, f"{data['azure_sql']['rows']:,} bars" if data["azure_sql"]["connected"] else "No connection"),
        ("IBKR Connection", ibkr_ready, ibkr_status),
    ]
    
    print()
    all_core_ready = ready_models and ready_data
    for check_name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name:20} {detail}")
    
    print()
    if all_core_ready:
        print("  " + "=" * 50)
        print("  ✅ MODELS & DATA READY FOR TRADING")
        if not ibkr_ready:
            print("  ⚠️  Start IBKR TWS/Gateway before live trading")
        print("  " + "=" * 50)
    else:
        print("  " + "-" * 50)
        print("  ⚠️  NOT YET READY - See issues above")
        print("  " + "-" * 50)
    
    # NEXT STEPS
    print("\n  📋 NEXT STEPS")
    print("  " + "-" * 40)
    
    if not ready_models:
        print("""
  1. CONTINUE TRAINING (Models need to pass promotion criteria)
     Training is automatic - models improve over cycles.
     
     To monitor: 
       Get-Content logs\\small_mid_cap_training_*.log -Tail 20 -Wait
     
     Or run the dashboard:
       python scripts\\model_dashboard.py
""")
    elif not ready_data:
        print("""
  1. HYDRATE MORE DATA
     Run: python scripts\\hydrate_full_universe.py
     
     Or for Alpha Vantage:
       python scripts\\hydrate_alpha_vantage.py
""")
    else:
        print("""
  1. START PAPER TRADING (Recommended First)
     Double-click: scripts\\START_PAPER_TRADING.bat
     
     Or run:
       python src\\trading\\production_algo.py --paper

  2. CHECK READINESS REPORT
     python src\\trading\\production_algo.py --check

  3. WHEN CONFIDENT - Start Live Trading
     ⚠️  CAUTION: Real money at risk!
     Double-click: scripts\\START_LIVE_TRADING.bat
""")
    
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Training Status Dashboard")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--brief", action="store_true", help="Brief output only")
    args = parser.parse_args()
    
    if args.json:
        import json
        status = {
            "models": analyze_models(),
            "training": check_active_training(),
            "data": get_data_status(),
            "timestamp": datetime.now().isoformat(),
        }
        # Convert datetime objects to strings for JSON serialization
        for k, v in status["training"].items():
            if v["last_cycle"]:
                v["last_cycle"] = v["last_cycle"].isoformat()
        print(json.dumps(status, indent=2, default=str))
    elif args.brief:
        models = analyze_models()
        training = check_active_training()
        is_running = any(t["running"] for t in training.values())
        
        print(f"Models: {models['total']} total, {models['promoted']} promoted")
        print(f"Training: {'Running' if is_running else 'Stopped'}")
        print(f"Last 24h: {models['recent_24h']} new models")
    else:
        print_status_dashboard()


if __name__ == "__main__":
    main()

