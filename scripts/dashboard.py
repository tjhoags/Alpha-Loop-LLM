"""================================================================================
ALPHA LOOP CAPITAL - UNIFIED DASHBOARD
================================================================================
Real-time monitoring dashboard for:
1. Data Ingestion Status (all sources)
2. Model Training Progress & Weights
3. Trading Performance & Signals
4. System Health

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --refresh 5   # Refresh every 5 seconds
    python scripts/dashboard.py --web         # Launch web dashboard
================================================================================
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger

# Try to import rich for nice terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn  # noqa: F401
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[WARNING] Install 'rich' for better dashboard: pip install rich")


# =============================================================================
# DATA INGESTION STATUS
# =============================================================================

class DataIngestionMonitor:
    """Monitor data ingestion status across all sources."""

    def __init__(self):
        self.sources = {
            "polygon": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
            "massive": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
            "alpha_vantage": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
            "ibkr": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
            "coinbase": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
            "fred": {"status": "unknown", "last_run": None, "rows": 0, "errors": 0},
        }

    def check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are configured (not placeholders)."""
        from src.config.settings import get_settings
        settings = get_settings()

        def is_real_key(key: str) -> bool:
            """Check if key is a real value, not a placeholder."""
            if not key:
                return False
            # Detect placeholder values
            placeholders = ["your_", "xxx", "placeholder", "REPLACE", "TODO", "changeme"]
            key_lower = key.lower()
            return not any(p in key_lower for p in placeholders)

        return {
            "polygon": is_real_key(settings.polygon_api_key),
            "massive": is_real_key(settings.massive_access_key) and is_real_key(settings.massive_secret_key),
            "alpha_vantage": is_real_key(settings.alpha_vantage_api_key),
            "ibkr": True,  # No API key needed, just connection
            "coinbase": is_real_key(settings.coinbase_api_key),
            "fred": is_real_key(settings.fred_api_key),
        }

    def get_database_stats(self) -> Dict[str, int]:
        """Get row counts from database tables."""
        try:
            from src.database.connection import get_engine
            engine = get_engine()

            tables = ["price_bars", "options_contracts", "macro_indicators", "fundamentals"]
            stats = {}

            for table in tables:
                try:
                    query = f"SELECT COUNT(*) as cnt FROM {table}"
                    df = pd.read_sql(query, engine)
                    stats[table] = df["cnt"].iloc[0]
                except Exception:
                    stats[table] = 0

            # Get latest timestamps
            try:
                query = "SELECT MAX(timestamp) as latest FROM price_bars"
                df = pd.read_sql(query, engine)
                stats["latest_price"] = df["latest"].iloc[0]
            except Exception:
                stats["latest_price"] = None

            return stats

        except Exception as e:
            logger.debug(f"Database stats error: {e}")
            return {"price_bars": 0, "options_contracts": 0, "macro_indicators": 0, "fundamentals": 0}

    def get_symbol_coverage(self) -> Dict[str, int]:
        """Get unique symbol counts by source."""
        try:
            from src.database.connection import get_engine
            engine = get_engine()

            query = """
            SELECT source, COUNT(DISTINCT symbol) as symbols
            FROM price_bars
            GROUP BY source
            """
            df = pd.read_sql(query, engine)
            return dict(zip(df["source"], df["symbols"]))
        except Exception:
            return {}


# =============================================================================
# MODEL TRAINING MONITOR
# =============================================================================

class ModelTrainingMonitor:
    """Monitor model training status and weights."""

    def __init__(self):
        from src.config.settings import get_settings
        self.settings = get_settings()
        self.models_dir = self.settings.models_dir

    def get_model_inventory(self) -> List[Dict]:
        """Get all trained models with their metrics."""
        models = []

        if not self.models_dir.exists():
            return models

        import joblib

        for model_file in self.models_dir.glob("*.pkl"):
            try:
                data = joblib.load(model_file)

                if isinstance(data, dict):
                    metrics = data.get("metrics", {})
                    trained_at = data.get("trained_at", "unknown")
                else:
                    metrics = getattr(data, "metrics", {})
                    trained_at = "unknown"

                models.append({
                    "name": model_file.stem,
                    "file": model_file.name,
                    "size_kb": model_file.stat().st_size / 1024,
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime),
                    "auc": metrics.get("auc", metrics.get("roc_auc", 0)),
                    "accuracy": metrics.get("accuracy", 0),
                    "f1": metrics.get("f1", 0),
                    "sharpe": metrics.get("sharpe", metrics.get("sharpe_ratio", 0)),
                    "trained_at": trained_at,
                    "promoted": self._check_promotion(metrics),
                })
            except Exception as e:
                logger.debug(f"Error loading {model_file}: {e}")

        return sorted(models, key=lambda x: x["modified"], reverse=True)

    def _check_promotion(self, metrics: Dict) -> bool:
        """Check if model passes promotion thresholds."""
        auc = metrics.get("auc", metrics.get("roc_auc", 0))
        accuracy = metrics.get("accuracy", 0)
        sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio", 0))
        drawdown = metrics.get("max_drawdown", 1.0)

        return (
            auc >= 0.52 and
            accuracy >= 0.52 and
            sharpe >= 1.5 and
            drawdown <= 0.05
        )

    def get_model_weights(self) -> Dict[str, float]:
        """Get ensemble model weights."""
        weights = {}
        models = self.get_model_inventory()

        # Calculate weights based on AUC
        total_auc = sum(m["auc"] for m in models if m["promoted"])

        if total_auc > 0:
            for model in models:
                if model["promoted"]:
                    weights[model["name"]] = model["auc"] / total_auc

        return weights

    def get_training_summary(self) -> Dict:
        """Get summary of training status."""
        models = self.get_model_inventory()

        return {
            "total_models": len(models),
            "promoted_models": sum(1 for m in models if m["promoted"]),
            "avg_auc": sum(m["auc"] for m in models) / len(models) if models else 0,
            "avg_accuracy": sum(m["accuracy"] for m in models) / len(models) if models else 0,
            "latest_training": max((m["modified"] for m in models), default=None),
        }


# =============================================================================
# TRADING MONITOR
# =============================================================================

class TradingMonitor:
    """Monitor trading status and performance."""

    def __init__(self):
        self.positions = {}
        self.daily_pnl = 0.0
        self.signals = []

    def check_ibkr_connection(self) -> Dict:
        """Check IBKR connection status."""
        import socket
        from src.config.settings import get_settings
        settings = get_settings()

        result = {
            "paper_connected": False,
            "live_connected": False,
            "port": settings.ibkr_port,
        }

        # Check paper trading port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            paper_result = sock.connect_ex((settings.ibkr_host, 7497))
            result["paper_connected"] = paper_result == 0
            sock.close()
        except OSError:
            pass

        # Check live trading port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            live_result = sock.connect_ex((settings.ibkr_host, 7496))
            result["live_connected"] = live_result == 0
            sock.close()
        except OSError:
            pass

        return result

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent trading signals from logs."""
        signals = []
        log_file = Path("logs/live_trading.log")

        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()[-100:]  # Last 100 lines

                for line in reversed(lines):
                    if "[SIGNAL]" in line or "[PAPER]" in line or "[LIVE]" in line:
                        signals.append({"log": line.strip()})
                        if len(signals) >= limit:
                            break
            except (OSError, IOError):
                pass

        return signals


# =============================================================================
# TERMINAL DASHBOARD (Rich)
# =============================================================================

def create_rich_dashboard(refresh_rate: int = 5):
    """Create rich terminal dashboard."""
    if not RICH_AVAILABLE:
        print("Rich not available. Install with: pip install rich")
        return

    console = Console()

    # Initialize monitors
    data_monitor = DataIngestionMonitor()
    model_monitor = ModelTrainingMonitor()
    trading_monitor = TradingMonitor()

    def generate_dashboard() -> Layout:
        """Generate the dashboard layout."""
        layout = Layout()

        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="data_status"),
            Layout(name="database"),
        )

        layout["right"].split_column(
            Layout(name="models"),
            Layout(name="trading"),
        )

        # Header
        header_text = Text()
        header_text.append("ALPHA LOOP CAPITAL", style="bold blue")
        header_text.append(" - Trading Dashboard - ", style="dim")
        header_text.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="green")
        layout["header"].update(Panel(header_text, box=box.DOUBLE))

        # Data Ingestion Status
        api_keys = data_monitor.check_api_keys()
        data_table = Table(title="Data Sources", box=box.SIMPLE)
        data_table.add_column("Source", style="cyan")
        data_table.add_column("API Key", justify="center")
        data_table.add_column("Status", justify="center")

        for source, configured in api_keys.items():
            key_status = "[green]OK[/green]" if configured else "[red]MISSING[/red]"
            status = "[green]Ready[/green]" if configured else "[yellow]Configure[/yellow]"
            data_table.add_row(source.upper(), key_status, status)

        layout["data_status"].update(Panel(data_table, title="[bold]Data Ingestion[/bold]"))

        # Database Stats
        db_stats = data_monitor.get_database_stats()
        db_table = Table(title="Database", box=box.SIMPLE)
        db_table.add_column("Table", style="cyan")
        db_table.add_column("Rows", justify="right")

        for table, count in db_stats.items():
            if table != "latest_price":
                db_table.add_row(table, f"{count:,}")

        if db_stats.get("latest_price"):
            db_table.add_row("Latest Data", str(db_stats["latest_price"])[:19])

        layout["database"].update(Panel(db_table, title="[bold]Database[/bold]"))

        # Models
        models = model_monitor.get_model_inventory()[:8]
        model_table = Table(title="Trained Models", box=box.SIMPLE)
        model_table.add_column("Model", style="cyan", max_width=25)
        model_table.add_column("AUC", justify="right")
        model_table.add_column("Acc", justify="right")
        model_table.add_column("Status", justify="center")

        for model in models:
            status = "[green]PROMOTED[/green]" if model["promoted"] else "[yellow]PENDING[/yellow]"
            model_table.add_row(
                model["name"][:25],
                f"{model['auc']:.3f}",
                f"{model['accuracy']:.3f}",
                status,
            )

        summary = model_monitor.get_training_summary()
        model_table.add_row("---", "---", "---", "---")
        model_table.add_row(
            f"Total: {summary['total_models']}",
            f"Avg: {summary['avg_auc']:.3f}",
            f"Avg: {summary['avg_accuracy']:.3f}",
            f"[green]{summary['promoted_models']}[/green] promoted",
        )

        layout["models"].update(Panel(model_table, title="[bold]Model Training[/bold]"))

        # Trading Status
        ibkr_status = trading_monitor.check_ibkr_connection()
        trading_table = Table(title="Trading", box=box.SIMPLE)
        trading_table.add_column("Component", style="cyan")
        trading_table.add_column("Status", justify="center")

        paper_status = "[green]CONNECTED[/green]" if ibkr_status["paper_connected"] else "[red]OFFLINE[/red]"
        live_status = "[green]CONNECTED[/green]" if ibkr_status["live_connected"] else "[dim]offline[/dim]"

        trading_table.add_row("IBKR Paper (7497)", paper_status)
        trading_table.add_row("IBKR Live (7496)", live_status)
        trading_table.add_row("Active Port", str(ibkr_status["port"]))

        # Recent signals
        signals = trading_monitor.get_recent_signals(5)
        if signals:
            trading_table.add_row("---", "---")
            trading_table.add_row("[bold]Recent Signals[/bold]", "")
            for sig in signals[:3]:
                trading_table.add_row("", sig["log"][-50:])

        layout["trading"].update(Panel(trading_table, title="[bold]Trading Status[/bold]"))

        # Footer
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold")
        footer_text.append(" to exit | Refresh: ", style="dim")
        footer_text.append(f"{refresh_rate}s", style="green")
        layout["footer"].update(Panel(footer_text, box=box.SIMPLE))

        return layout

    # Run live dashboard
    with Live(generate_dashboard(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                time.sleep(refresh_rate)
                live.update(generate_dashboard())
        except KeyboardInterrupt:
            pass


# =============================================================================
# SIMPLE TEXT DASHBOARD (Fallback)
# =============================================================================

def create_simple_dashboard(refresh_rate: int = 5):
    """Create simple text-based dashboard (no rich)."""
    data_monitor = DataIngestionMonitor()
    model_monitor = ModelTrainingMonitor()
    trading_monitor = TradingMonitor()

    def print_dashboard():
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 70)
        print("        ALPHA LOOP CAPITAL - TRADING DASHBOARD")
        print(f"        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # API Keys
        print("\n[DATA SOURCES]")
        api_keys = data_monitor.check_api_keys()
        for source, configured in api_keys.items():
            status = "[OK]" if configured else "[MISSING]"
            print(f"  {source.upper():15s} {status}")

        # Database
        print("\n[DATABASE]")
        db_stats = data_monitor.get_database_stats()
        for table, count in db_stats.items():
            if table != "latest_price":
                print(f"  {table:20s} {count:>10,} rows")

        # Models
        print("\n[MODELS]")
        summary = model_monitor.get_training_summary()
        print(f"  Total Models:    {summary['total_models']}")
        print(f"  Promoted:        {summary['promoted_models']}")
        print(f"  Avg AUC:         {summary['avg_auc']:.4f}")
        print(f"  Avg Accuracy:    {summary['avg_accuracy']:.4f}")

        # Model weights
        weights = model_monitor.get_model_weights()
        if weights:
            print("\n[MODEL WEIGHTS]")
            for name, weight in sorted(weights.items(), key=lambda x: -x[1])[:5]:
                print(f"  {name:25s} {weight:.2%}")

        # Trading
        print("\n[TRADING]")
        ibkr = trading_monitor.check_ibkr_connection()
        paper = "CONNECTED" if ibkr["paper_connected"] else "OFFLINE"
        live = "CONNECTED" if ibkr["live_connected"] else "offline"
        print(f"  IBKR Paper (7497): {paper}")
        print(f"  IBKR Live (7496):  {live}")

        print("\n" + "=" * 70)
        print(f"Refreshing every {refresh_rate}s... Press Ctrl+C to exit")

    try:
        while True:
            print_dashboard()
            time.sleep(refresh_rate)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


# =============================================================================
# WEB DASHBOARD (Flask)
# =============================================================================

def create_web_dashboard(port: int = 5000):
    """Create web-based dashboard using Flask."""
    try:
        from flask import Flask, render_template_string, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return

    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alpha Loop Capital - Dashboard</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
            .header { text-align: center; padding: 20px; background: #16213e; border-radius: 10px; margin-bottom: 20px; }
            .header h1 { color: #0f4c75; margin: 0; }
            .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .panel { background: #16213e; padding: 20px; border-radius: 10px; }
            .panel h2 { color: #3282b8; margin-top: 0; border-bottom: 1px solid #3282b8; padding-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #333; }
            th { color: #3282b8; }
            .ok { color: #00ff88; }
            .warn { color: #ffaa00; }
            .error { color: #ff4444; }
            .promoted { background: #004400; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ALPHA LOOP CAPITAL</h1>
            <p>Trading Dashboard - {{ timestamp }}</p>
        </div>
        <div class="grid">
            <div class="panel">
                <h2>Data Sources</h2>
                <table>
                    <tr><th>Source</th><th>API Key</th><th>Status</th></tr>
                    {% for source, configured in api_keys.items() %}
                    <tr>
                        <td>{{ source.upper() }}</td>
                        <td class="{{ 'ok' if configured else 'error' }}">{{ 'OK' if configured else 'MISSING' }}</td>
                        <td class="{{ 'ok' if configured else 'warn' }}">{{ 'Ready' if configured else 'Configure' }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            <div class="panel">
                <h2>Database</h2>
                <table>
                    <tr><th>Table</th><th>Rows</th></tr>
                    {% for table, count in db_stats.items() %}
                    {% if table != 'latest_price' %}
                    <tr><td>{{ table }}</td><td>{{ "{:,}".format(count) }}</td></tr>
                    {% endif %}
                    {% endfor %}
                </table>
            </div>
            <div class="panel">
                <h2>Models ({{ summary.promoted_models }}/{{ summary.total_models }} Promoted)</h2>
                <table>
                    <tr><th>Model</th><th>AUC</th><th>Accuracy</th><th>Status</th></tr>
                    {% for model in models[:10] %}
                    <tr class="{{ 'promoted' if model.promoted else '' }}">
                        <td>{{ model.name[:30] }}</td>
                        <td>{{ "%.3f"|format(model.auc) }}</td>
                        <td>{{ "%.3f"|format(model.accuracy) }}</td>
                        <td class="{{ 'ok' if model.promoted else 'warn' }}">{{ 'PROMOTED' if model.promoted else 'Pending' }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            <div class="panel">
                <h2>Trading Status</h2>
                <table>
                    <tr><th>Component</th><th>Status</th></tr>
                    <tr>
                        <td>IBKR Paper (7497)</td>
                        <td class="{{ 'ok' if ibkr.paper_connected else 'error' }}">
                            {{ 'CONNECTED' if ibkr.paper_connected else 'OFFLINE' }}
                        </td>
                    </tr>
                    <tr>
                        <td>IBKR Live (7496)</td>
                        <td class="{{ 'ok' if ibkr.live_connected else 'warn' }}">
                            {{ 'CONNECTED' if ibkr.live_connected else 'offline' }}
                        </td>
                    </tr>
                </table>
                <h3>Model Weights</h3>
                <table>
                    <tr><th>Model</th><th>Weight</th></tr>
                    {% for name, weight in weights.items() %}
                    <tr><td>{{ name[:25] }}</td><td>{{ "%.1f%%"|format(weight * 100) }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    @app.route("/")
    def dashboard():
        data_monitor = DataIngestionMonitor()
        model_monitor = ModelTrainingMonitor()
        trading_monitor = TradingMonitor()

        return render_template_string(
            HTML_TEMPLATE,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            api_keys=data_monitor.check_api_keys(),
            db_stats=data_monitor.get_database_stats(),
            models=model_monitor.get_model_inventory(),
            summary=model_monitor.get_training_summary(),
            weights=model_monitor.get_model_weights(),
            ibkr=trading_monitor.check_ibkr_connection(),
        )

    @app.route("/api/status")
    def api_status():
        data_monitor = DataIngestionMonitor()
        model_monitor = ModelTrainingMonitor()
        trading_monitor = TradingMonitor()

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "api_keys": data_monitor.check_api_keys(),
            "database": data_monitor.get_database_stats(),
            "models": model_monitor.get_training_summary(),
            "weights": model_monitor.get_model_weights(),
            "trading": trading_monitor.check_ibkr_connection(),
        })

    print(f"\n[+] Web dashboard starting at http://localhost:{port}")
    print("[+] Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Alpha Loop Capital Dashboard")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval (seconds)")
    parser.add_argument("--web", action="store_true", help="Launch web dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port")
    parser.add_argument("--simple", action="store_true", help="Use simple text dashboard")

    args = parser.parse_args()

    if args.web:
        create_web_dashboard(args.port)
    elif args.simple or not RICH_AVAILABLE:
        create_simple_dashboard(args.refresh)
    else:
        create_rich_dashboard(args.refresh)


if __name__ == "__main__":
    main()
