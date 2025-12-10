#!/usr/bin/env python
"""
ALC-Algo Setup Verification Script
Author: Tom Hogan | Alpha Loop Capital, LLC

Run this script to verify your ALC-Algo installation is complete and working.

Usage:
    python verify_setup.py

Requirements:
    - Python 3.10+
    - Virtual environment activated
    - Dependencies installed (pip install -r requirements.txt)
"""

import sys
import os
from pathlib import Path
from datetime import datetime


def print_header():
    """Print script header."""
    print()
    print("=" * 70)
    print("ALC-ALGO SETUP VERIFICATION")
    print("Alpha Loop Capital, LLC | Tom Hogan")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()


def check(name: str, condition: bool, details: str = "", critical: bool = False) -> bool:
    """
    Print check result.
    
    Args:
        name: Check name
        condition: Whether check passed
        details: Additional details on failure
        critical: Whether this check is critical
    
    Returns:
        Whether check passed
    """
    status = "✓" if condition else ("✗" if critical else "⚠")
    color = "\033[92m" if condition else ("\033[91m" if critical else "\033[93m")
    reset = "\033[0m"
    
    print(f"  {color}[{status}]{reset} {name}")
    if details and not condition:
        print(f"      └─ {details}")
    
    return condition


def check_python_version() -> bool:
    """Check Python version."""
    print("\n[1/8] Python Environment")
    
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 10
    
    check(
        f"Python 3.10+ (found: {version.major}.{version.minor}.{version.micro})",
        is_valid,
        "Install Python 3.10 or higher from python.org",
        critical=True
    )
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    check(
        "Virtual environment active",
        in_venv,
        "Activate venv: .\\venv\\Scripts\\Activate.ps1 (Windows) or source venv/bin/activate"
    )
    
    return is_valid


def check_core_imports() -> bool:
    """Check core package imports."""
    print("\n[2/8] Core Dependencies")
    
    results = []
    
    # Core data packages
    try:
        import pandas as pd
        import numpy as np
        results.append(check(f"Pandas {pd.__version__}, NumPy {np.__version__}", True))
    except ImportError as e:
        results.append(check("Pandas & NumPy", False, str(e), critical=True))
    
    # HTTP and async
    try:
        import requests
        import aiohttp
        results.append(check("Requests & Aiohttp", True))
    except ImportError as e:
        results.append(check("HTTP Libraries", False, str(e)))
    
    # Configuration
    try:
        from dotenv import load_dotenv
        import pydantic
        results.append(check("python-dotenv & Pydantic", True))
    except ImportError as e:
        results.append(check("Configuration Libraries", False, str(e)))
    
    return all(results)


def check_ml_imports() -> bool:
    """Check ML/AI package imports."""
    print("\n[3/8] AI/ML Dependencies")
    
    results = []
    
    try:
        import openai
        results.append(check("OpenAI", True))
    except ImportError:
        results.append(check("OpenAI", False, "pip install openai"))
    
    try:
        import anthropic
        results.append(check("Anthropic", True))
    except ImportError:
        results.append(check("Anthropic", False, "pip install anthropic"))
    
    try:
        import google.generativeai
        results.append(check("Google Generative AI", True))
    except ImportError:
        results.append(check("Google Generative AI", False, "pip install google-generativeai"))
    
    return any(results)  # At least one AI package should be available


def check_broker_imports() -> bool:
    """Check broker package imports."""
    print("\n[4/8] Broker Dependencies")
    
    results = []
    
    try:
        from ib_insync import IB
        results.append(check("IB-Insync (Interactive Brokers)", True))
    except ImportError:
        results.append(check("IB-Insync", False, "pip install ib_insync"))
    
    try:
        from alpha_vantage.timeseries import TimeSeries
        results.append(check("Alpha Vantage", True))
    except ImportError:
        results.append(check("Alpha Vantage", False, "pip install alpha-vantage"))
    
    return any(results)


def check_project_structure() -> bool:
    """Check project directory structure."""
    print("\n[5/8] Project Structure")
    
    results = []
    project_root = Path(__file__).parent
    
    # Core directories
    required_dirs = [
        "src",
        "src/agents",
        "src/core",
        "src/features",
        "src/backtest",
        "config",
        "tests",
        "data",
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        if not exists and "data" in dir_path:
            # Create data directories if missing
            full_path.mkdir(parents=True, exist_ok=True)
            exists = True
        results.append(check(f"Directory: {dir_path}", exists, f"Create: mkdir {dir_path}"))
    
    # Data subdirectories
    data_subdirs = ["data/raw", "data/processed", "data/portfolio", "data/logs", "data/models"]
    for subdir in data_subdirs:
        full_path = project_root / subdir
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
    
    return all(results)


def check_configuration() -> bool:
    """Check configuration files and settings."""
    print("\n[6/8] Configuration")
    
    results = []
    project_root = Path(__file__).parent
    
    # Config files
    config_files = [
        ("config/settings.py", True),
        ("config/secrets.py", False),  # Not required, but recommended
        (".env", False),
    ]
    
    for file_path, required in config_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        status = "Required" if required else "Optional"
        results.append(check(
            f"{file_path} ({status})",
            exists or not required,
            f"Copy from {file_path}.example" if not exists else ""
        ))
    
    # Try to load settings
    try:
        sys.path.insert(0, str(project_root))
        from config.settings import settings
        results.append(check("Settings module loads", True))
        
        # Check specific keys
        has_alpha_vantage = bool(settings.alpha_vantage_api_key)
        results.append(check(
            "Alpha Vantage API Key",
            has_alpha_vantage,
            "Set ALPHA_VANTAGE_API_KEY in .env or config/secrets.py"
        ))
        
    except Exception as e:
        results.append(check("Settings module", False, str(e), critical=True))
    
    return all(results)


def check_agent_system() -> bool:
    """Check agent system imports."""
    print("\n[7/8] Agent System")
    
    results = []
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Core agent base
    try:
        from src.core.agent_base import BaseAgent, AgentTier, AgentToughness
        results.append(check("BaseAgent class", True))
        results.append(check("Agent enums (Tier, Toughness)", True))
    except ImportError as e:
        results.append(check("Agent base module", False, str(e), critical=True))
    
    # Agent imports
    try:
        from src.agents import (
            GhostAgent, DataAgent, StrategyAgent,
            RiskAgent, ExecutionAgent, TOTAL_AGENTS
        )
        results.append(check(f"Senior agents ({TOTAL_AGENTS} total available)", True))
    except ImportError as e:
        results.append(check("Agent imports", False, str(e)))
    
    # Technical indicators
    try:
        from src.features.technical import TechnicalIndicators
        results.append(check("Technical indicators module", True))
    except ImportError as e:
        results.append(check("Technical indicators", False, str(e)))
    
    # Backtest engine
    try:
        from src.backtesting.backtest_engine import BacktestEngine
        results.append(check("Backtest engine", True))
    except ImportError as e:
        results.append(check("Backtest engine", False, str(e)))
    
    return all(results)


def check_external_connections() -> bool:
    """Check external service connections."""
    print("\n[8/8] External Connections")
    
    results = []
    
    # Internet connectivity
    try:
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        results.append(check("Internet connectivity", response.status_code == 200))
    except Exception:
        results.append(check("Internet connectivity", False, "Check network connection"))
    
    # Alpha Vantage API (if key is set)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config.settings import settings
        
        if settings.alpha_vantage_api_key:
            import requests
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={settings.alpha_vantage_api_key}"
            response = requests.get(url, timeout=10)
            has_data = "Global Quote" in response.text and "Error" not in response.text
            results.append(check("Alpha Vantage API", has_data, "Check API key validity"))
        else:
            results.append(check("Alpha Vantage API", False, "No API key configured"))
    except Exception as e:
        results.append(check("Alpha Vantage API", False, str(e)))
    
    # IBKR (just check if we can import - actual connection requires TWS running)
    try:
        from ib_insync import IB
        results.append(check("IBKR client available (connect requires TWS)", True))
    except ImportError:
        results.append(check("IBKR client", False, "pip install ib_insync"))
    
    return all(results)


def run_quick_test() -> bool:
    """Run a quick functional test."""
    print("\n[BONUS] Quick Functional Test")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 101 + np.random.randn(100).cumsum(),
            'Low': 99 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 100),
        }, index=dates)
        
        # Fix OHLC relationships
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Test technical indicators
        sys.path.insert(0, str(Path(__file__).parent))
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(data)
        sma = ti.sma(window=20)
        rsi = ti.rsi(window=14)
        
        check("Sample data creation", True)
        check("SMA calculation", len(sma) == 100)
        check("RSI calculation", len(rsi) == 100)
        check("RSI values in range [0, 100]", rsi.dropna().between(0, 100).all())
        
        return True
        
    except Exception as e:
        check("Quick functional test", False, str(e))
        return False


def main():
    """Run all verification checks."""
    print_header()
    
    results = []
    
    # Run all checks
    results.append(("Python Environment", check_python_version()))
    results.append(("Core Dependencies", check_core_imports()))
    results.append(("AI/ML Dependencies", check_ml_imports()))
    results.append(("Broker Dependencies", check_broker_imports()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Configuration", check_configuration()))
    results.append(("Agent System", check_agent_system()))
    results.append(("External Connections", check_external_connections()))
    results.append(("Functional Test", run_quick_test()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color = "\033[92m" if result else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {name}")
    
    print()
    
    if passed == total:
        print("\033[92m" + "=" * 70)
        print("ALL CHECKS PASSED!")
        print("Your ALC-Algo setup is complete and ready to use.")
        print("=" * 70 + "\033[0m")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Review docs/COMPLETE_SETUP_INSTRUCTIONS.md")
        print("  3. Configure your API keys in config/secrets.py")
    else:
        print("\033[93m" + "=" * 70)
        print(f"{passed}/{total} CHECKS PASSED")
        print("Please address the failed checks above.")
        print("=" * 70 + "\033[0m")
        print("\nFor detailed instructions, see:")
        print("  docs/COMPLETE_SETUP_INSTRUCTIONS.md")
    
    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

