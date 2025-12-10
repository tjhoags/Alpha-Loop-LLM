#!/usr/bin/env python
"""
Test Critical Imports After Codebase Consolidation
Verifies all major modules import correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all critical module imports"""
    print("\n" + "="*70)
    print("TESTING CRITICAL IMPORTS AFTER CONSOLIDATION")
    print("="*70 + "\n")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Backtesting Framework
    try:
        from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestMode
        print("[OK] Backtesting framework: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Backtesting framework: FAIL - {e}")
        tests_failed += 1

    # Test 2: Data Ingestion
    try:
        from src.ingest.alpha_vantage import AlphaVantageClient
        from src.ingest.yahoo_finance import YahooFinanceClient
        from src.ingest.collector import MarketDataCollector
        print("[OK] Data ingestion (ingest/): PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Data ingestion: FAIL - {e}")
        tests_failed += 1

    # Test 3: Analytics
    try:
        from src.analytics.performance_attribution import PerformanceAttributionEngine
        from src.analytics.regime_detection import MarketRegimeDetector
        print("[OK] Analytics modules: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Analytics: FAIL - {e}")
        tests_failed += 1

    # Test 4: Portfolio Optimization
    try:
        from src.portfolio.optimization_engine import PortfolioOptimizationEngine
        print("[OK] Portfolio optimization: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Portfolio optimization: FAIL - {e}")
        tests_failed += 1

    # Test 5: Configuration
    try:
        from config import settings
        # Just test that we can import it
        print(f"[OK] Configuration: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Configuration: FAIL - {e}")
        tests_failed += 1

    # Test 6: Agents
    try:
        from src.agents.compliance_agent.compliance_agent import ComplianceAgent
        from src.agents.portfolio_agent.portfolio_agent import PortfolioAgent
        print("[OK] Agent framework: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Agents: FAIL - {e}")
        tests_failed += 1

    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("="*70 + "\n")

    return tests_failed == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
