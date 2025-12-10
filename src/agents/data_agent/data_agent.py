"""
================================================================================
DATA AGENT - Data Ingestion & Normalization
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Equity and options ingestion (Yahoo/Alpha Vantage)
- Data normalization and cleaning
- Cache management

Tier: SENIOR (2)
Reports To: HOAGS
Cluster: data

Core Philosophy:
"Garbage in, garbage out. Data quality is everything."

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT DATA_AGENT DOES:
    DATA_AGENT is the data backbone of Alpha Loop Capital. Every agent
    that needs market data, financial data, or options data goes through
    DATA_AGENT. It handles ingestion, normalization, cleaning, and caching.

    Think of DATA_AGENT as the "data engineer" who ensures all other
    agents get clean, consistent, reliable data. Without good data,
    all the sophisticated algorithms are worthless.

KEY FUNCTIONS:
    1. fetch_equity_snapshot() - Gets comprehensive equity data including
       price history, volume, and optionally options and intraday data.

    2. fetch_options_chain() - Gets full options chain for a ticker
       including all strikes, expirations, greeks, and volume.

    3. fetch_alpha_vantage_daily() - Gets daily data from Alpha Vantage
       API for fundamental analysis.

    4. ingest_universe() - Batch ingestion of multiple tickers
       for portfolio-wide analysis.

    5. process() - Main entry point. Routes tasks to appropriate
       data fetching methods.

DATA SOURCES:
    - Yahoo Finance: Primary for price data, options chains
    - Alpha Vantage: Fundamentals, intraday data
    - FinViz: Screening data
    - Fiscal AI: Alternative data (future integration)

RELATIONSHIPS WITH OTHER AGENTS:
    - ALL AGENTS: DATA_AGENT serves every agent that needs market data.
      SCOUT, BOOKMAKER, HUNTER, strategy agents all depend on it.

    - SCOUT: Provides real-time options data for arbitrage detection.

    - BOOKMAKER: Provides fundamental data for valuation analysis.

    - CONVERSION_REVERSAL: Provides options chain data for arb detection.

    - KILLJOY: Provides volatility and correlation data for risk calcs.

PATHS OF GROWTH/TRANSFORMATION:
    1. REAL-TIME STREAMING: Move from polling to streaming data for
       faster reaction times.

    2. ALTERNATIVE DATA: Integrate satellite imagery, social media
       sentiment, web traffic, etc.

    3. DATA QUALITY MONITORING: Active monitoring for stale, missing,
       or anomalous data.

    4. INTELLIGENT CACHING: Cache data based on access patterns and
       value of recency.

    5. DATA PROVENANCE: Track data lineage for audit and debugging.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train DATA_AGENT individually:
    python -m src.training.agent_training_utils --agent DATA_AGENT

    # Train with dependent agents:
    python -m src.training.agent_training_utils --agents DATA_AGENT,SCOUT,BOOKMAKER

RUNNING THE AGENT:
    from src.agents.data_agent.data_agent import DataAgent

    data_agent = DataAgent()

    # Fetch equity snapshot with options
    result = data_agent.process({
        "type": "fetch_equity_snapshot",
        "ticker": "AAPL",
        "include_options": True,
        "include_intraday": True
    })

    # Fetch options chain
    result = data_agent.process({
        "type": "fetch_options_chain",
        "ticker": "NVDA"
    })

    # Batch ingest universe
    result = data_agent.process({
        "type": "ingest_universe",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "period": "1y",
        "include_options": True
    })

================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List
from src.data_ingestion.equity_option_pipeline import EquityOptionPipeline


class DataAgent(BaseAgent):
    """
    Senior Agent - Data Ingestion & Normalization

    Handles all data ingestion from external APIs and normalizes
    data for use by other agents.
    """

    def __init__(self, user_id: str = "TJH"):
        """Initialize DataAgent."""
        super().__init__(
            name="DataAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                "alpha_vantage_api",
                "finviz_scraping",
                "fiscal_ai_api",
                "data_normalization",
                "cache_management",
                "yahoo_finance_fetch",
                "options_chain_ingestion",
                "alpha_vantage_intraday",
            ],
            user_id=user_id
        )
        self.pipeline = EquityOptionPipeline()
        self.logger.info("DataAgent initialized with equity/options pipeline")

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data ingestion task. Supports equities, options, and batch ingestion.

        Args:
            task: Task dictionary with data source and parameters

        Returns:
            Processed data
        """
        task_type = task.get('type', 'fetch_data')
        source = task.get('source')
        ticker = task.get('ticker')
        include_options = bool(task.get('include_options', False))
        include_intraday = bool(task.get('include_intraday', True))
        period = task.get('period', '1y')

        if task_type in ('fetch_data', 'fetch_equity_snapshot'):
            return self.pipeline.fetch_equity_snapshot(
                ticker=ticker,
                period=period,
                include_options=include_options,
                include_intraday=include_intraday,
            )

        if task_type == 'fetch_options_chain':
            return self.pipeline.fetch_options_chain(ticker=ticker)

        if task_type == 'alpha_vantage_daily' or source == 'alpha_vantage':
            outputsize = task.get('outputsize', 'compact')
            return self.pipeline.fetch_alpha_vantage_daily(
                ticker=ticker,
                outputsize=outputsize,
            )

        if task_type == 'ingest_universe':
            tickers = task.get('tickers') or []
            return self.pipeline.ingest_universe(
                tickers=tickers,
                period=period,
                include_options=include_options,
            )

        return {'success': False, 'error': f'Unknown task type: {task_type}'}

    def get_capabilities(self) -> List[str]:
        """Return DataAgent capabilities."""
        return self.capabilities
