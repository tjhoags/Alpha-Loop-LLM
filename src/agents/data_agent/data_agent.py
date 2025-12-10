"""
DataAgent - Handles data ingestion and normalization
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Equity and options ingestion (Yahoo/Alpha Vantage)
- Data normalization and cleaning
- Cache management
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
