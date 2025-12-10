"""
Azure Application Insights Integration
Real-time monitoring and telemetry for ALC-Algo trading system

Features:
- Custom event tracking (trades, signals, errors)
- Performance metrics (latency, throughput)
- Exception tracking with context
- Custom dimensions for filtering
- Request tracking for API calls

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import time

# Try to import Application Insights SDK
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    from opencensus.ext.azure import metrics_exporter
    from opencensus.stats import aggregation as aggregation_module
    from opencensus.stats import measure as measure_module
    from opencensus.stats import stats as stats_module
    from opencensus.stats import view as view_module
    from opencensus.tags import tag_map as tag_map_module
    AZURE_INSIGHTS_AVAILABLE = True
except ImportError:
    AZURE_INSIGHTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AzureInsightsTracker:
    """
    Azure Application Insights integration for real-time monitoring.

    Usage:
        tracker = AzureInsightsTracker(connection_string=os.getenv("AZURE_INSIGHTS_CONNECTION_STRING"))

        # Track custom events
        tracker.track_event("TradeExecuted", {"symbol": "AAPL", "quantity": 100})

        # Track metrics
        tracker.track_metric("PortfolioValue", 125000.50)

        # Track exceptions
        try:
            risky_operation()
        except Exception as e:
            tracker.track_exception(e, {"operation": "portfolio_rebalance"})
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        enable_local_logging: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize Azure Insights tracker.

        Args:
            connection_string: Azure Application Insights connection string
            enable_local_logging: Also log to local files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.connection_string = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.enabled = AZURE_INSIGHTS_AVAILABLE and self.connection_string is not None
        self.enable_local_logging = enable_local_logging

        # Set up logging
        self.logger = logging.getLogger("ALC.Azure")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        if self.enabled:
            try:
                # Configure Azure log handler
                azure_handler = AzureLogHandler(connection_string=self.connection_string)
                self.logger.addHandler(azure_handler)

                # Configure metrics exporter
                self.metrics_exporter = metrics_exporter.new_metrics_exporter(
                    connection_string=self.connection_string
                )

                self.logger.info("Azure Application Insights tracking enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure Insights: {e}")
                self.enabled = False
        else:
            if not AZURE_INSIGHTS_AVAILABLE:
                self.logger.warning(
                    "Azure Insights SDK not installed. "
                    "Install with: pip install opencensus-ext-azure"
                )
            elif not self.connection_string:
                self.logger.warning(
                    "Azure Insights connection string not provided. "
                    "Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable"
                )

    def track_event(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None,
        measurements: Optional[Dict[str, float]] = None
    ):
        """
        Track a custom event.

        Args:
            event_name: Name of the event (e.g., "TradeExecuted", "SignalGenerated")
            properties: String properties (dimensions) for filtering
            measurements: Numeric measurements
        """
        properties = properties or {}
        measurements = measurements or {}

        # Add timestamp
        properties["timestamp"] = datetime.now().isoformat()

        if self.enabled:
            # Log as custom event to Azure
            self.logger.info(
                f"Event: {event_name}",
                extra={
                    "custom_dimensions": {
                        **properties,
                        **{f"measurement_{k}": v for k, v in measurements.items()}
                    }
                }
            )

        if self.enable_local_logging:
            self.logger.info(f"Event: {event_name} | {properties} | {measurements}")

    def track_metric(
        self,
        metric_name: str,
        value: float,
        properties: Optional[Dict[str, str]] = None
    ):
        """
        Track a numeric metric.

        Args:
            metric_name: Name of the metric (e.g., "PortfolioValue", "SharpeRatio")
            value: Numeric value
            properties: Additional properties for filtering
        """
        properties = properties or {}

        if self.enabled:
            self.logger.info(
                f"Metric: {metric_name}={value}",
                extra={
                    "custom_dimensions": {
                        "metric_name": metric_name,
                        "metric_value": value,
                        **properties
                    }
                }
            )

        if self.enable_local_logging:
            self.logger.info(f"Metric: {metric_name}={value} | {properties}")

    def track_exception(
        self,
        exception: Exception,
        properties: Optional[Dict[str, str]] = None,
        measurements: Optional[Dict[str, float]] = None
    ):
        """
        Track an exception with context.

        Args:
            exception: The exception instance
            properties: Additional context properties
            measurements: Related measurements
        """
        properties = properties or {}
        measurements = measurements or {}

        properties["exception_type"] = type(exception).__name__
        properties["exception_message"] = str(exception)
        properties["timestamp"] = datetime.now().isoformat()

        if self.enabled:
            self.logger.exception(
                f"Exception: {exception}",
                extra={
                    "custom_dimensions": {
                        **properties,
                        **{f"measurement_{k}": v for k, v in measurements.items()}
                    }
                }
            )

        if self.enable_local_logging:
            self.logger.exception(f"Exception: {exception} | {properties}")

    def track_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
        pnl: Optional[float] = None
    ):
        """
        Convenience method to track trade execution.

        Args:
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares/contracts
            price: Execution price
            strategy: Strategy name
            pnl: Realized P&L (if closing position)
        """
        measurements = {
            "quantity": quantity,
            "price": price,
        }
        if pnl is not None:
            measurements["pnl"] = pnl

        self.track_event(
            event_name="TradeExecuted",
            properties={
                "symbol": symbol,
                "side": side,
                "strategy": strategy,
            },
            measurements=measurements
        )

    def track_performance(
        self,
        strategy: str,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: Optional[float] = None
    ):
        """
        Convenience method to track strategy performance.

        Args:
            strategy: Strategy name
            total_return: Total return (%)
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown (%)
            win_rate: Win rate (%)
        """
        measurements = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }
        if win_rate is not None:
            measurements["win_rate"] = win_rate

        self.track_event(
            event_name="PerformanceUpdate",
            properties={"strategy": strategy},
            measurements=measurements
        )

    def time_operation(self, operation_name: str):
        """
        Decorator to track operation execution time.

        Usage:
            @tracker.time_operation("backtest_execution")
            def run_backtest():
                ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time

                    self.track_metric(
                        f"{operation_name}_duration_seconds",
                        duration,
                        properties={"operation": operation_name, "status": "success"}
                    )

                    return result
                except Exception as e:
                    duration = time.time() - start_time

                    self.track_exception(
                        e,
                        properties={
                            "operation": operation_name,
                            "duration_seconds": str(duration)
                        }
                    )
                    raise
            return wrapper
        return decorator


# Global singleton instance
_global_tracker: Optional[AzureInsightsTracker] = None


def get_tracker() -> AzureInsightsTracker:
    """Get or create the global Azure Insights tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AzureInsightsTracker()
    return _global_tracker


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = AzureInsightsTracker()

    # Track events
    tracker.track_event("SystemStartup", {"version": "1.0.0", "environment": "production"})

    # Track metrics
    tracker.track_metric("PortfolioValue", 125000.50, {"account": "main"})

    # Track trades
    tracker.track_trade(
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.25,
        strategy="momentum",
        pnl=None
    )

    # Track performance
    tracker.track_performance(
        strategy="momentum",
        total_return=15.5,
        sharpe_ratio=2.1,
        max_drawdown=-8.5,
        win_rate=62.5
    )

    # Track exceptions
    try:
        raise ValueError("Example error")
    except Exception as e:
        tracker.track_exception(e, {"component": "example"})

    print("Azure Insights tracking demo complete")
