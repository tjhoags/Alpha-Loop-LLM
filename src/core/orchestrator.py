"""================================================================================
SYSTEM ORCHESTRATOR - Dynamic Resource Allocation
================================================================================
Manages system resources based on market regime and system load.
Adjusts polling intervals and pauses non-essential tasks during high volatility.
================================================================================
"""

import psutil
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.sources.fred import FredClient


class SystemOrchestrator:
    """The 'Boss' of the system.
    Dynamically reallocates resources based on Market Regime and System Load.
    """

    def __init__(self):
        self.settings = get_settings()
        self.fred = FredClient()
        self.current_regime = "NORMAL"
        self.polling_intervals = {
            "risk": 60,
            "execution": 60,
            "training": 3600,
        }

    def detect_market_regime(self) -> str:
        """Check VIX (Volatility) to determine system urgency.
        """
        try:
            # Quick check on VIX level (simplified)
            # In production, this would read from the live data feed
            vix_df = self.fred.fetch_series("VIXCLS")
            if not vix_df.empty:
                last_vix = vix_df.iloc[-1]["value"]
                if last_vix > 30:
                    return "CRISIS"
                elif last_vix > 20:
                    return "VOLATILE"
        except Exception:
            pass
        return "NORMAL"

    def optimize_system(self):
        """Adjusts polling rates and pauses non-essential tasks based on regime.
        """
        regime = self.detect_market_regime()
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        logger.info(f"Orchestrator Status: Regime={regime} | CPU={cpu_usage}% | RAM={ram_usage}%")

        if regime == "CRISIS":
            logger.warning("🚨 MARKET IN CRISIS MODE. Optimizing for SURVIVAL.")
            # 1. Accelerate Risk Checks
            self.polling_intervals["risk"] = 5  # Check every 5 seconds
            # 2. Accelerate Execution
            self.polling_intervals["execution"] = 10
            # 3. Kill Background Tasks
            self.pause_task("training")
            self.pause_task("research")

        elif regime == "VOLATILE":
            logger.info("⚠️ High Volatility. Increasing responsiveness.")
            self.polling_intervals["risk"] = 30
            self.polling_intervals["execution"] = 30

        else: # NORMAL
            # Optimize for long-term alpha (Research/Training)
            self.polling_intervals["risk"] = 60
            self.polling_intervals["execution"] = 60
            self.resume_task("training")
            self.resume_task("research")

        # Resource Guard
        if ram_usage > 90:
            logger.critical("Memory critical! Garbage collecting and pausing ingestion.")
            self.pause_task("ingestion")

    def pause_task(self, task_name: str):
        # Implementation would signal the specific thread/process to sleep
        logger.info(f"Orchestrator: Pausing {task_name} to free resources.")

    def resume_task(self, task_name: str):
        logger.info(f"Orchestrator: Resuming {task_name}.")

    def get_interval(self, task_name: str) -> int:
        return self.polling_intervals.get(task_name, 60)

# Global Orchestrator
orchestrator = SystemOrchestrator()


