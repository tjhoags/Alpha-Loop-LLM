"""
SLEUTH MASTER - INTEGRATED OPTIMIZATION ENGINE
===============================================
Alpha Loop Capital - Consequence Engine

The "Secondary Tom" - An algorithmic co-pilot that:
1. Continuously monitors all portfolio optimization vectors
2. Integrates signals from all 4 Consequence Engine modules
3. Generates prioritized action lists
4. Maximizes risk-adjusted returns through systematic optimization

Author: Tom Hogan
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import logging

from .chain_mapper import ChainMapper, ChainNode, OrderLevel
from .passive_flow import PassiveFlowDetector, FlowRegime
from .narrative_tracker import NarrativeTracker, NarrativeStage
from .liquidity_distortion import LiquidityDistortionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionPriority(Enum):
    """Priority levels for actions"""
    CRITICAL = "critical"      # Do immediately
    HIGH = "high"              # Do today
    MEDIUM = "medium"          # Do this week
    LOW = "low"                # Monitor


class ActionCategory(Enum):
    """Categories of portfolio actions"""
    TAX = "tax"
    RISK = "risk"
    SIZING = "sizing"
    OPTIONS = "options"
    ALPHA = "alpha"
    REBALANCE = "rebalance"


@dataclass
class PortfolioAction:
    """A single recommended portfolio action"""
    priority: ActionPriority
    category: ActionCategory
    ticker: str
    action: str
    description: str
    value_estimate: float = 0.0
    confidence: float = 0.5
    source_module: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "priority": self.priority.value,
            "category": self.category.value,
            "ticker": self.ticker,
            "action": self.action,
            "description": self.description,
            "value_estimate": self.value_estimate,
            "confidence": self.confidence,
            "source": self.source_module
        }


class SleuthMaster:
    """
    MASTER SLEUTH ORCHESTRATOR
    
    The "Secondary Tom" that integrates all modules and
    generates unified, prioritized action lists.
    """
    
    def __init__(self):
        self.chain_mapper = ChainMapper()
        self.flow_detector = PassiveFlowDetector()
        self.narrative_tracker = NarrativeTracker()
        self.distortion_detector = LiquidityDistortionDetector()
        
        self.actions: List[PortfolioAction] = []
        self.last_scan_time: Optional[str] = None
    
    def run_morning_scan(self) -> Dict:
        """
        Run complete morning scan across all modules.
        The "wake up and tell me what to do" function.
        """
        self.actions = []
        self.last_scan_time = datetime.now().isoformat()
        
        # Collect signals from all modules
        flow_signal = self.flow_detector.get_trading_signal()
        narrative_signals = self.narrative_tracker.get_all_signals()
        
        # Generate actions from each module
        self._process_flow_signals(flow_signal)
        self._process_narrative_signals(narrative_signals)
        self._process_chain_signals()
        
        # Sort by priority
        priority_order = {
            ActionPriority.CRITICAL: 0,
            ActionPriority.HIGH: 1,
            ActionPriority.MEDIUM: 2,
            ActionPriority.LOW: 3
        }
        self.actions.sort(key=lambda x: priority_order[x.priority])
        
        return {
            "scan_time": self.last_scan_time,
            "action_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions[:10]],
            "flow_regime": flow_signal.get("regime", "unknown"),
            "stress_index": flow_signal.get("stress_index", 50),
            "summary": self._generate_summary()
        }
    
    def _process_flow_signals(self, flow_signal: Dict) -> None:
        """Process passive flow signals into actions"""
        signal = flow_signal.get("signal", "NEUTRAL")
        
        if signal == "DEFENSIVE":
            self.actions.append(PortfolioAction(
                priority=ActionPriority.HIGH,
                category=ActionCategory.RISK,
                ticker="PORTFOLIO",
                action="REDUCE_EXPOSURE",
                description="Reduce equity exposure 20-30% - flow regime defensive",
                confidence=flow_signal.get("confidence", 50) / 100,
                source_module="passive_flow"
            ))
        elif signal == "RISK_ON":
            self.actions.append(PortfolioAction(
                priority=ActionPriority.MEDIUM,
                category=ActionCategory.ALPHA,
                ticker="PORTFOLIO",
                action="ADD_EXPOSURE",
                description="Flow regime supportive - can add risk",
                confidence=flow_signal.get("confidence", 50) / 100,
                source_module="passive_flow"
            ))
    
    def _process_narrative_signals(self, narrative_signals: List[Dict]) -> None:
        """Process narrative signals into actions"""
        for sig in narrative_signals:
            signal = sig.get("signal", "")
            
            if signal == "entry":
                for ticker in sig.get("primary_tickers", [])[:3]:
                    self.actions.append(PortfolioAction(
                        priority=ActionPriority.MEDIUM,
                        category=ActionCategory.ALPHA,
                        ticker=ticker,
                        action="BUY",
                        description=f"Narrative early stage: {sig.get('narrative_name', '')}",
                        confidence=sig.get("confidence", 50) / 100,
                        source_module="narrative_tracker"
                    ))
            elif signal == "exit":
                for ticker in sig.get("primary_tickers", []):
                    self.actions.append(PortfolioAction(
                        priority=ActionPriority.HIGH,
                        category=ActionCategory.RISK,
                        ticker=ticker,
                        action="SELL",
                        description=f"Narrative saturated - EXIT: {sig.get('narrative_name', '')}",
                        confidence=sig.get("confidence", 50) / 100,
                        source_module="narrative_tracker"
                    ))
    
    def _process_chain_signals(self) -> None:
        """Process chain mapper signals into actions"""
        for chain_name, chain in self.chain_mapper.chains.items():
            recommendations = self.chain_mapper.get_recommendations(chain, min_tom_score=70)
            
            for rec in recommendations[:3]:
                self.actions.append(PortfolioAction(
                    priority=ActionPriority.MEDIUM,
                    category=ActionCategory.ALPHA,
                    ticker=rec["ticker"],
                    action="BUY" if rec["tom_score"] >= 75 else "WATCHLIST",
                    description=f"Order {rec['order_level']} beneficiary - Tom Score {rec['tom_score']}",
                    confidence=rec["tom_score"] / 100,
                    source_module="chain_mapper"
                ))
    
    def _generate_summary(self) -> Dict:
        """Generate executive summary"""
        critical_count = len([a for a in self.actions if a.priority == ActionPriority.CRITICAL])
        high_count = len([a for a in self.actions if a.priority == ActionPriority.HIGH])
        
        return {
            "status": "ATTENTION_NEEDED" if critical_count > 0 else "NORMAL",
            "critical_actions": critical_count,
            "high_priority_actions": high_count,
            "total_actions": len(self.actions),
            "top_action": self.actions[0].to_dict() if self.actions else None
        }
    
    def get_daily_brief(self) -> str:
        """Generate human-readable daily brief"""
        report = self.run_morning_scan()
        
        lines = [
            "=" * 60,
            "SLEUTH MASTER - DAILY BRIEF",
            f"Alpha Loop Capital | {datetime.now().strftime('%B %d, %Y %H:%M')}",
            "=" * 60,
            "",
            f"📊 STATUS: {report['summary']['status']}",
            f"⚠️ ACTIONS: {report['action_count']} total",
            f"   Critical: {report['summary']['critical_actions']}",
            f"   High: {report['summary']['high_priority_actions']}",
            "",
            "🎯 PRIORITY ACTIONS:"
        ]
        
        for i, action in enumerate(report['actions'][:5], 1):
            lines.append(f"   {i}. [{action['category'].upper()}] {action['ticker']}: {action['action']}")
            lines.append(f"      {action['description']}")
        
        lines.extend([
            "",
            f"📈 FLOW REGIME: {report['flow_regime']}",
            f"⚡ STRESS INDEX: {report['stress_index']:.1f}/100",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    master = SleuthMaster()
    print(master.get_daily_brief())

