from .hoags_agent.hoags_agent import HoagsAgent
from .data_agent.data_agent import DataAgent
from .strategy_agent.strategy_agent import StrategyAgent
from .risk_agent.risk_agent import RiskAgent
from .execution_agent.execution_agent import ExecutionAgent
from .portfolio_agent.portfolio_agent import PortfolioAgent
from .research_agent.research_agent import ResearchAgent
from .compliance_agent.compliance_agent import ComplianceAgent
from .sentiment_agent.sentiment_agent import SentimentAgent
from .ghost_agent.ghost_agent import GhostAgent
from .orchestrator_agent.orchestrator_agent import OrchestratorAgent
from .swarm.swarm_factory import SwarmFactory
from .hackers.black_hat import BlackHatAgent
from .hackers.white_hat import WhiteHatAgent

# Define total agents constant
TOTAL_AGENTS = 76  # 1 Master + 8 Senior + 2 Hackers + 65 Specialized
