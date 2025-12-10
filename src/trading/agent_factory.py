"""
================================================================================
AGENT FACTORY - Agent Create Agent (ACA) System
================================================================================
Intelligent meta-agent system that:
1. Monitors all trading agent performance
2. Spawns new specialized agents when needed
3. Terminates underperforming agents
4. Mutates hyperparameters for evolution
5. Orchestrates multi-strategy portfolios

AGENT TYPES:
- MomentumAgent: Trend-following strategies
- MeanReversionAgent: Statistical arbitrage
- SentimentAgent: NLP-driven trading
- MacroAgent: Economic indicator based
- VolatilityAgent: Options/VIX strategies
- LiquidityAgent: Market microstructure alpha

================================================================================
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import get_settings


class AgentType(Enum):
    """Types of trading agents."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    ENSEMBLE = "ensemble"


class AgentStatus(Enum):
    """Agent lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


@dataclass
class AgentConfig:
    """Configuration for a trading agent."""
    agent_id: str
    agent_type: AgentType
    symbol: str
    
    # Hyperparameters
    lookback_window: int = 60
    prediction_horizon: int = 1
    confidence_threshold: float = 0.6
    
    # Model parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.01
    
    # Feature flags
    use_momentum_features: bool = True
    use_mean_reversion_features: bool = True
    use_volume_features: bool = True
    use_sentiment_features: bool = False
    use_macro_features: bool = False
    
    # Risk parameters
    max_position_pct: float = 0.05
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None  # If spawned from another agent
    generation: int = 1  # Evolution generation


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    trades_count: int = 0
    accuracy: float = 0.0
    auc_score: float = 0.0
    
    @property
    def fitness_score(self) -> float:
        """
        Composite fitness score for agent evaluation.
        Used for survival-of-the-fittest agent selection.
        """
        # Weighted combination of metrics
        score = (
            0.30 * self.sharpe_ratio +
            0.20 * self.sortino_ratio +
            0.15 * self.win_rate +
            0.15 * self.profit_factor +
            0.10 * (1 - abs(self.max_drawdown)) +  # Penalize drawdown
            0.10 * self.accuracy
        )
        return max(0, score)
    
    @property
    def is_viable(self) -> bool:
        """Minimum viability check."""
        return (
            self.sharpe_ratio > 0.5 and
            self.accuracy > 0.52 and
            abs(self.max_drawdown) < 0.10 and
            self.trades_count >= 10
        )


class TradingAgent:
    """
    Individual trading agent with its own strategy and parameters.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.TRAINING
        self.performance = AgentPerformance(agent_id=config.agent_id)
        self.model = None
        self.trade_history: List[Dict] = []
        
        logger.info(f"Agent {config.agent_id} created | Type: {config.agent_type.value} | Symbol: {config.symbol}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the agent's model."""
        from src.ml.models import build_models, time_series_cv
        
        logger.info(f"Training agent {self.config.agent_id}...")
        self.status = AgentStatus.TRAINING
        
        # Get base models
        models = build_models(random_state=hash(self.config.agent_id) % 2**31)
        
        # Modify based on agent config
        for name, pipeline in models.items():
            # Adjust hyperparameters from config
            model = pipeline.named_steps['model']
            if hasattr(model, 'n_estimators'):
                model.n_estimators = self.config.n_estimators
            if hasattr(model, 'max_depth'):
                model.max_depth = self.config.max_depth
            if hasattr(model, 'learning_rate'):
                model.learning_rate = self.config.learning_rate
        
        # Cross-validate
        best_model_name = None
        best_metrics = None
        best_score = -np.inf
        
        for name, model in models.items():
            metrics = time_series_cv(model, X, y, splits=5)
            fitness = 0.5 * metrics['auc'] + 0.3 * metrics['accuracy'] + 0.2 * metrics['precision']
            
            if fitness > best_score:
                best_score = fitness
                best_model_name = name
                best_metrics = metrics
                self.model = model
        
        if best_metrics:
            self.performance.accuracy = best_metrics['accuracy']
            self.performance.auc_score = best_metrics['auc']
        
        # Final fit on all data
        if self.model:
            self.model.fit(X, y)
            self.status = AgentStatus.VALIDATING
            logger.info(f"Agent {self.config.agent_id} trained | Best: {best_model_name} | AUC: {best_metrics['auc']:.3f}")
        
        return best_metrics or {}
    
    def predict(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate prediction.
        Returns (direction, confidence) where direction is 1 (long) or -1 (short).
        """
        if self.model is None:
            return 0, 0.0
        
        prob = self.model.predict_proba(X)[:, 1][-1]
        
        if prob > self.config.confidence_threshold:
            return 1, prob
        elif prob < (1 - self.config.confidence_threshold):
            return -1, 1 - prob
        else:
            return 0, prob
    
    def record_trade(self, trade: Dict) -> None:
        """Record a trade for performance tracking."""
        self.trade_history.append(trade)
        
        # Update performance metrics
        if trade.get('pnl'):
            self.performance.total_pnl += trade['pnl']
            self.performance.trades_count += 1
            
            wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            self.performance.win_rate = wins / len(self.trade_history)


class MetaAgent:
    """
    The 'Manager' Agent - Orchestrates the agent ecosystem.
    
    Responsibilities:
    1. Monitor performance of all active trading agents
    2. ACA: Spawn new agents when needed (Agents Creating Agents)
    3. Terminate underperforming agents
    4. Mutate successful agents to explore parameter space
    5. Allocate capital across agents based on performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.agents: Dict[str, TradingAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.performance_history: List[Dict] = []
        
        # Evolution parameters
        self.mutation_rate = 0.2
        self.population_size = 10
        self.elite_fraction = 0.2
        
        # Config storage
        self.config_path = self.settings.base_dir / "agent_configs"
        self.config_path.mkdir(exist_ok=True)
        
        logger.info("MetaAgent initialized - ACA system ready")
    
    # =========================================================================
    # AGENT LIFECYCLE
    # =========================================================================
    
    def create_agent(
        self,
        symbol: str,
        agent_type: AgentType = AgentType.ENSEMBLE,
        parent_config: Optional[AgentConfig] = None
    ) -> TradingAgent:
        """
        Create a new trading agent.
        If parent_config is provided, mutate from parent.
        """
        agent_id = f"{symbol}_{agent_type.value}_{uuid.uuid4().hex[:8]}"
        
        if parent_config:
            # Mutate from parent
            config = self._mutate_config(parent_config, agent_id)
            config.generation = parent_config.generation + 1
            config.parent_id = parent_config.agent_id
        else:
            # Fresh agent
            config = AgentConfig(
                agent_id=agent_id,
                agent_type=agent_type,
                symbol=symbol,
            )
        
        agent = TradingAgent(config)
        self.agents[agent_id] = agent
        self.agent_configs[agent_id] = config
        
        # Save config
        self._save_config(config)
        
        logger.info(f"Created agent {agent_id} | Gen: {config.generation}")
        return agent
    
    def terminate_agent(self, agent_id: str, reason: str = "underperformance") -> None:
        """Terminate an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].status = AgentStatus.TERMINATED
            logger.warning(f"Agent {agent_id} TERMINATED | Reason: {reason}")
            del self.agents[agent_id]
    
    def suspend_agent(self, agent_id: str) -> None:
        """Temporarily suspend an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].status = AgentStatus.SUSPENDED
            logger.info(f"Agent {agent_id} SUSPENDED")
    
    def activate_agent(self, agent_id: str) -> None:
        """Activate an agent for live trading."""
        if agent_id in self.agents:
            self.agents[agent_id].status = AgentStatus.ACTIVE
            logger.info(f"Agent {agent_id} ACTIVATED")
    
    # =========================================================================
    # ACA - AGENTS CREATING AGENTS
    # =========================================================================
    
    def evaluate_agents(self) -> Dict[str, float]:
        """
        Evaluate all agents and trigger ACA if needed.
        Returns dict of agent_id -> fitness_score.
        """
        fitness_scores = {}
        
        for agent_id, agent in self.agents.items():
            fitness = agent.performance.fitness_score
            fitness_scores[agent_id] = fitness
            
            # Check if agent needs to be terminated
            if not agent.performance.is_viable and agent.performance.trades_count >= 20:
                self.terminate_agent(agent_id, "failed viability check")
        
        # Record history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "scores": fitness_scores.copy(),
            "agent_count": len(self.agents)
        })
        
        return fitness_scores
    
    def evolve_population(self, symbols: List[str]) -> List[TradingAgent]:
        """
        Evolution step: Select, mutate, and spawn new agents.
        """
        fitness_scores = self.evaluate_agents()
        new_agents = []
        
        for symbol in symbols:
            symbol_agents = [
                (aid, score) for aid, score in fitness_scores.items()
                if self.agent_configs.get(aid, AgentConfig("", AgentType.ENSEMBLE, "")).symbol == symbol
            ]
            
            if not symbol_agents:
                # No agents for this symbol - create initial population
                for agent_type in [AgentType.MOMENTUM, AgentType.MEAN_REVERSION, AgentType.ENSEMBLE]:
                    new_agent = self.create_agent(symbol, agent_type)
                    new_agents.append(new_agent)
                continue
            
            # Sort by fitness
            symbol_agents.sort(key=lambda x: x[1], reverse=True)
            
            # Elite selection - keep top performers
            n_elite = max(1, int(len(symbol_agents) * self.elite_fraction))
            elites = [aid for aid, _ in symbol_agents[:n_elite]]
            
            # Spawn children from elites
            for elite_id in elites:
                if random.random() < self.mutation_rate:
                    parent_config = self.agent_configs.get(elite_id)
                    if parent_config:
                        child = self.create_agent(symbol, parent_config.agent_type, parent_config)
                        new_agents.append(child)
                        logger.info(f"ACA: Spawned child {child.config.agent_id} from {elite_id}")
        
        return new_agents
    
    def spawn_specialist_agent(self, symbol: str, specialization: str) -> TradingAgent:
        """
        ACA: Create a specialized agent for specific market conditions.
        """
        config = AgentConfig(
            agent_id=f"{symbol}_{specialization}_{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.ENSEMBLE,
            symbol=symbol,
        )
        
        # Customize based on specialization
        if specialization == "high_volatility":
            config.use_momentum_features = False
            config.use_mean_reversion_features = True
            config.stop_loss_pct = 0.03
            config.confidence_threshold = 0.7
        elif specialization == "trending":
            config.use_momentum_features = True
            config.use_mean_reversion_features = False
            config.lookback_window = 120
        elif specialization == "news_driven":
            config.use_sentiment_features = True
            config.use_macro_features = True
            config.confidence_threshold = 0.65
        
        agent = TradingAgent(config)
        self.agents[config.agent_id] = agent
        self.agent_configs[config.agent_id] = config
        
        logger.info(f"ACA: Spawned specialist agent {config.agent_id} ({specialization})")
        return agent
    
    # =========================================================================
    # MUTATION
    # =========================================================================
    
    def _mutate_config(self, parent: AgentConfig, new_id: str) -> AgentConfig:
        """Mutate parent configuration to create child."""
        # Copy parent
        config = AgentConfig(
            agent_id=new_id,
            agent_type=parent.agent_type,
            symbol=parent.symbol,
            lookback_window=parent.lookback_window,
            prediction_horizon=parent.prediction_horizon,
            confidence_threshold=parent.confidence_threshold,
            n_estimators=parent.n_estimators,
            max_depth=parent.max_depth,
            learning_rate=parent.learning_rate,
            use_momentum_features=parent.use_momentum_features,
            use_mean_reversion_features=parent.use_mean_reversion_features,
            use_volume_features=parent.use_volume_features,
            use_sentiment_features=parent.use_sentiment_features,
            use_macro_features=parent.use_macro_features,
            max_position_pct=parent.max_position_pct,
            stop_loss_pct=parent.stop_loss_pct,
            take_profit_pct=parent.take_profit_pct,
        )
        
        # Apply mutations
        if random.random() < 0.3:
            config.lookback_window = int(parent.lookback_window * random.uniform(0.7, 1.3))
        
        if random.random() < 0.3:
            config.n_estimators = int(parent.n_estimators * random.uniform(0.8, 1.2))
        
        if random.random() < 0.3:
            config.max_depth = max(3, min(10, parent.max_depth + random.randint(-2, 2)))
        
        if random.random() < 0.3:
            config.learning_rate = parent.learning_rate * random.uniform(0.5, 2.0)
            config.learning_rate = max(0.001, min(0.1, config.learning_rate))
        
        if random.random() < 0.2:
            config.confidence_threshold = parent.confidence_threshold + random.uniform(-0.1, 0.1)
            config.confidence_threshold = max(0.5, min(0.8, config.confidence_threshold))
        
        # Feature toggles
        if random.random() < 0.1:
            config.use_sentiment_features = not parent.use_sentiment_features
        
        if random.random() < 0.1:
            config.use_macro_features = not parent.use_macro_features
        
        return config
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_config(self, config: AgentConfig) -> None:
        """Save agent config to disk."""
        filepath = self.config_path / f"{config.agent_id}.json"
        
        data = {
            "agent_id": config.agent_id,
            "agent_type": config.agent_type.value,
            "symbol": config.symbol,
            "lookback_window": config.lookback_window,
            "prediction_horizon": config.prediction_horizon,
            "confidence_threshold": config.confidence_threshold,
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "use_momentum_features": config.use_momentum_features,
            "use_mean_reversion_features": config.use_mean_reversion_features,
            "use_volume_features": config.use_volume_features,
            "use_sentiment_features": config.use_sentiment_features,
            "use_macro_features": config.use_macro_features,
            "max_position_pct": config.max_position_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "take_profit_pct": config.take_profit_pct,
            "created_at": config.created_at.isoformat(),
            "parent_id": config.parent_id,
            "generation": config.generation,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_agents(self) -> int:
        """Load saved agent configs from disk."""
        count = 0
        for filepath in self.config_path.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                config = AgentConfig(
                    agent_id=data["agent_id"],
                    agent_type=AgentType(data["agent_type"]),
                    symbol=data["symbol"],
                    lookback_window=data.get("lookback_window", 60),
                    prediction_horizon=data.get("prediction_horizon", 1),
                    confidence_threshold=data.get("confidence_threshold", 0.6),
                    n_estimators=data.get("n_estimators", 500),
                    max_depth=data.get("max_depth", 6),
                    learning_rate=data.get("learning_rate", 0.01),
                    parent_id=data.get("parent_id"),
                    generation=data.get("generation", 1),
                )
                
                agent = TradingAgent(config)
                self.agents[config.agent_id] = agent
                self.agent_configs[config.agent_id] = config
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to load agent config {filepath}: {e}")
        
        logger.info(f"Loaded {count} agent configs")
        return count
    
    # =========================================================================
    # CAPITAL ALLOCATION
    # =========================================================================
    
    def get_capital_allocation(self, total_capital: float) -> Dict[str, float]:
        """
        Allocate capital to agents based on fitness scores.
        Uses Kelly-inspired allocation.
        """
        fitness_scores = self.evaluate_agents()
        
        # Filter active agents only
        active_scores = {
            aid: score for aid, score in fitness_scores.items()
            if self.agents.get(aid) and self.agents[aid].status == AgentStatus.ACTIVE
        }
        
        if not active_scores:
            return {}
        
        # Normalize scores to sum to 1
        total_fitness = sum(active_scores.values())
        if total_fitness <= 0:
            # Equal allocation
            n = len(active_scores)
            return {aid: total_capital / n for aid in active_scores}
        
        # Pro-rata allocation
        allocations = {
            aid: (score / total_fitness) * total_capital
            for aid, score in active_scores.items()
        }
        
        return allocations
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_status_report(self) -> Dict:
        """Generate status report for all agents."""
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE),
            "training_agents": sum(1 for a in self.agents.values() if a.status == AgentStatus.TRAINING),
            "agents": [
                {
                    "agent_id": agent.config.agent_id,
                    "type": agent.config.agent_type.value,
                    "symbol": agent.config.symbol,
                    "status": agent.status.value,
                    "generation": agent.config.generation,
                    "fitness": agent.performance.fitness_score,
                    "sharpe": agent.performance.sharpe_ratio,
                    "trades": agent.performance.trades_count,
                    "pnl": agent.performance.total_pnl,
                }
                for agent in self.agents.values()
            ]
        }
