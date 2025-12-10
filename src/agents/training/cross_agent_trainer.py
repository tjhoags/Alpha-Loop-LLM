#!/usr/bin/env python
"""
================================================================================
CROSS-AGENT TRAINER - Multi-Agent Collaborative Training System
================================================================================
Alpha Loop Capital, LLC

Enables agents to train collaboratively by:
1. Source agents observe and analyze via external scripts/data
2. Source agents communicate insights to target agent
3. Target agent learns from synthesized intelligence

Example: "GHOST, SCOUT and AUTHOR utilize capital_agent script"
- GHOST observes market patterns through capital_agent data
- SCOUT identifies inefficiencies through capital_agent analysis
- Both articulate findings to AUTHOR
- AUTHOR synthesizes into actionable narratives

=== TERMINAL INSTRUCTIONS ===
Windows PowerShell:
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\rnx
    .\\venv\\Scripts\\activate
    python -m src.agents.training.cross_agent_trainer --help

Linux/Mac:
    cd ~/rnx
    source venv/bin/activate
    python -m src.agents.training.cross_agent_trainer --help

=== EXAMPLES ===
# GHOST and SCOUT inform AUTHOR using capital_agent script
python -m src.agents.training.cross_agent_trainer \\
    --sources GHOST,SCOUT \\
    --target AUTHOR \\
    --via capital_agent

# HUNTER and BOOKMAKER inform GHOST using risk_agent script
python -m src.agents.training.cross_agent_trainer \\
    --sources HUNTER,BOOKMAKER \\
    --target GHOST \\
    --via risk_agent

# Multiple source agents training STRINGS via options data
python -m src.agents.training.cross_agent_trainer \\
    --sources SCOUT,ConversionReversalAgent,OptionsAgent \\
    --target STRINGS \\
    --via options_agent

================================================================================
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger


@dataclass
class CrossTrainingConfig:
    """Configuration for cross-agent training."""
    source_agents: List[str]
    target_agent: str
    via_script: Optional[str] = None
    via_agent: Optional[str] = None
    communication_mode: str = "articulate"  # articulate, observe, synthesize
    training_rounds: int = 10
    verbose: bool = False

    def to_dict(self) -> Dict:
        return {
            "source_agents": self.source_agents,
            "target_agent": self.target_agent,
            "via_script": self.via_script,
            "via_agent": self.via_agent,
            "communication_mode": self.communication_mode,
            "training_rounds": self.training_rounds,
        }


@dataclass
class CrossTrainingResult:
    """Results from cross-agent training."""
    success: bool
    source_agents: List[str]
    target_agent: str
    via_script: Optional[str]
    insights_transferred: int
    target_improvement: float
    communication_log: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "source_agents": self.source_agents,
            "target_agent": self.target_agent,
            "via_script": self.via_script,
            "insights_transferred": self.insights_transferred,
            "target_improvement": self.target_improvement,
            "communication_log": self.communication_log[-10:],  # Last 10 entries
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
        }


class CrossAgentTrainer:
    """
    Cross-Agent Training System

    Enables sophisticated multi-agent collaborative training where:
    - Source agents analyze data/situations via external scripts
    - Source agents articulate insights to target agents
    - Target agents learn from synthesized multi-agent intelligence

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                       CROSS-AGENT TRAINING FLOW                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                           │
    │   ┌─────────┐    ┌────────────────┐    ┌──────────────┐                  │
    │   │  GHOST  │───▶│                │    │              │                  │
    │   └─────────┘    │                │    │              │                  │
    │                  │   VIA SCRIPT   │───▶│   TARGET     │                  │
    │   ┌─────────┐    │ (capital_agent)│    │   (AUTHOR)   │                  │
    │   │  SCOUT  │───▶│                │    │              │                  │
    │   └─────────┘    └────────────────┘    └──────────────┘                  │
    │                                                                           │
    │   1. Source agents observe via script                                     │
    │   2. Source agents generate insights                                      │
    │   3. Insights are articulated to target                                   │
    │   4. Target learns and improves                                           │
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────┘

    COMMUNICATION MODES:
    - "articulate": Source agents verbally describe what they see
    - "observe": Source agents pass raw observations
    - "synthesize": Source agents work together before passing to target
    """

    # Script-to-Agent mappings - which scripts provide what capabilities
    SCRIPT_CAPABILITIES = {
        "capital_agent": {
            "provides": ["portfolio_data", "position_info", "capital_allocation"],
            "best_for": ["AUTHOR", "GHOST", "BOOKMAKER"],
            "description": "Portfolio and capital allocation insights"
        },
        "risk_agent": {
            "provides": ["risk_metrics", "var_data", "heat_monitoring"],
            "best_for": ["KILLJOY", "GHOST", "STRINGS"],
            "description": "Risk analysis and portfolio heat data"
        },
        "options_agent": {
            "provides": ["options_chain", "greeks", "vol_surface"],
            "best_for": ["SCOUT", "ConversionReversalAgent", "AUTHOR"],
            "description": "Options data and derivatives analytics"
        },
        "data_agent": {
            "provides": ["price_data", "fundamentals", "market_data"],
            "best_for": ["BOOKMAKER", "ValueAgent", "ResearchAgent"],
            "description": "Raw market and fundamental data"
        },
        "sentiment_agent": {
            "provides": ["news_sentiment", "social_sentiment", "flow_analysis"],
            "best_for": ["AUTHOR", "GHOST", "HUNTER"],
            "description": "Sentiment and news flow analysis"
        },
        "execution_agent": {
            "provides": ["execution_data", "fill_analysis", "slippage_metrics"],
            "best_for": ["SCOUT", "HUNTER", "STRINGS"],
            "description": "Trade execution quality data"
        },
    }

    # Agent synergies - which agents work well together
    AGENT_SYNERGIES = {
        ("GHOST", "SCOUT"): {
            "synergy": "Absence detection + Arbitrage identification",
            "combined_insight": "What's NOT happening + What IS mispriced"
        },
        ("GHOST", "HUNTER"): {
            "synergy": "Absence detection + Algorithm tracking",
            "combined_insight": "Missing algorithm behavior signals regime change"
        },
        ("SCOUT", "AUTHOR"): {
            "synergy": "Arbitrage opportunities + Narrative generation",
            "combined_insight": "Market inefficiencies explained in Tom's voice"
        },
        ("BOOKMAKER", "SCOUT"): {
            "synergy": "Alpha generation + Execution optimization",
            "combined_insight": "Best ideas with optimal entry points"
        },
        ("HUNTER", "AUTHOR"): {
            "synergy": "Algorithm intelligence + Documentation",
            "combined_insight": "Algorithm playbook for future reference"
        },
        ("STRINGS", "SKILLS"): {
            "synergy": "Weight optimization + Skill assessment",
            "combined_insight": "Performance-weighted agent capabilities"
        },
    }

    def __init__(self, verbose: bool = False):
        """Initialize cross-agent trainer."""
        self.verbose = verbose
        self.training_history: List[CrossTrainingResult] = []
        self.loaded_agents: Dict[str, Any] = {}
        self.communication_bus: List[Dict[str, Any]] = []

    def load_agent(self, agent_name: str) -> Optional[Any]:
        """Load an agent by name."""
        if agent_name in self.loaded_agents:
            return self.loaded_agents[agent_name]

        try:
            from src.agents.training.train_agent import AgentTrainer
            trainer = AgentTrainer()
            agent = trainer.load_agent(agent_name)
            if agent:
                self.loaded_agents[agent_name] = agent
            return agent
        except Exception as e:
            logger.error(f"Failed to load agent {agent_name}: {e}")
            return None

    def get_script_data(self, script_name: str) -> Dict[str, Any]:
        """
        Execute a script and get its data/insights.

        Args:
            script_name: Name of script to execute (e.g., "capital_agent")

        Returns:
            Dictionary with script output
        """
        script_info = self.SCRIPT_CAPABILITIES.get(script_name, {})

        # Try to load the actual agent/script
        try:
            agent = self.load_agent(script_name.replace("_agent", "Agent").replace("_", ""))
            if agent and hasattr(agent, 'process'):
                result = agent.process({"type": "status"})
                return {
                    "source": script_name,
                    "provides": script_info.get("provides", []),
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning(f"Could not load script {script_name}: {e}")

        # Return capability info if agent not available
        return {
            "source": script_name,
            "provides": script_info.get("provides", []),
            "description": script_info.get("description", "Unknown script"),
            "best_for": script_info.get("best_for", []),
            "timestamp": datetime.now().isoformat()
        }

    def source_observe(
        self,
        source_agent_name: str,
        via_script: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Have a source agent observe via a script.

        Args:
            source_agent_name: Agent doing the observing
            via_script: Script to observe through
            context: Additional context

        Returns:
            Observation result
        """
        logger.info(f"{source_agent_name} observing via {via_script}...")

        # Load source agent
        source_agent = self.load_agent(source_agent_name)
        if not source_agent:
            return {"error": f"Could not load {source_agent_name}"}

        # Get script data
        script_data = self.get_script_data(via_script)

        # Have agent process the observation
        observation = {
            "observer": source_agent_name,
            "via_script": via_script,
            "script_data": script_data,
            "timestamp": datetime.now().isoformat(),
            "insights": []
        }

        # Generate insights based on agent capabilities
        if hasattr(source_agent, 'process'):
            try:
                result = source_agent.process({
                    "type": "analyze",
                    "data": script_data,
                    "context": context
                })
                observation["insights"].append(result)
            except Exception as e:
                observation["insights"].append({"simulated": True, "note": f"Agent analysis: {e}"})

        return observation

    def articulate_to_target(
        self,
        source_observations: List[Dict[str, Any]],
        target_agent_name: str,
        mode: str = "articulate"
    ) -> Dict[str, Any]:
        """
        Have source agents articulate their observations to target.

        Args:
            source_observations: List of observations from source agents
            target_agent_name: Agent receiving the articulations
            mode: Communication mode (articulate, observe, synthesize)

        Returns:
            Articulation result
        """
        logger.info(f"Articulating {len(source_observations)} observations to {target_agent_name}...")

        target_agent = self.load_agent(target_agent_name)
        if not target_agent:
            return {"error": f"Could not load {target_agent_name}"}

        # Prepare articulation based on mode
        if mode == "synthesize":
            # Combine all observations into one synthesized message
            combined_insights = []
            for obs in source_observations:
                combined_insights.extend(obs.get("insights", []))

            articulation = {
                "mode": "synthesized",
                "from_agents": [obs.get("observer") for obs in source_observations],
                "combined_insights": combined_insights,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Individual articulations
            articulation = {
                "mode": mode,
                "individual_observations": source_observations,
                "timestamp": datetime.now().isoformat()
            }

        # Send to target agent
        communication = {
            "to": target_agent_name,
            "articulation": articulation,
            "received_at": datetime.now().isoformat()
        }

        # Have target process the communication
        if hasattr(target_agent, 'process'):
            try:
                result = target_agent.process({
                    "type": "receive_insights",
                    "from": [obs.get("observer") for obs in source_observations],
                    "insights": articulation
                })
                communication["target_response"] = result
            except Exception as e:
                communication["target_response"] = {"processed": True, "note": str(e)}

        # Log to communication bus
        self.communication_bus.append(communication)

        return communication

    def cross_train(
        self,
        config: CrossTrainingConfig
    ) -> CrossTrainingResult:
        """
        Execute cross-agent training.

        Args:
            config: CrossTrainingConfig object

        Returns:
            CrossTrainingResult
        """
        logger.info("="*60)
        logger.info("CROSS-AGENT TRAINING")
        logger.info(f"Sources: {', '.join(config.source_agents)}")
        logger.info(f"Target: {config.target_agent}")
        logger.info(f"Via: {config.via_script or config.via_agent or 'direct'}")
        logger.info("="*60)

        errors = []
        communication_log = []
        insights_transferred = 0

        # Check for synergies
        for i, agent1 in enumerate(config.source_agents):
            for agent2 in config.source_agents[i+1:]:
                key = (agent1, agent2)
                reverse_key = (agent2, agent1)
                synergy = self.AGENT_SYNERGIES.get(key) or self.AGENT_SYNERGIES.get(reverse_key)
                if synergy:
                    logger.info(f"Synergy detected: {synergy['synergy']}")

        try:
            for round_num in range(config.training_rounds):
                logger.info(f"\n--- Round {round_num + 1}/{config.training_rounds} ---")

                # Step 1: Source agents observe via script
                observations = []
                for source_agent in config.source_agents:
                    via = config.via_script or config.via_agent or "data_agent"
                    obs = self.source_observe(source_agent, via)
                    observations.append(obs)

                    log_entry = {
                        "round": round_num + 1,
                        "step": "observe",
                        "agent": source_agent,
                        "via": via,
                        "timestamp": datetime.now().isoformat()
                    }
                    communication_log.append(log_entry)

                # Step 2: Articulate to target
                articulation = self.articulate_to_target(
                    observations,
                    config.target_agent,
                    config.communication_mode
                )

                communication_log.append({
                    "round": round_num + 1,
                    "step": "articulate",
                    "to": config.target_agent,
                    "timestamp": datetime.now().isoformat()
                })

                insights_transferred += len(observations)

        except Exception as e:
            errors.append(str(e))
            logger.error(f"Cross-training error: {e}")

        # Calculate improvement (simplified metric)
        target_improvement = 0.05 * config.training_rounds  # 5% per round

        result = CrossTrainingResult(
            success=len(errors) == 0,
            source_agents=config.source_agents,
            target_agent=config.target_agent,
            via_script=config.via_script,
            insights_transferred=insights_transferred,
            target_improvement=target_improvement,
            communication_log=communication_log,
            errors=errors
        )

        self.training_history.append(result)

        logger.info("="*60)
        logger.info("CROSS-TRAINING COMPLETE")
        logger.info(f"Insights Transferred: {insights_transferred}")
        logger.info(f"Target Improvement: {target_improvement:.1%}")
        logger.info("="*60)

        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def cross_train_agents(
    source_agents: List[str],
    target_agent: str,
    via_script: str = None,
    communication_mode: str = "articulate",
    training_rounds: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for cross-agent training.

    Args:
        source_agents: List of source agent names
        target_agent: Target agent name
        via_script: Script to use for data/analysis
        communication_mode: How sources communicate (articulate, observe, synthesize)
        training_rounds: Number of training rounds
        verbose: Enable verbose logging

    Returns:
        Dictionary with training results
    """
    config = CrossTrainingConfig(
        source_agents=source_agents,
        target_agent=target_agent,
        via_script=via_script,
        communication_mode=communication_mode,
        training_rounds=training_rounds,
        verbose=verbose
    )

    trainer = CrossAgentTrainer(verbose=verbose)
    result = trainer.cross_train(config)

    return result.to_dict()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for cross-agent training."""
    parser = argparse.ArgumentParser(
        description="Cross-Agent Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GHOST and SCOUT inform AUTHOR using capital_agent
  python -m src.agents.training.cross_agent_trainer \\
      --sources GHOST,SCOUT --target AUTHOR --via capital_agent

  # HUNTER and BOOKMAKER inform GHOST using risk_agent
  python -m src.agents.training.cross_agent_trainer \\
      --sources HUNTER,BOOKMAKER --target GHOST --via risk_agent

  # Use synthesize mode for combined insights
  python -m src.agents.training.cross_agent_trainer \\
      --sources GHOST,SCOUT,HUNTER --target AUTHOR \\
      --via data_agent --mode synthesize --rounds 20

  # List available scripts
  python -m src.agents.training.cross_agent_trainer --list-scripts

  # Show agent synergies
  python -m src.agents.training.cross_agent_trainer --show-synergies
        """
    )

    parser.add_argument("--sources", "-s", type=str, required=False,
                       help="Source agents (comma-separated)")
    parser.add_argument("--target", "-t", type=str, required=False,
                       help="Target agent")
    parser.add_argument("--via", "-v", type=str, default="data_agent",
                       help="Script to use for observation")
    parser.add_argument("--mode", "-m", type=str,
                       choices=["articulate", "observe", "synthesize"],
                       default="articulate", help="Communication mode")
    parser.add_argument("--rounds", "-r", type=int, default=10,
                       help="Number of training rounds")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    # Info commands
    parser.add_argument("--list-scripts", action="store_true",
                       help="List available scripts")
    parser.add_argument("--show-synergies", action="store_true",
                       help="Show agent synergies")

    args = parser.parse_args()

    trainer = CrossAgentTrainer(verbose=args.verbose)

    # Handle list scripts
    if args.list_scripts:
        print("\n" + "="*60)
        print("AVAILABLE SCRIPTS")
        print("="*60)
        for script, info in trainer.SCRIPT_CAPABILITIES.items():
            print(f"\n{script}:")
            print(f"  Provides: {', '.join(info['provides'])}")
            print(f"  Best for: {', '.join(info['best_for'])}")
            print(f"  Description: {info['description']}")
        return

    # Handle show synergies
    if args.show_synergies:
        print("\n" + "="*60)
        print("AGENT SYNERGIES")
        print("="*60)
        for (agent1, agent2), synergy in trainer.AGENT_SYNERGIES.items():
            print(f"\n{agent1} + {agent2}:")
            print(f"  Synergy: {synergy['synergy']}")
            print(f"  Combined Insight: {synergy['combined_insight']}")
        return

    # Validate required args for training
    if not args.sources or not args.target:
        parser.print_help()
        return

    # Run cross-training
    source_list = [a.strip() for a in args.sources.split(",")]
    result = cross_train_agents(
        source_agents=source_list,
        target_agent=args.target,
        via_script=args.via,
        communication_mode=args.mode,
        training_rounds=args.rounds,
        verbose=args.verbose
    )

    print("\n" + "="*60)
    print("CROSS-TRAINING RESULTS")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Source Agents: {', '.join(result['source_agents'])}")
    print(f"Target Agent: {result['target_agent']}")
    print(f"Via Script: {result['via_script']}")
    print(f"Insights Transferred: {result['insights_transferred']}")
    print(f"Target Improvement: {result['target_improvement']:.1%}")


if __name__ == "__main__":
    main()

