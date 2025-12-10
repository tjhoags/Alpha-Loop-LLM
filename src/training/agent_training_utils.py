"""
================================================================================
AGENT TRAINING UTILITIES - Universal Training Framework
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

This module provides universal training utilities for all agents in the
ALC-Algo ecosystem. It supports:

1. Individual agent training
2. Specific sets of agents training together
3. Random agent combinations training
4. Cross-training with external ML scripts

USAGE EXAMPLES:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    .\\venv\\Scripts\\activate

    # Train single agent:
    python -m src.training.agent_training_utils --agent GHOST

    # Train multiple specific agents:
    python -m src.training.agent_training_utils --agents GHOST,SCOUT,AUTHOR

    # Train random combination of N agents:
    python -m src.training.agent_training_utils --random 3

    # Cross-train: agents utilize external script:
    # Format: "AGENT1,AGENT2:SYNTHESIS_AGENT:script_name"
    python -m src.training.agent_training_utils --cross-train "GHOST,SCOUT:AUTHOR:capital_agent"

    # Train all agents:
    python -m src.training.agent_training_utils --all

    # Train by tier:
    python -m src.training.agent_training_utils --tier SENIOR

    # List all available agents:
    python -m src.training.agent_training_utils --list

================================================================================
"""

import argparse
import importlib
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# AGENT REGISTRY - All trainable agents
# ============================================================================

AGENT_REGISTRY = {
    # MASTER TIER (Tier 1)
    "GHOST": {
        "module": "src.agents.ghost_agent.ghost_agent",
        "class": "GhostAgent",
        "tier": "MASTER",
        "description": "Autonomous Master Controller - coordinates all agents",
    },
    "HOAGS": {
        "module": "src.agents.hoags_agent.hoags_agent",
        "class": "HoagsAgent",
        "tier": "MASTER",
        "description": "Tom Hogan's direct authority - final decision maker",
    },

    # SENIOR TIER (Tier 2)
    "AUTHOR": {
        "module": "src.agents.senior.author_agent",
        "class": "THE_AUTHOR",
        "tier": "SENIOR",
        "description": "Natural language generation in Tom's voice",
    },
    "BOOKMAKER": {
        "module": "src.agents.senior.bookmaker_agent",
        "class": "BOOKMAKER",
        "tier": "SENIOR",
        "description": "Alpha generation and valuation tactics",
    },
    "HUNTER": {
        "module": "src.agents.senior.hunter_agent",
        "class": "HUNTER",
        "tier": "SENIOR",
        "description": "Algorithm intelligence and counter-strategies",
    },
    "SCOUT": {
        "module": "src.agents.senior.scout_agent",
        "class": "SCOUT",
        "tier": "SENIOR",
        "description": "Market inefficiency hunter - retail arbitrage",
    },
    "SKILLS": {
        "module": "src.agents.senior.skills_agent",
        "class": "SKILLS",
        "tier": "SENIOR",
        "description": "NLP instruction interpreter and skill assessor",
    },
    "STRINGS": {
        "module": "src.agents.senior.strings_agent",
        "class": "STRINGS",
        "tier": "SENIOR",
        "description": "ML training and weight optimization",
    },
    "ORCHESTRATOR": {
        "module": "src.agents.orchestrator_agent.orchestrator_agent",
        "class": "ORCHESTRATOR",
        "tier": "SENIOR",
        "description": "Creative task coordination and agent improvement",
    },
    "KILLJOY": {
        "module": "src.agents.killjoy_agent.killjoy_agent",
        "class": "KILLJOY",
        "tier": "SENIOR",
        "description": "Capital allocation and risk guardrails",
    },
    "DATA_AGENT": {
        "module": "src.agents.data_agent.data_agent",
        "class": "DataAgent",
        "tier": "SENIOR",
        "description": "Data ingestion and normalization",
    },
    "EXECUTION_AGENT": {
        "module": "src.agents.execution_agent.execution_agent",
        "class": "ExecutionAgent",
        "tier": "SENIOR",
        "description": "Trade execution via brokers",
    },
    "PORTFOLIO_AGENT": {
        "module": "src.agents.portfolio_agent.portfolio_agent",
        "class": "PortfolioAgent",
        "tier": "SENIOR",
        "description": "Portfolio management and rebalancing",
    },
    "RISK_AGENT": {
        "module": "src.agents.risk_agent.risk_agent",
        "class": "RiskAgent",
        "tier": "SENIOR",
        "description": "Risk assessment and margin of safety",
    },
    "COMPLIANCE_AGENT": {
        "module": "src.agents.compliance_agent.compliance_agent",
        "class": "ComplianceAgent",
        "tier": "SENIOR",
        "description": "Audit trail and compliance enforcement",
    },
    "RESEARCH_AGENT": {
        "module": "src.agents.research_agent.research_agent",
        "class": "ResearchAgent",
        "tier": "SENIOR",
        "description": "Fundamental and macro analysis",
    },
    "SENTIMENT_AGENT": {
        "module": "src.agents.sentiment_agent.sentiment_agent",
        "class": "SentimentAgent",
        "tier": "SENIOR",
        "description": "Market sentiment analysis",
    },

    # SECURITY TIER
    "BLACKHAT": {
        "module": "src.agents.hackers.black_hat",
        "class": "BlackHatAgent",
        "tier": "SENIOR",
        "description": "Internal adversary - finds vulnerabilities",
    },
    "WHITEHAT": {
        "module": "src.agents.hackers.white_hat",
        "class": "WhiteHatAgent",
        "tier": "SENIOR",
        "description": "Guardian - patches vulnerabilities",
    },

    # MASTER TIER - NOBUS (Top Tier Chaos Engineering)
    "NOBUS": {
        "module": "src.agents.nobus_agent.nobus_agent",
        "class": "NOBUSAgent",
        "tier": "MASTER",
        "description": "TOP TIER - System resilience & chaos engineering",
    },

    # STRATEGY TIER (Specialized)
    "CONVERSION_REVERSAL": {
        "module": "src.agents.specialized.conversion_reversal_agent",
        "class": "ConversionReversalAgent",
        "tier": "STRATEGY",
        "description": "Options arbitrage - conversions/reversals",
    },
    "MOMENTUM": {
        "module": "src.agents.specialized.momentum_agent",
        "class": "MomentumAgent",
        "tier": "STRATEGY",
        "description": "Momentum strategy agent",
    },
    "VALUE": {
        "module": "src.agents.specialized.value_agent",
        "class": "ValueAgent",
        "tier": "STRATEGY",
        "description": "Value investing strategy agent",
    },
    "MEAN_REVERSION": {
        "module": "src.agents.specialized.mean_reversion_agent",
        "class": "MeanReversionAgent",
        "tier": "STRATEGY",
        "description": "Mean reversion strategy agent",
    },
}

# ============================================================================
# TRAINING SCRIPTS REGISTRY - External ML scripts
# ============================================================================

ML_SCRIPTS = {
    "capital_agent": {
        "module": "src.agents.senior.capital_agent",
        "description": "Capital allocation optimization script",
    },
    "agent_trainer": {
        "module": "src.training.agent_trainer",
        "description": "Main agent trainer with backtesting",
    },
    "train_small_mid_cap": {
        "module": "scripts.train_small_mid_cap",
        "description": "Small/mid cap model training",
    },
}


# ============================================================================
# AGENT LOADER
# ============================================================================

def load_agent(agent_name: str) -> Optional[Any]:
    """
    Load an agent instance by name.

    Args:
        agent_name: Name of the agent (e.g., "GHOST", "SCOUT")

    Returns:
        Agent instance or None if not found
    """
    agent_name = agent_name.upper()

    if agent_name not in AGENT_REGISTRY:
        print(f"[ERROR] Unknown agent: {agent_name}")
        return None

    agent_info = AGENT_REGISTRY[agent_name]

    try:
        module = importlib.import_module(agent_info["module"])
        agent_class = getattr(module, agent_info["class"])
        agent = agent_class()
        print(f"[LOADED] {agent_name}: {agent_info['description']}")
        return agent
    except Exception as e:
        print(f"[ERROR] Failed to load {agent_name}: {e}")
        return None


def load_ml_script(script_name: str) -> Optional[Any]:
    """
    Load an ML script module by name.

    Args:
        script_name: Name of the script (e.g., "capital_agent")

    Returns:
        Module or None if not found
    """
    if script_name not in ML_SCRIPTS:
        print(f"[ERROR] Unknown ML script: {script_name}")
        return None

    script_info = ML_SCRIPTS[script_name]

    try:
        module = importlib.import_module(script_info["module"])
        print(f"[LOADED] Script {script_name}: {script_info['description']}")
        return module
    except Exception as e:
        print(f"[ERROR] Failed to load script {script_name}: {e}")
        return None


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_single_agent(agent_name: str, epochs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Train a single agent.

    Args:
        agent_name: Name of the agent to train
        epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Training results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING AGENT: {agent_name}")
        print(f"{'='*60}\n")

    agent = load_agent(agent_name)
    if agent is None:
        return {"success": False, "error": f"Failed to load {agent_name}"}

    # Run agent-specific training if available
    results = {
        "agent": agent_name,
        "start_time": datetime.now().isoformat(),
        "epochs": epochs,
    }

    # Check for train method
    if hasattr(agent, "train"):
        try:
            train_result = agent.train(epochs=epochs)
            results["train_result"] = train_result
        except Exception as e:
            results["train_error"] = str(e)

    # Check for fit method (sklearn-style)
    if hasattr(agent, "fit"):
        try:
            # Load training data if available
            results["fit_available"] = True
        except Exception as e:
            results["fit_error"] = str(e)

    results["end_time"] = datetime.now().isoformat()
    results["success"] = True

    if verbose:
        print(f"[COMPLETE] {agent_name} training finished")

    return results


def train_multiple_agents(agent_names: List[str], epochs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Train multiple agents together.

    Args:
        agent_names: List of agent names to train
        epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Training results for all agents
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING {len(agent_names)} AGENTS TOGETHER")
        print(f"Agents: {', '.join(agent_names)}")
        print(f"{'='*60}\n")

    results = {
        "mode": "multi_agent",
        "agents": agent_names,
        "start_time": datetime.now().isoformat(),
        "agent_results": {},
    }

    # Load all agents
    agents = {}
    for name in agent_names:
        agent = load_agent(name)
        if agent:
            agents[name] = agent

    # Train each agent
    for name, agent in agents.items():
        result = train_single_agent(name, epochs=epochs, verbose=verbose)
        results["agent_results"][name] = result

    results["end_time"] = datetime.now().isoformat()
    results["success"] = len(agents) == len(agent_names)

    return results


def train_random_agents(count: int, epochs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Train a random combination of agents.

    Args:
        count: Number of agents to randomly select
        epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Training results
    """
    available = list(AGENT_REGISTRY.keys())
    count = min(count, len(available))
    selected = random.sample(available, count)

    if verbose:
        print(f"\n{'='*60}")
        print("RANDOM AGENT TRAINING")
        print(f"Randomly selected {count} agents: {', '.join(selected)}")
        print(f"{'='*60}\n")

    results = train_multiple_agents(selected, epochs=epochs, verbose=verbose)
    results["mode"] = "random"
    results["random_seed"] = random.getstate()[1][0]

    return results


def cross_train(
    source_agents: List[str],
    synthesis_agent: str,
    ml_script: str,
    epochs: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Cross-train agents using an external ML script.

    This enables scenarios like:
    "GHOST, SCOUT and AUTHOR utilize capital_agent script"

    Where GHOST and SCOUT articulate observations to AUTHOR,
    processed through the capital_agent script.

    Args:
        source_agents: Agents that provide observations (e.g., GHOST, SCOUT)
        synthesis_agent: Agent that synthesizes output (e.g., AUTHOR)
        ml_script: External ML script to use (e.g., capital_agent)
        epochs: Training epochs
        verbose: Print progress

    Returns:
        Cross-training results
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-TRAINING WITH ML SCRIPT")
        print(f"Source agents: {', '.join(source_agents)}")
        print(f"Synthesis agent: {synthesis_agent}")
        print(f"ML Script: {ml_script}")
        print(f"{'='*60}\n")

    results = {
        "mode": "cross_train",
        "source_agents": source_agents,
        "synthesis_agent": synthesis_agent,
        "ml_script": ml_script,
        "start_time": datetime.now().isoformat(),
    }

    # Load source agents
    sources = {}
    for name in source_agents:
        agent = load_agent(name)
        if agent:
            sources[name] = agent

    # Load synthesis agent
    synthesizer = load_agent(synthesis_agent)

    # Load ML script
    script_module = load_ml_script(ml_script)

    if not sources or not synthesizer:
        results["success"] = False
        results["error"] = "Failed to load required agents"
        return results

    # PHASE 1: Source agents generate observations
    observations = {}
    if verbose:
        print("\n[PHASE 1] Source agents generating observations...")

    for name, agent in sources.items():
        if hasattr(agent, "process"):
            obs = agent.process({"type": "generate_observations"})
            observations[name] = obs
            if verbose:
                print(f"  {name}: Generated observation")

    # PHASE 2: Process through ML script
    processed = {}
    if script_module and verbose:
        print(f"\n[PHASE 2] Processing through {ml_script}...")

        # If script has a process function, use it
        if hasattr(script_module, "process"):
            processed = script_module.process(observations)
        else:
            processed = {"raw_observations": observations}

    # PHASE 3: Synthesis agent creates output
    if verbose:
        print(f"\n[PHASE 3] {synthesis_agent} synthesizing output...")

    if hasattr(synthesizer, "process"):
        synthesis_result = synthesizer.process({
            "type": "synthesize",
            "observations": observations,
            "processed": processed,
        })
        results["synthesis_result"] = synthesis_result

    results["observations"] = observations
    results["processed"] = processed
    results["end_time"] = datetime.now().isoformat()
    results["success"] = True

    if verbose:
        print("\n[COMPLETE] Cross-training finished")

    return results


def train_by_tier(tier: str, epochs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Train all agents in a specific tier.

    Args:
        tier: Tier name (MASTER, SENIOR, SUPPORT, STRATEGY)
        epochs: Training epochs
        verbose: Print progress

    Returns:
        Training results
    """
    tier = tier.upper()
    agents_in_tier = [
        name for name, info in AGENT_REGISTRY.items()
        if info["tier"] == tier
    ]

    if not agents_in_tier:
        return {"success": False, "error": f"No agents in tier: {tier}"}

    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING TIER: {tier}")
        print(f"Agents in tier: {', '.join(agents_in_tier)}")
        print(f"{'='*60}\n")

    results = train_multiple_agents(agents_in_tier, epochs=epochs, verbose=verbose)
    results["tier"] = tier

    return results


def train_all_agents(epochs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Train all registered agents.

    Args:
        epochs: Training epochs
        verbose: Print progress

    Returns:
        Training results
    """
    all_agents = list(AGENT_REGISTRY.keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING ALL {len(all_agents)} AGENTS")
        print(f"{'='*60}\n")

    results = train_multiple_agents(all_agents, epochs=epochs, verbose=verbose)
    results["mode"] = "all"

    return results


def list_agents() -> None:
    """Print all available agents organized by tier."""
    print("\n" + "="*70)
    print("AVAILABLE AGENTS")
    print("="*70)

    # Group by tier
    tiers = {}
    for name, info in AGENT_REGISTRY.items():
        tier = info["tier"]
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append((name, info["description"]))

    # Print by tier
    tier_order = ["MASTER", "SENIOR", "STRATEGY", "SUPPORT"]
    for tier in tier_order:
        if tier in tiers:
            print(f"\n{tier} TIER:")
            print("-" * 50)
            for name, desc in sorted(tiers[tier]):
                print(f"  {name:25s} - {desc}")

    print("\n" + "="*70)
    print(f"Total: {len(AGENT_REGISTRY)} agents")
    print("="*70)

    print("\nML SCRIPTS:")
    print("-" * 50)
    for name, info in ML_SCRIPTS.items():
        print(f"  {name:25s} - {info['description']}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_cross_train(spec: str) -> Tuple[List[str], str, str]:
    """
    Parse cross-train specification.

    Format: "AGENT1,AGENT2:SYNTHESIS_AGENT:script_name"

    Example: "GHOST,SCOUT:AUTHOR:capital_agent"
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            "Cross-train format: AGENT1,AGENT2:SYNTHESIS_AGENT:script_name"
        )

    source_agents = [a.strip() for a in parts[0].split(",")]
    synthesis_agent = parts[1].strip()
    ml_script = parts[2].strip()

    return source_agents, synthesis_agent, ml_script


def main():
    """Main entry point for agent training."""
    parser = argparse.ArgumentParser(
        description="Alpha Loop Capital Agent Training Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train single agent:
    python -m src.training.agent_training_utils --agent GHOST

  Train multiple agents:
    python -m src.training.agent_training_utils --agents GHOST,SCOUT,AUTHOR

  Train random agents:
    python -m src.training.agent_training_utils --random 3

  Cross-train (GHOST & SCOUT articulate to AUTHOR via capital_agent):
    python -m src.training.agent_training_utils --cross-train "GHOST,SCOUT:AUTHOR:capital_agent"

  Train by tier:
    python -m src.training.agent_training_utils --tier SENIOR

  Train all:
    python -m src.training.agent_training_utils --all

  List agents:
    python -m src.training.agent_training_utils --list
        """
    )

    parser.add_argument(
        "--agent",
        type=str,
        help="Train a single agent by name"
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="Train multiple agents (comma-separated)"
    )
    parser.add_argument(
        "--random",
        type=int,
        help="Train N randomly selected agents"
    )
    parser.add_argument(
        "--cross-train",
        type=str,
        dest="cross_train",
        help="Cross-train: 'SOURCE_AGENTS:SYNTHESIS_AGENT:script'"
    )
    parser.add_argument(
        "--tier",
        type=str,
        help="Train all agents in a tier (MASTER/SENIOR/STRATEGY/SUPPORT)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all agents"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available agents"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()
    verbose = not args.quiet
    results = None

    # Header
    if verbose:
        print("\n" + "="*70)
        print("ALPHA LOOP CAPITAL - AGENT TRAINING UTILITIES")
        print("Author: Tom Hogan | Alpha Loop Capital, LLC")
        print("="*70)

    # Execute requested action
    if args.list:
        list_agents()
    elif args.agent:
        results = train_single_agent(args.agent, epochs=args.epochs, verbose=verbose)
    elif args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]
        results = train_multiple_agents(agent_names, epochs=args.epochs, verbose=verbose)
    elif args.random:
        results = train_random_agents(args.random, epochs=args.epochs, verbose=verbose)
    elif args.cross_train:
        source_agents, synthesis_agent, ml_script = parse_cross_train(args.cross_train)
        results = cross_train(
            source_agents, synthesis_agent, ml_script,
            epochs=args.epochs, verbose=verbose
        )
    elif args.tier:
        results = train_by_tier(args.tier, epochs=args.epochs, verbose=verbose)
    elif args.all:
        results = train_all_agents(epochs=args.epochs, verbose=verbose)
    else:
        parser.print_help()

    # Save results if requested
    if results and args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[SAVED] Results written to {output_path}")

    # Print summary
    if results and verbose:
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Success: {results.get('success', 'Unknown')}")
        if "agent_results" in results:
            print(f"Agents trained: {len(results['agent_results'])}")
        print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
