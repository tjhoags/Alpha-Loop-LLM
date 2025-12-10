"""
ALC-Algo Main Entry Point
Author: Tom Hogan | Alpha Loop Capital, LLC

Main orchestration script for the ALC-Algo trading platform.
Implements the Agent Coordination Architecture (ACA) with 76+ agents.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.agents import (
    GhostAgent,
    DataAgent,
    StrategyAgent,
    RiskAgent,
    ExecutionAgent,
    PortfolioAgent,
    ResearchAgent,
    ComplianceAgent,
    SentimentAgent,
    SwarmFactory,
    BlackHatAgent,
    WhiteHatAgent,
    TOTAL_AGENTS,
)
from src.core.logger import alc_logger
from config.settings import settings


def initialize_agents(user_id: str = "TJH"):
    """
    Initialize all agents in the ACA ecosystem.
    
    Args:
        user_id: User ID for audit trail
        
    Returns:
        Dictionary containing GhostAgent, senior agents, and swarm
    """
    alc_logger.info("=" * 70, user_id=user_id)
    alc_logger.info("INITIALIZING ALC-ALGO AGENT ECOSYSTEM", user_id=user_id)
    alc_logger.info(f"Expected agents: {TOTAL_AGENTS}", user_id=user_id)
    alc_logger.info("=" * 70, user_id=user_id)
    
    # Initialize Tier 1: GhostAgent (Autonomous Master Controller)
    alc_logger.info("Tier 1: Initializing GhostAgent...", user_id=user_id)
    ghost = GhostAgent(user_id=user_id)
    
    # Initialize Tier 2: Senior Agents
    alc_logger.info("Tier 2: Initializing Senior Agents...", user_id=user_id)
    senior_agents = {
        'data': DataAgent(user_id=user_id),
        'strategy': StrategyAgent(user_id=user_id),
        'risk': RiskAgent(user_id=user_id),
        'execution': ExecutionAgent(user_id=user_id),
        'portfolio': PortfolioAgent(user_id=user_id),
        'research': ResearchAgent(user_id=user_id),
        'compliance': ComplianceAgent(user_id=user_id),
        'sentiment': SentimentAgent(user_id=user_id),
    }

    # Initialize Hacker Team (Specialized Tier 2/3)
    alc_logger.info("Tier 2.5: Initializing Hacker Team (Adversarial/Defense)...", user_id=user_id)
    hacker_agents = {
        'black_hat': BlackHatAgent(user_id=user_id),
        'white_hat': WhiteHatAgent(user_id=user_id),
    }
    
    # Register senior agents with GhostAgent
    for name, agent in senior_agents.items():
        ghost.register_agent(name, agent)
        
    # Register Hacker Team
    for name, agent in hacker_agents.items():
        ghost.register_agent(name, agent)
    
    # Initialize Tier 3: Swarm Agents
    alc_logger.info("Tier 3: Initializing Swarm Agents...", user_id=user_id)
    swarm_factory = SwarmFactory(user_id=user_id)
    swarm_agents = swarm_factory.create_all_agents()
    
    # Register swarm with GhostAgent
    ghost.register_swarm(swarm_agents)
    
    total_initialized = 1 + len(senior_agents) + len(hacker_agents) + len(swarm_agents)
    alc_logger.info(f"Initialized {total_initialized} agents", user_id=user_id)
    alc_logger.info("=" * 70, user_id=user_id)
    
    return {
        'ghost': ghost,
        'senior': senior_agents,
        'hackers': hacker_agents,
        'swarm': swarm_agents,
        'swarm_factory': swarm_factory,
    }


def run_daily_workflow(agents, user_id: str = "TJH"):
    """
    Run the daily trading workflow.
    
    Args:
        agents: Dictionary of agent instances
        user_id: User ID for audit trail
    """
    ghost = agents['ghost']
    senior = agents['senior']
    hackers = agents['hackers']
    
    alc_logger.info("=" * 70, user_id=user_id)
    alc_logger.info("STARTING DAILY TRADING WORKFLOW", user_id=user_id)
    alc_logger.info("Coordinated by: GhostAgent (Tier 1 Autonomous Master)", user_id=user_id)
    alc_logger.info("=" * 70, user_id=user_id)
    
    # 0. System Stress Test (Hacker Team)
    alc_logger.info("Step 0: Pre-Flight Hack Simulation", user_id=user_id)
    attack_result = hackers['black_hat'].execute({
        'type': 'system_attack',
        'target': 'logic_layer',
        'attack_type': 'fuzzing'
    })
    
    defense_result = hackers['white_hat'].execute({
        'type': 'system_defense',
        'attack_report': attack_result
    })
    
    if defense_result.get('system_integrity', 0) < 0.99:
        alc_logger.critical("SYSTEM COMPROMISED - ABORTING WORKFLOW", user_id=user_id)
        return {'success': False, 'reason': 'Security Compromise Detected'}

    # 1. Data Ingestion
    alc_logger.info("Step 1: Data Ingestion", user_id=user_id)
    data_result = senior['data'].execute({
        'type': 'fetch_data',
        'source': 'alpha_vantage',
        'ticker': 'AAPL',
    })
    
    # 2. Swarm Analysis (Market + Sector)
    alc_logger.info("Step 2: Swarm Analysis", user_id=user_id)
    swarm_result = ghost.execute({
        'type': 'coordinate_swarm',
        'category': 'market',
        'swarm_task': {'type': 'analyze'},
    })
    
    # 3. Sentiment Analysis
    alc_logger.info("Step 3: Sentiment Analysis", user_id=user_id)
    sentiment_result = senior['sentiment'].execute({
        'type': 'analyze_sentiment',
        'ticker': 'AAPL',
    })
    
    # 4. Research & Valuation
    alc_logger.info("Step 4: Research & Valuation (HOGAN MODEL)", user_id=user_id)
    research_result = senior['research'].execute({
        'type': 'dcf_valuation',
        'ticker': 'AAPL',
    })
    
    # 5. Strategy Signal Generation
    alc_logger.info("Step 5: Strategy Signal Generation", user_id=user_id)
    strategy_result = senior['strategy'].execute({
        'type': 'generate_signal',
        'ticker': 'AAPL',
        'data': data_result,
    })
    
    # 6. Risk Assessment
    alc_logger.info("Step 6: Risk Assessment (30% MoS Check)", user_id=user_id)
    risk_result = senior['risk'].execute({
        'type': 'assess_trade',
        'ticker': 'AAPL',
        'intrinsic_value': research_result.get('intrinsic_value', 100),
        'current_price': 100,
        'position_size': 0.05,
    })
    
    # 7. GhostAgent Final Approval
    alc_logger.info("Step 7: GhostAgent Final Approval", user_id=user_id)
    ghost_result = ghost.execute({
        'type': 'coordinate_workflow',
        'workflow': 'execution',
    })
    
    # 8. Execute (if approved)
    if risk_result.get('approved', False):
        alc_logger.info("Step 8: Trade Execution (PAPER MODE)", user_id=user_id)
        execution_result = senior['execution'].execute({
            'type': 'execute_trade',
            'broker': 'ibkr',
            'ticker': 'AAPL',
            'action': 'BUY',
            'quantity': 10,
            'mode': 'PAPER',
        })
    else:
        alc_logger.warning("Trade NOT approved - insufficient margin of safety", user_id=user_id)
        execution_result = {'success': False, 'reason': 'Not approved'}
    
    # 9. Portfolio Update
    alc_logger.info("Step 9: Portfolio Update", user_id=user_id)
    portfolio_result = senior['portfolio'].execute({
        'type': 'get_positions',
    })
    
    # 10. Compliance Logging
    alc_logger.info("Step 10: Compliance Audit", user_id=user_id)
    compliance_result = senior['compliance'].execute({
        'type': 'log_action',
        'action': 'daily_workflow',
        'user_id': user_id,
        'details': {
            'approved': risk_result.get('approved', False),
            'executed': execution_result.get('success', False),
        },
    })
    
    # 11. Learning Synthesis
    alc_logger.info("Step 11: Learning Synthesis", user_id=user_id)
    learning_result = ghost.execute({
        'type': 'synthesize_learnings',
        'learnings': [
            {'category': 'strategy', 'insight': 'AAPL signal generated'},
            {'category': 'risk', 'insight': 'MoS check completed'},
            {'category': 'security', 'insight': f"Patches deployed: {defense_result.get('patches_applied', 0)}"}
        ],
    })
    
    alc_logger.info("=" * 70, user_id=user_id)
    alc_logger.info("DAILY WORKFLOW COMPLETE", user_id=user_id)
    alc_logger.info("=" * 70, user_id=user_id)
    
    return {
        'success': True,
        'steps_completed': 11,
        'approved': risk_result.get('approved', False),
        'executed': execution_result.get('success', False),
        'learnings_synthesized': learning_result.get('total_learnings', 0),
        'security_status': 'SECURE'
    }


def display_agent_stats(agents):
    """Display statistics for all agents."""
    print("\n" + "=" * 70)
    print("AGENT ECOSYSTEM STATISTICS")
    print("=" * 70)
    
    # GhostAgent stats
    ghost = agents['ghost']
    ghost_stats = ghost.get_stats()
    print(f"\n[TIER 1] {ghost_stats['name']}")
    print(f"  Mode: {ghost.mode.value}")
    print(f"  Registered Agents: {len(ghost.registered_agents)}")
    print(f"  Swarm Agents: {len(ghost.swarm_agents)}")
    print(f"  Decisions Made: {ghost.decisions_made}")
    
    # Hacker Team Stats
    print("\n[TIER 2.5] Hacker Team:")
    for name, agent in agents['hackers'].items():
        stats = agent.get_stats()
        print(f"  {stats['name']:20} | Status: {stats['status']}")
        
    # Senior Agent stats
    print("\n[TIER 2] Senior Agents:")
    for name, agent in agents['senior'].items():
        stats = agent.get_stats()
        print(f"  {stats['name']:20} | Executions: {stats['execution_count']:3} | Success: {stats['success_rate']}")
    
    # Swarm stats
    swarm_stats = agents['swarm_factory'].get_stats()
    print(f"\n[TIER 3] Swarm Agents: {swarm_stats['total_agents']} total")
    for category, count in swarm_stats['by_category'].items():
        print(f"  {category.capitalize():15} | {count} agents")
    print(f"  Total Signals: {swarm_stats['total_signals_generated']}")
    print(f"  Total Analyses: {swarm_stats['total_analyses_completed']}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("ALC-ALGO - ALGORITHMIC TRADING PLATFORM")
    print("Agent Coordination Architecture (ACA)")
    print("Author: Tom Hogan | Alpha Loop Capital, LLC")
    print("=" * 70 + "\n")
    
    print("Hierarchy:")
    print("  Tier 0: HOAGS (Human Oversight - Tom Hogan)")
    print("  Tier 1: GhostAgent (Autonomous Master Controller)")
    print("  Tier 2: 8 Senior Agents")
    print("  Tier 2.5: Hacker Team (BlackHat + WhiteHat)")
    print("  Tier 3: 65+ Swarm Agents")
    print("")
    
    # Initialize agents
    agents = initialize_agents(user_id="TJH")
    
    # Run daily workflow
    result = run_daily_workflow(agents, user_id="TJH")
    
    # Display stats
    display_agent_stats(agents)
    
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"  Workflow Steps Completed: {result['steps_completed']}")
    print(f"  Trade Approved: {result['approved']}")
    print(f"  Trade Executed: {result['executed']}")
    print(f"  Learnings Synthesized: {result['learnings_synthesized']}")
    print(f"  Security Status: {result.get('security_status', 'UNKNOWN')}")
    print("")
    print("  All actions logged to audit trail")
    print("  All outputs attributed to: Tom Hogan")
    print("  DCF methodology: HOGAN MODEL")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
