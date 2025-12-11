"""
Launch the flash team: submit, approve, and instantiate ACA proposals
for the five specialized agents needed for fast product launches.
"""

from src.core.aca_engine import get_aca_engine
from src.agents.software_agent.software_agent import SoftwareAgent
from src.agents.marketing_agent.marketing_agent import MarketingAgent
from src.agents.strategy_agent.strategy_agent import StrategyAgent
from src.agents.research_agent.research_agent import ResearchAgent
from src.agents.nobus_agent.nobus_agent import NOBUSAgent


def main() -> None:
    aca = get_aca_engine()

    proposals = [
        {
            "proposal_id": "aca_backend_arch",
            "name": "Backend_Architect",
            "tier": "SENIOR",
            "capabilities": [
                "api_design",
                "auth",
                "payments",
                "observability",
                "async_execution",
            ],
            "gap": "Need fast, secure APIs for launch",
            "rationale": "Flash launch backend coverage",
            "agent_class": SoftwareAgent,
        },
        {
            "proposal_id": "aca_frontend_virt",
            "name": "Frontend_Virtuoso",
            "tier": "SENIOR",
            "capabilities": [
                "ui_build",
                "landing_pages",
                "a/b_testing",
                "component_theming",
            ],
            "gap": "Need rapid UI/UX deployment",
            "rationale": "Flash launch frontend coverage",
            "agent_class": SoftwareAgent,
        },
        {
            "proposal_id": "aca_growth_hack",
            "name": "Growth_Hacker",
            "tier": "SENIOR",
            "capabilities": [
                "viral_loops",
                "referrals",
                "paid_social",
                "conversion_copy",
            ],
            "gap": "Need aggressive acquisition engine",
            "rationale": "Growth loop activation",
            "agent_class": MarketingAgent,
        },
        {
            "proposal_id": "aca_product_owner",
            "name": "Product_Owner",
            "tier": "SENIOR",
            "capabilities": [
                "mvp_scoping",
                "backlog",
                "qa_signoff",
                "release_planning",
            ],
            "gap": "Need ruthless scope control",
            "rationale": "Keep launch shippable tonight",
            "agent_class": StrategyAgent,
        },
        {
            "proposal_id": "aca_market_seer",
            "name": "Market_Seer",
            "tier": "SENIOR",
            "capabilities": [
                "intel_gathering",
                "void_detection",
                "pricing_signals",
                "lead_lists",
            ],
            "gap": "Need market void radar",
            "rationale": "Find underserved niches fast",
            "agent_class": ResearchAgent,
        },
    ]

    for spec in proposals:
        aca.submit_proposal(
            proposal_id=spec["proposal_id"],
            proposed_by="ACA_ORCHESTRATOR",
            agent_name=spec["name"],
            agent_tier=spec["tier"],
            capabilities=spec["capabilities"],
            gap_description=spec["gap"],
            rationale=spec["rationale"],
            priority="high",
        )
        aca.approve_proposal(spec["proposal_id"], "HOAGS")
        created = aca.create_agent_from_proposal(
            spec["proposal_id"], agent_class=spec["agent_class"]
        )
        print(f"[ACA] Created {spec['name']}: {bool(created)}")

    print("[ACA] Flash team ready.")


if __name__ == "__main__":
    main()


