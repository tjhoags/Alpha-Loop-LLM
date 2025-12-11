"""
Launch a Venture Division via ACA (Agent Creating Agents).

This script proposes and approves new agents for a software + product/marketing
team plus an advanced business development lead, then instantiates lightweight
agent stubs so capabilities are registered in the ACA registry.

Usage (from repo root):
    python scripts/launch_venture_division.py
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.aca_engine import get_aca_engine
from src.core.agent_base import AgentStatus, AgentTier, BaseAgent


def build_venture_agent(agent_name: str, tier_name: str, capabilities: List[str]):
    """
    Create a minimal venture agent class that satisfies BaseAgent requirements.

    create_agent_from_proposal() expects a class accepting user_id; this factory
    captures the agent_name/tier/capabilities for instantiation.
    """
    tier_enum = AgentTier[tier_name] if tier_name in AgentTier.__members__ else AgentTier.SENIOR

    class VentureAgent(BaseAgent):
        def __init__(self, user_id: str = "TJH"):
            super().__init__(
                name=agent_name,
                tier=tier_enum,
                capabilities=capabilities,
                user_id=user_id,
                aca_enabled=False,
                learning_enabled=False,
            )
            self.status = AgentStatus.ACTIVE

        def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "success": True,
                "agent": agent_name,
                "received_task": task,
                "message": f"{agent_name} is initialized and ready for routing.",
            }

        def get_capabilities(self) -> List[str]:
            return self.capabilities

    return VentureAgent


def submit_and_create(engine, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Submit, approve, and instantiate an agent based on a spec."""
    proposal_id = spec.get("proposal_id") or f"VENTURE_{uuid.uuid4().hex[:8]}"

    proposal = engine.submit_proposal(
        proposal_id=proposal_id,
        proposed_by=spec["proposed_by"],
        agent_name=spec["name"],
        agent_tier=spec["tier"],
        capabilities=spec["capabilities"],
        gap_description=spec["gap"],
        rationale=spec["rationale"],
        priority=spec["priority"],
    )

    engine.approve_proposal(proposal_id=proposal_id, approver=spec.get("approver", "HoagsAgent"))

    agent_class = build_venture_agent(spec["name"], spec["tier"], spec["capabilities"])
    created = engine.create_agent_from_proposal(proposal_id=proposal_id, agent_class=agent_class)

    return {
        "proposal": proposal.to_dict(),
        "created": created is not None,
        "agent_name": spec["name"],
    }


def main():
    logging.basicConfig(level=logging.INFO)
    engine = get_aca_engine()

    specs: List[Dict[str, Any]] = [
        {
            "name": "BACKEND_LEAD",
            "tier": "SENIOR",
            "capabilities": [
                "api_architecture",
                "service_orchestration",
                "database_design",
                "ci_cd_pipeline",
                "observability",
            ],
            "gap": "Need backend leadership for new B2C launches.",
            "rationale": "Owns backend delivery speed and reliability.",
            "priority": "high",
            "proposed_by": "GHOST",
            "approver": "HoagsAgent",
        },
        {
            "name": "FRONTEND_LEAD",
            "tier": "SENIOR",
            "capabilities": [
                "design_system",
                "responsive_ui",
                "a11y",
                "web_performance",
                "frontend_observability",
            ],
            "gap": "Need rapid UI/UX shipping for consumer pilots.",
            "rationale": "Owns frontend velocity and UX quality.",
            "priority": "high",
            "proposed_by": "GHOST",
            "approver": "HoagsAgent",
        },
        {
            "name": "PRODUCT_OWNER",
            "tier": "SENIOR",
            "capabilities": [
                "roadmap_prioritization",
                "user_research",
                "spec_writing",
                "launch_checklists",
            ],
            "gap": "Need product owner to translate market signals into shippable specs.",
            "rationale": "Bridges market demand to engineering output.",
            "priority": "high",
            "proposed_by": "ORCHESTRATOR",
            "approver": "HoagsAgent",
        },
        {
            "name": "MARKETING_MAVEN",
            "tier": "SENIOR",
            "capabilities": [
                "growth_loops",
                "paid_channels",
                "seo_sem",
                "landing_page_conversion",
                "community_building",
            ],
            "gap": "Need GTM engine to validate demand and drive signups tonight.",
            "rationale": "Owns acquisition and activation experiments.",
            "priority": "high",
            "proposed_by": "ORCHESTRATOR",
            "approver": "HoagsAgent",
        },
        {
            "name": "RAINMAKER",
            "tier": "SENIOR",
            "capabilities": [
                "enterprise_outreach",
                "deal_sourcing",
                "partnership_structuring",
                "competitive_intel",
                "market_gap_detection",
            ],
            "gap": "Need advanced business development pointing to markets needing solutions.",
            "rationale": "Surfaces high-value market gaps and partners.",
            "priority": "critical",
            "proposed_by": "NOBUS",
            "approver": "HoagsAgent",
        },
    ]

    results = [submit_and_create(engine, spec) for spec in specs]

    print("\n=== Venture Division Launch Summary ===")
    for result in results:
        status = "CREATED" if result["created"] else "PENDING_CLASS"
        print(f"- {result['agent_name']}: {status}")
    print("\nACA Status Snapshot:", engine.get_status())


if __name__ == "__main__":
    main()


